"""Config-driven TPU v6e-1 curriculum trainer with Lookahead Time-Efficiency switching."""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from huggingface_hub import hf_hub_download
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tokenizers import Tokenizer

from loopmoe.checkpoint import load_latest_checkpoint, save_checkpoint
from loopmoe.config import LoopMoEConfig, StageConfig, estimate_step_flops
from loopmoe.data import ShardBatchPrefetcher, StreamingTokenPrefetcher
from loopmoe.lookahead import (
    ExplicitStageConfig,
    LookaheadTracker,
    TrainFileConfig,
    load_train_config,
)
from loopmoe.model import (
    count_tree_params,
    expand_checkpoint_1_to_n_loops,
    forward_model,
    init_model_params,
    summarize_model,
)
from loopmoe.moe import update_router_biases
from loopmoe.optim import create_muon_adamw_optimizer


class SyntheticBatchPrefetcher:
    """Zero-I/O synthetic token generator for local/TPU smoke runs."""

    def __init__(self, micro_bs: int, seq_len: int, vocab_size: int, seed: int = 42):
        self.micro_bs = micro_bs
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.rng.integers(64, self.vocab_size, size=(self.micro_bs, self.seq_len), dtype=np.uint16)
        y = np.roll(x, -1, axis=1)
        return x, y

    def close(self) -> None:
        return None


@dataclass
class _CompiledStageRuntime:
    stage_idx: int
    explicit_cfg: ExplicitStageConfig
    stage_cfg: StageConfig
    model_cfg: LoopMoEConfig
    compiled: Any
    step_flops: float
    loader: Any


def _extract_opt_step(opt_state: Any) -> jax.Array:
    int_leaves = [
        v
        for v in jax.tree_util.tree_leaves(opt_state)
        if hasattr(v, "ndim") and v.ndim == 0 and jnp.issubdtype(v.dtype, jnp.integer)
    ]
    return int_leaves[0].astype(jnp.int32) if int_leaves else jnp.asarray(0, dtype=jnp.int32)


def make_train_step(
    model_cfg: LoopMoEConfig,
    optimizer: optax.GradientTransformation,
    grad_accum_steps: int,
    seed: int = 42,
):
    """Build a donated JIT-compiled global optimizer step over `grad_accum_steps` microbatches."""

    def _loss_fn(params: dict[str, Any], x: jax.Array, y: jax.Array, rng_key: jax.Array | None):
        return forward_model(params, x, y, model_cfg, rng_key=rng_key)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

    @jax.jit(donate_argnums=(0, 1))
    def train_step(
        params: dict[str, Any],
        opt_state: Any,
        batch_xs: jax.Array,
        batch_ys: jax.Array,
    ) -> tuple[dict[str, Any], Any, dict[str, jax.Array]]:
        step_idx = _extract_opt_step(opt_state)
        base_rng = jax.random.fold_in(jax.random.PRNGKey(seed), step_idx)
        default_exp_r = jnp.asarray(float(model_cfg.num_loops), dtype=jnp.float32)
        default_h_ent = jnp.asarray(0.0, dtype=jnp.float32)

        if grad_accum_steps == 1:
            mb_key = (
                jax.random.fold_in(base_rng, jnp.asarray(0, dtype=jnp.int32))
                if model_cfg.num_loops > 1
                else None
            )
            with jax.named_scope("fwd_bwd_microbatch"):
                (loss_mean, aux), grads_mean = grad_fn(
                    params, batch_xs[0].astype(jnp.int32), batch_ys[0].astype(jnp.int32), mb_key
                )
            ce_mean = aux["ce_loss"]
            gate_mean = aux["engram_gate_mean"]
            mmr_mean = aux["router_mmr"]
            exp_r_mean = aux.get("expected_loops", default_exp_r)
            h_ent_mean = aux.get("halt_entropy", default_h_ent)
            counts_mean = aux["expert_counts"]
        else:
            zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
            init_counts = jnp.zeros(
                (model_cfg.num_loops, model_cfg.n_routed_experts), dtype=jnp.float32
            )
            init_carry = (
                zero_grads,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                init_counts,
            )
            inv_g = 1.0 / float(grad_accum_steps)

            def accum_body(carry, microbatch):
                g_acc, loss_acc, ce_acc, gate_acc, mmr_acc, exp_r_acc, h_ent_acc, cnt_acc = carry
                x_mb, y_mb, mb_idx = microbatch
                x_mb = x_mb.astype(jnp.int32)
                y_mb = y_mb.astype(jnp.int32)
                mb_key = jax.random.fold_in(base_rng, mb_idx) if model_cfg.num_loops > 1 else None
                with jax.named_scope("fwd_bwd_microbatch"):
                    (loss, aux), grads = grad_fn(params, x_mb, y_mb, mb_key)
                g_acc = jax.tree_util.tree_map(
                    lambda a, g: a + g * jnp.asarray(inv_g, dtype=g.dtype), g_acc, grads
                )
                return (
                    g_acc,
                    loss_acc + loss * inv_g,
                    ce_acc + aux["ce_loss"] * inv_g,
                    gate_acc + aux["engram_gate_mean"] * inv_g,
                    mmr_acc + aux["router_mmr"] * inv_g,
                    exp_r_acc + aux.get("expected_loops", default_exp_r) * inv_g,
                    h_ent_acc + aux.get("halt_entropy", default_h_ent) * inv_g,
                    cnt_acc + aux["expert_counts"] * inv_g,
                ), None

            mb_indices = jnp.arange(grad_accum_steps, dtype=jnp.int32)
            with jax.named_scope("grad_accum_scan"):
                (
                    grads_mean,
                    loss_mean,
                    ce_mean,
                    gate_mean,
                    mmr_mean,
                    exp_r_mean,
                    h_ent_mean,
                    counts_mean,
                ), _ = jax.lax.scan(accum_body, init_carry, (batch_xs, batch_ys, mb_indices))

        with jax.named_scope("optimizer_muon_adamw"):
            updates, new_opt_state = optimizer.update(grads_mean, opt_state, params)
            new_params = optax.apply_updates(params, updates)

        with jax.named_scope("router_bias_update"):
            moe_dict = dict(new_params["moe"])
            moe_dict["router_bias"] = update_router_biases(
                moe_dict["router_bias"],
                counts_mean,
                num_loops=model_cfg.num_loops,
                update_rate=model_cfg.router_bias_update_rate,
            )
            new_params = {**new_params, "moe": moe_dict}

        expert_mean = jnp.mean(counts_mean)
        expert_cv = jnp.std(counts_mean) / jnp.maximum(expert_mean, 1e-6)
        metrics = {
            "loss": loss_mean,
            "ce_loss": ce_mean,
            "engram_gate": gate_mean,
            "router_mmr": mmr_mean,
            "expected_loops": exp_r_mean,
            "halt_entropy": h_ent_mean,
            "expert_cv": expert_cv,
            "loop_alpha_1": jnp.mean(jnp.abs(new_params["iter_adaln"]["loop_alpha"][1:]))
            if model_cfg.num_loops > 1
            else jnp.asarray(0.0, dtype=jnp.float32),
        }
        return new_params, new_opt_state, metrics

    return train_step


def _configure_compilation_cache(cache_dir: str | None) -> None:
    """Enable JAX persistent compilation cache prior to JIT."""
    if not cache_dir:
        return
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    print(f"[JAX Cache] Persistent compilation cache enabled: {cache_path}")


def _load_tokenizer(
    data_mode: str,
    tokenizer_path: str,
    hf_model_repo: str | None,
    hf_token: str | None,
) -> Tokenizer | None:
    if data_mode == "synthetic":
        return None
    tok_path = Path(tokenizer_path)
    if tok_path.exists():
        return Tokenizer.from_file(str(tok_path))
    if hf_model_repo:
        downloaded = hf_hub_download(
            repo_id=hf_model_repo,
            filename="tokenizer/tokenizer.json",
            repo_type="model",
            token=hf_token,
        )
        return Tokenizer.from_file(downloaded)
    if data_mode == "stream":
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path} and could not download from --hf-model-repo."
        )
    return None


def _build_loader(
    stage_cfg: StageConfig,
    data_mode: str,
    tokenizer: Tokenizer | None,
    vocab_size: int,
    microbatch_tokens: int,
    seed: int,
    shard_dir: str,
    hf_dataset_repo: str | None,
    hf_token: str | None,
):
    if data_mode == "synthetic":
        return SyntheticBatchPrefetcher(
            micro_bs=stage_cfg.microbatch,
            seq_len=stage_cfg.seq_len,
            vocab_size=vocab_size,
            seed=seed + stage_cfg.stage_id * 17,
        )
    if data_mode == "stream":
        assert tokenizer is not None
        return StreamingTokenPrefetcher(
            stage=stage_cfg,
            tokenizer=tokenizer,
            hf_token=hf_token,
            seed=seed,
            queue_size=32,
            microbatch_tokens=microbatch_tokens,
        )
    return ShardBatchPrefetcher(
        stage=stage_cfg,
        shard_dir=shard_dir,
        hf_dataset_repo=hf_dataset_repo,
        hf_token=hf_token,
        seed=seed,
        queue_size=32,
        microbatch_tokens=microbatch_tokens,
    )


def _fetch_global_batch(loader: Any, grad_accum: int) -> tuple[jax.Array, jax.Array]:
    xs_list = []
    ys_list = []
    for _ in range(grad_accum):
        x_np, y_np = loader.next_batch()
        xs_list.append(x_np)
        ys_list.append(y_np)
    return jnp.asarray(np.stack(xs_list, axis=0)), jnp.asarray(np.stack(ys_list, axis=0))


def _prepare_stage_runtime(
    stage_idx: int,
    file_cfg: TrainFileConfig,
    optimizer: optax.GradientTransformation,
    params: dict[str, Any],
    opt_state: Any,
    data_mode: str,
    tokenizer: Tokenizer | None,
    args: argparse.Namespace,
) -> _CompiledStageRuntime:
    st = file_cfg.stages[stage_idx]
    model_cfg = st.build_model_config(file_cfg.model)
    stage_cfg = st.to_stage_config(model_cfg.use_csa2_indexer)
    train_step = make_train_step(
        model_cfg, optimizer, st.grad_accum, seed=file_cfg.training.seed
    )
    dummy_xs = jnp.zeros((st.grad_accum, st.microbatch, st.seq_len), dtype=jnp.uint16)
    dummy_ys = jnp.zeros((st.grad_accum, st.microbatch, st.seq_len), dtype=jnp.uint16)
    t0 = time.perf_counter()
    compiled = train_step.lower(params, opt_state, dummy_xs, dummy_ys).compile()
    compile_s = time.perf_counter() - t0
    del dummy_xs, dummy_ys

    loader = _build_loader(
        stage_cfg=stage_cfg,
        data_mode=data_mode,
        tokenizer=tokenizer,
        vocab_size=model_cfg.vocab_size,
        microbatch_tokens=st.microbatch_tokens,
        seed=file_cfg.training.seed,
        shard_dir=args.shard_dir,
        hf_dataset_repo=args.hf_dataset_repo,
        hf_token=args.hf_token,
    )
    step_flops = estimate_step_flops(model_cfg, st.microbatch, st.seq_len) * st.grad_accum
    print(
        f"[Compiled S{st.stage_id}: {st.name}] seq={st.seq_len} loops={st.num_loops} "
        f"micro={st.microbatch}x{st.seq_len} accum={st.grad_accum} remat_moe={st.remat_moe} "
        f"chunk_ce={st.chunk_ce_size} ({compile_s:.1f}s)"
    )
    return _CompiledStageRuntime(
        stage_idx=stage_idx,
        explicit_cfg=st,
        stage_cfg=stage_cfg,
        model_cfg=model_cfg,
        compiled=compiled,
        step_flops=step_flops,
        loader=loader,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicit config-driven TPU v6e-1 curriculum trainer with Lookahead switching."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to explicit training config JSON (e.g. configs/v6e_1_train.json).",
    )
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional JAX persistent compilation cache directory.",
    )
    parser.add_argument("--shard-dir", type=str, default="artifacts/shards")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional runtime override for training.tokenizer in --config.",
    )
    parser.add_argument(
        "--data-mode",
        choices=["shards", "stream", "synthetic"],
        default=None,
        help="Optional runtime override for training.data_mode in --config.",
    )
    parser.add_argument("--hf-model-repo", type=str, default=None)
    parser.add_argument("--hf-dataset-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--start-stage", type=int, default=None)
    parser.add_argument("--end-stage", type=int, default=None)
    parser.add_argument("--max-steps-per-stage", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_cfg = load_train_config(args.config)
    _configure_compilation_cache(args.cache_dir)

    data_mode = args.data_mode or file_cfg.training.data_mode
    tokenizer_path = args.tokenizer or file_cfg.training.tokenizer
    log_every = args.log_every or file_cfg.training.log_every
    peak_tflops = file_cfg.training.peak_tflops

    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()} | config: {args.config}")
    print(summarize_model(file_cfg.stages[0].build_model_config(file_cfg.model)))
    print(
        f"Lookahead: enabled={file_cfg.lookahead.enabled}, main_steps={file_cfg.lookahead.main_steps}, "
        f"probe_steps={file_cfg.lookahead.probe_steps}, edge_window={file_cfg.lookahead.edge_window}, "
        f"metric={file_cfg.lookahead.metric}, switch_margin={file_cfg.lookahead.switch_margin}"
    )

    tokenizer = _load_tokenizer(data_mode, tokenizer_path, args.hf_model_repo, args.hf_token)
    rng_key = jax.random.PRNGKey(file_cfg.training.seed)

    init_model_cfg = file_cfg.stages[0].build_model_config(file_cfg.model)
    template_params = init_model_params(rng_key, init_model_cfg)
    print(f"Initialized parameter tree leaves: {count_tree_params(template_params):,}")

    optimizer = create_muon_adamw_optimizer(file_cfg.optimizer)
    template_opt_state = optimizer.init(template_params)

    resumed_stage_id = file_cfg.stages[0].stage_id
    resumed_step = 0
    tokens_seen = 0
    expanded_to_multi_loop = False
    resumed_prev_tail: float | None = None

    loaded = load_latest_checkpoint(
        ckpt_root=args.ckpt_dir,
        params_template=template_params,
        opt_state_template=template_opt_state,
        hf_repo_id=args.hf_model_repo,
        hf_token=args.hf_token,
    )
    if loaded is not None:
        params, loaded_opt_state, meta = loaded
        opt_state = loaded_opt_state if loaded_opt_state is not None else template_opt_state
        resumed_stage_id = int(meta["stage_id"])
        resumed_step = int(meta["step"])
        tokens_seen = int(meta["tokens_seen"])
        expanded_to_multi_loop = bool(
            meta.get("expanded_to_multi_loop", int(meta["model_config"]["num_loops"]) > 1)
        )
        resumed_prev_tail = meta.get("lookahead_prev_tail")
        print(
            f"Resumed checkpoint: stage={resumed_stage_id}, step={resumed_step}, "
            f"tokens_seen={tokens_seen:,}, expanded={expanded_to_multi_loop}"
        )
    else:
        params = template_params
        opt_state = template_opt_state
        print("Starting fresh from random initialization.")

    start_stage_id = args.start_stage if args.start_stage is not None else file_cfg.stages[0].stage_id
    end_stage_id = args.end_stage if args.end_stage is not None else file_cfg.stages[-1].stage_id
    target_start_id = max(start_stage_id, resumed_stage_id)

    active_indices = [
        i
        for i, st in enumerate(file_cfg.stages)
        if target_start_id <= st.stage_id <= end_stage_id
    ]
    if not active_indices:
        print("No curriculum stages selected to run.")
        return

    pos = 0
    pending_upload: Future | None = None
    cached_rt: _CompiledStageRuntime | None = None
    tracker = LookaheadTracker(file_cfg.lookahead, prev_probe_tail_loss=resumed_prev_tail)
    stage_step = resumed_step if file_cfg.stages[active_indices[0]].stage_id == resumed_stage_id else 0

    def _maybe_expand_for_stage(st: ExplicitStageConfig) -> None:
        nonlocal params, rng_key, expanded_to_multi_loop
        if st.num_loops > 1 and not expanded_to_multi_loop:
            print(f"\n[Transition] Expanding R=1 -> R={st.num_loops} (bptt_window={file_cfg.model.bptt_window})!")
            rng_key, exp_key = jax.random.split(rng_key)
            params = expand_checkpoint_1_to_n_loops(params, exp_key)
            expanded_to_multi_loop = True

    def _save_stage_ckpt(rt: _CompiledStageRuntime, cur_step: int) -> None:
        nonlocal pending_upload
        if pending_upload is not None:
            pending_upload.result()
        ckpt_dir, pending_upload = save_checkpoint(
            ckpt_root=args.ckpt_dir,
            stage_id=rt.explicit_cfg.stage_id,
            step=cur_step,
            tokens_seen=tokens_seen,
            params=params,
            opt_state=opt_state,
            cfg=rt.model_cfg,
            optim_cfg=file_cfg.optimizer,
            hf_repo_id=args.hf_model_repo,
            hf_token=args.hf_token,
            keep_last_k=file_cfg.training.keep_last_k,
            async_upload=True,
            extra_meta={
                "expanded_to_multi_loop": expanded_to_multi_loop,
                "lookahead_prev_tail": tracker.state.prev_probe_tail_loss,
            },
        )
        print(f"[Checkpoint] Saved {ckpt_dir}")

    try:
        while pos < len(active_indices):
            idx_a = active_indices[pos]
            st_a = file_cfg.stages[idx_a]
            has_next = (pos + 1) < len(active_indices)
            idx_b = active_indices[pos + 1] if has_next else None

            _maybe_expand_for_stage(st_a)
            if cached_rt is not None and cached_rt.stage_idx == idx_a:
                rt_a = cached_rt
                cached_rt = None
            else:
                if cached_rt is not None:
                    cached_rt.loader.close()
                    cached_rt = None
                rt_a = _prepare_stage_runtime(
                    idx_a, file_cfg, optimizer, params, opt_state, data_mode, tokenizer, args
                )

            max_steps_a = args.max_steps_per_stage or (
                (st_a.target_tokens + st_a.global_batch_tokens - 1) // st_a.global_batch_tokens
            )
            rt_b: _CompiledStageRuntime | None = None
            switched = False

            try:
                while stage_step < max_steps_a:
                    in_warmup = stage_step < st_a.warmup_steps
                    if in_warmup or not tracker.should_probe(stage_step, st_a.warmup_steps, has_next):
                        n_steps = (
                            min(st_a.warmup_steps - stage_step, max_steps_a - stage_step)
                            if in_warmup
                            else (max_steps_a - stage_step)
                        )
                        t_log = time.perf_counter()
                        logged_at = stage_step
                        for _ in range(n_steps):
                            bx, by = _fetch_global_batch(rt_a.loader, st_a.grad_accum)
                            params, opt_state, metrics = rt_a.compiled(params, opt_state, bx, by)
                            stage_step += 1
                            tokens_seen += st_a.global_batch_tokens

                            if stage_step % log_every == 0 or stage_step == max_steps_a:
                                jax.block_until_ready(metrics["loss"])
                                dt = max(time.perf_counter() - t_log, 1e-6)
                                elapsed_n = max(1, stage_step - logged_at)
                                sec_per_step = dt / elapsed_n
                                tok_s = st_a.global_batch_tokens / sec_per_step
                                mfu = 100.0 * (rt_a.step_flops / sec_per_step) / (peak_tflops * 1e12)
                                tag = " [warmup]" if in_warmup else ""
                                print(
                                    f"[S{st_a.stage_id}:{st_a.name} step {stage_step}/{max_steps_a}]{tag} "
                                    f"loss={float(metrics['loss']):.4f} ce={float(metrics['ce_loss']):.4f} "
                                    f"eng_g={float(metrics['engram_gate']):.3f} exp_cv={float(metrics['expert_cv']):.3f} "
                                    f"| {tok_s:,.0f} tok/s | MFU={mfu:.1f}% | total_tok={tokens_seen/1e9:.4f}B"
                                )
                                t_log = time.perf_counter()
                                logged_at = stage_step

                            if stage_step % st_a.checkpoint_every_steps == 0 or stage_step == max_steps_a:
                                _save_stage_ckpt(rt_a, stage_step)
                        continue

                    # Lookahead Block A -> Probe B
                    assert idx_b is not None
                    st_b = file_cfg.stages[idx_b]
                    n_a = min(file_cfg.lookahead.main_steps, max_steps_a - stage_step)
                    jax.block_until_ready(params["embed_tokens"])
                    t_a0 = time.perf_counter()
                    for _ in range(n_a):
                        bx, by = _fetch_global_batch(rt_a.loader, st_a.grad_accum)
                        params, opt_state, metrics_a = rt_a.compiled(params, opt_state, bx, by)
                        stage_step += 1
                        tokens_seen += st_a.global_batch_tokens
                        if stage_step % log_every == 0:
                            print(
                                f"[S{st_a.stage_id}:{st_a.name} step {stage_step}/{max_steps_a}] "
                                f"loss={float(metrics_a['loss']):.4f} ce={float(metrics_a['ce_loss']):.4f} "
                                f"| total_tok={tokens_seen/1e9:.4f}B"
                            )
                        if stage_step % st_a.checkpoint_every_steps == 0:
                            _save_stage_ckpt(rt_a, stage_step)
                    jax.block_until_ready(metrics_a["loss"])
                    t_a_sec = max(time.perf_counter() - t_a0, 1e-6)
                    sec_a = t_a_sec / max(1, n_a)
                    mfu_a = 100.0 * (rt_a.step_flops / sec_a) / (peak_tflops * 1e12)
                    print(
                        f"[Block S{st_a.stage_id}:{st_a.name}] {n_a} steps in {t_a_sec:.2f}s "
                        f"({st_a.global_batch_tokens/sec_a:,.0f} tok/s, MFU={mfu_a:.1f}%)"
                    )

                    if rt_b is None:
                        _maybe_expand_for_stage(st_b)
                        rt_b = _prepare_stage_runtime(
                            idx_b, file_cfg, optimizer, params, opt_state, data_mode, tokenizer, args
                        )

                    n_b = file_cfg.lookahead.probe_steps
                    b_losses: list[float] = []
                    metric_key = file_cfg.lookahead.metric
                    jax.block_until_ready(params["embed_tokens"])
                    t_b0 = time.perf_counter()
                    for _ in range(n_b):
                        bx_b, by_b = _fetch_global_batch(rt_b.loader, st_b.grad_accum)
                        params, opt_state, metrics_b = rt_b.compiled(params, opt_state, bx_b, by_b)
                        b_losses.append(float(metrics_b[metric_key]))
                        tokens_seen += st_b.global_batch_tokens
                    t_b_sec = max(time.perf_counter() - t_b0, 1e-6)

                    decision = tracker.record_probe(t_a_sec=t_a_sec, t_b_sec=t_b_sec, b_losses=b_losses)
                    eff_a_str = f"{decision.eff_a:.6f}/s" if decision.eff_a is not None else "N/A"
                    eff_b_str = f"{decision.eff_b:.6f}/s" if decision.eff_b is not None else "N/A"
                    prev_str = (
                        f"{decision.prev_tail_loss:.4f}" if decision.prev_tail_loss is not None else "N/A"
                    )
                    print(
                        f"[Lookahead S{st_a.stage_id}->S{st_b.stage_id} probe #{decision.probe_idx}] "
                        f"prev_tail={prev_str} head={decision.head_loss:.4f} tail={decision.tail_loss:.4f} | "
                        f"eff_A={eff_a_str} eff_B={eff_b_str} -> {decision.reason}"
                    )

                    if decision.should_switch:
                        _save_stage_ckpt(rt_b, n_b)
                        cached_rt = rt_b
                        rt_b = None
                        stage_step = n_b
                        tracker.reset()
                        switched = True
                        break

                if not switched and stage_step >= max_steps_a:
                    _save_stage_ckpt(rt_a, stage_step)
            finally:
                rt_a.loader.close()
                if rt_b is not None and not switched:
                    rt_b.loader.close()

            if not switched:
                tracker.reset()
                stage_step = 0
            pos += 1
    finally:
        if cached_rt is not None:
            cached_rt.loader.close()

    if pending_upload is not None:
        pending_upload.result()
    print("\nTraining curriculum complete!")


if __name__ == "__main__":
    main()
