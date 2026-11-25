"""High-MFU single-chip TPU v6e-1 curriculum trainer for 1B 1-Layer DeepSeek-V4.1-Flash LoopMoE."""

from __future__ import annotations

import argparse
from concurrent.futures import Future
import contextlib
import gc
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
from loopmoe.config import (
    DEFAULT_CURRICULUM,
    TPU_SPECS,
    LoopMoEConfig,
    StageConfig,
    estimate_step_flops,
)
from loopmoe.data import ShardBatchPrefetcher, StreamingTokenPrefetcher
from loopmoe.model import (
    count_tree_params,
    expand_checkpoint_1_to_8_loops,
    forward_model,
    init_model_params,
    summarize_model,
)
from loopmoe.moe import update_router_biases
from loopmoe.optim import OptimConfig, create_muon_adamw_optimizer


class SyntheticBatchPrefetcher:
    """Zero-I/O synthetic token generator for instant TPU v6e MFU benchmarking."""

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


def maybe_tokamax_autotune(
    cfg: LoopMoEConfig,
    params: dict[str, Any],
    x_mb: jax.Array,
    y_mb: jax.Array,
    enabled: bool = True,
) -> Any:
    """Run Tokamax kernel autotuning on TPU to resolve cache misses and select optimal Pallas tiles."""
    if not enabled or jax.default_backend() != "tpu":
        return contextlib.nullcontext()
    try:
        import tokamax  # type: ignore

        autotune_fn = getattr(tokamax, "autotune", None)
        if autotune_fn is None:
            return contextlib.nullcontext()

        fwd_bwd = jax.jit(
            jax.value_and_grad(lambda p, x, y: forward_model(p, x, y, cfg), has_aux=True)
        )
        t0 = time.perf_counter()
        res = autotune_fn(fwd_bwd, params, x_mb.astype(jnp.int32), y_mb.astype(jnp.int32))
        dt = time.perf_counter() - t0
        del fwd_bwd
        gc.collect()
        n_tuned = (
            len(res)
            if hasattr(res, "__len__")
            else len(getattr(res, "data", {}))
            if hasattr(res, "data")
            else "?"
        )
        print(f"[Tokamax Autotune] Tuned {n_tuned} kernel configs in {dt:.1f}s")
        return res
    except Exception as exc:
        print(f"[Tokamax Autotune] Skipped ({type(exc).__name__}: {exc}); using default heuristics.")
        return contextlib.nullcontext()


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
                # Fuse inv_g scaling into accumulation to avoid a duplicate full-model grads_mean pytree
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

        # Auxiliary-Loss-Free dynamic bias adjustment outside gradient flow
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


def _load_tokenizer(args: argparse.Namespace) -> Tokenizer | None:
    if args.data_mode == "synthetic":
        return None
    tok_path = Path(args.tokenizer)
    if tok_path.exists():
        return Tokenizer.from_file(str(tok_path))
    if args.hf_model_repo:
        downloaded = hf_hub_download(
            repo_id=args.hf_model_repo,
            filename="tokenizer/tokenizer.json",
            repo_type="model",
            token=args.hf_token,
        )
        return Tokenizer.from_file(downloaded)
    if args.data_mode == "stream":
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path} and could not download from --hf-model-repo."
        )
    return None


def _build_loader(
    stage: StageConfig,
    args: argparse.Namespace,
    model_cfg: LoopMoEConfig,
    tokenizer: Tokenizer | None,
    microbatch_tokens: int,
):
    micro_bs = microbatch_tokens // stage.seq_len
    if args.data_mode == "synthetic":
        return SyntheticBatchPrefetcher(micro_bs, stage.seq_len, model_cfg.vocab_size, seed=args.seed)
    if args.data_mode == "stream":
        assert tokenizer is not None
        return StreamingTokenPrefetcher(
            stage=stage,
            tokenizer=tokenizer,
            hf_token=args.hf_token,
            seed=args.seed,
            queue_size=32,
            microbatch_tokens=microbatch_tokens,
        )
    return ShardBatchPrefetcher(
        stage=stage,
        shard_dir=args.shard_dir,
        hf_dataset_repo=args.hf_dataset_repo,
        hf_token=args.hf_token,
        seed=args.seed,
        queue_size=32,
        microbatch_tokens=microbatch_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1B DeepSeek-V4.1-Flash LoopMoE on TPU v6e-1.")
    parser.add_argument("--hf-model-repo", type=str, default=None, help="HF model repo for auto-resume & ckpts.")
    parser.add_argument("--hf-dataset-repo", type=str, default=None, help="HF dataset repo containing uint16 shards.")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=3,
        help="Number of recent local checkpoints to retain (<=0 keeps all).",
    )
    parser.add_argument("--shard-dir", type=str, default="artifacts/shards")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Use Ultra-FineWeb EN only (0%% ZH) and default to artifacts/tokenizer_64k_en/tokenizer.json.",
    )
    parser.add_argument("--data-mode", choices=["shards", "stream", "synthetic"], default="stream")

    # Stage & optimization overrides
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=7)
    parser.add_argument("--max-steps-per-stage", type=int, default=None, help="Optional cap for smoke testing.")
    parser.add_argument("--microbatch", type=int, default=None, help="Microbatch size in sequences (default: 4 on v5e-1, 8 on v6e-1).")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps (default: 16 on v5e-1, 8 on v6e-1).")
    parser.add_argument("--global-batch-tokens", type=int, default=None, help="Override global batch tokens.")
    parser.add_argument("--seq-len-override", type=int, default=None, help="Optional seq_len override for debugging.")
    parser.add_argument("--muon-lr", type=float, default=0.015)
    parser.add_argument("--adam-lr", type=float, default=1.5e-3)
    parser.add_argument(
        "--tpu-preset",
        choices=["v5e-1", "v6e-1"],
        default="v6e-1",
        help="Hardware preset adjusting peak TFLOPS, chunk CE size, sublayer remat, and default microbatch.",
    )
    parser.add_argument(
        "--remat-sublayers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable sublayer remat inside loop (auto-enabled on v5e-1 at >=16K).",
    )
    parser.add_argument(
        "--remat-moe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint only the MoE sublayer (saves ~3 GB HBM at microbatch>=8 without recomputing MLA/mHC).",
    )
    parser.add_argument("--chunk-ce-size", type=int, default=None, help="Chunk size for chunked cross-entropy.")
    parser.add_argument(
        "--moe-chunk-size",
        type=int,
        default=None,
        help="Token chunk size inside MoE to reduce peak HBM at >=16K (0=unchunked; auto 4096 on v5e-1 at seq>=16384).",
    )
    parser.add_argument(
        "--autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in slow Tokamax autotuning sweep on TPU (explicit tile configs are already set in moe.py/mla.py).",
    )
    parser.add_argument("--profile-dir", type=str, default=None, help="Optional JAX profiler trace output directory.")
    parser.add_argument("--profile-start-step", type=int, default=2, help="Step index to start profiling trace.")
    parser.add_argument("--profile-steps", type=int, default=3, help="Number of steps to trace in profiler.")
    parser.add_argument("--peak-tflops", type=float, default=None, help="Override TPU BF16 peak TFLOPS.")
    parser.add_argument("--use-csa2-long", action="store_true", help="Enable CSA2 indexer on >=16K stages.")
    parser.add_argument(
        "--dynamic-loop-ce-subsample",
        type=int,
        default=None,
        help="Stride for deeper-loop CE in arXiv:2611.05724 dynamic loops (1=exact, 2/4=faster R>1 training).",
    )
    parser.add_argument(
        "--dynamic-loop-ce-depths",
        type=int,
        default=None,
        help="Number of deep depths (r >= 1) sampled per microbatch for CE (default 2; <=0 evaluates all).",
    )
    parser.add_argument(
        "--train-loop-capacity",
        type=float,
        default=None,
        help="Fraction of hardest/continuing tokens routed through MoE on r >= 1 (default 0.5).",
    )
    parser.add_argument(
        "--bptt-window",
        type=int,
        default=None,
        help="Truncated BPTT suffix window for high R (e.g. R=32 uses Loop 0 + last K loops; default 8).",
    )
    parser.add_argument(
        "--halt-balance-weight",
        type=float,
        default=None,
        help="Marginal halting entropy weight beta in arXiv:2611.05724 Eq. 6 (default 0.01).",
    )
    parser.add_argument("--tiny", action="store_true", help="Use tiny model config for local debugging.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = TPU_SPECS[args.tpu_preset]
    peak_tflops = args.peak_tflops or spec["peak_tflops_bf16"]
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()} | preset: {args.tpu_preset}")

    base_cfg = LoopMoEConfig.tiny() if args.tiny else LoopMoEConfig()
    cfg_overrides: dict[str, Any] = {}
    if args.use_csa2_long:
        cfg_overrides["use_csa2_indexer"] = True
    if args.dynamic_loop_ce_subsample is not None:
        cfg_overrides["dynamic_loop_ce_subsample"] = args.dynamic_loop_ce_subsample
    if args.dynamic_loop_ce_depths is not None:
        cfg_overrides["dynamic_loop_ce_sample_depths"] = args.dynamic_loop_ce_depths
    if args.train_loop_capacity is not None:
        cfg_overrides["train_loop_capacity_factor"] = args.train_loop_capacity
    if args.bptt_window is not None:
        cfg_overrides["bptt_window"] = args.bptt_window
    if args.halt_balance_weight is not None:
        cfg_overrides["halt_balance_weight"] = args.halt_balance_weight
    if args.tpu_preset == "v5e-1" and not args.tiny:
        cfg_overrides["chunk_ce_size"] = int(spec["default_chunk_ce_size"])
    if args.remat_sublayers:
        cfg_overrides["remat_sublayers"] = True
    remat_moe = False if args.remat_moe is None else args.remat_moe
    if remat_moe:
        cfg_overrides["remat_moe"] = True
    if args.chunk_ce_size is not None:
        cfg_overrides["chunk_ce_size"] = args.chunk_ce_size
    if args.moe_chunk_size is not None:
        cfg_overrides["moe_chunk_size"] = args.moe_chunk_size
    if cfg_overrides:
        base_cfg = LoopMoEConfig(**{**base_cfg.to_dict(), **cfg_overrides})
    print(summarize_model(base_cfg))

    if args.tokenizer is None:
        args.tokenizer = (
            "artifacts/tokenizer_64k_en/tokenizer.json"
            if args.english_only
            else "artifacts/tokenizer_64k/tokenizer.json"
        )
    tokenizer = _load_tokenizer(args)
    rng_key = jax.random.PRNGKey(args.seed)
    params: dict[str, Any] | None = None
    opt_state: Any | None = None
    resumed_stage = 1
    resumed_step = 0
    tokens_seen = 0
    prev_num_loops = 1

    # Attempt checkpoint resume
    stage1_cfg = base_cfg.with_stage(DEFAULT_CURRICULUM[0].seq_len, DEFAULT_CURRICULUM[0].num_loops)
    template_params = init_model_params(rng_key, stage1_cfg)
    print(f"Initialized parameter tree leaves: {count_tree_params(template_params):,}")

    init_optim_cfg = OptimConfig(
        muon_lr=args.muon_lr,
        adam_lr=args.adam_lr,
        warmup_steps=DEFAULT_CURRICULUM[0].warmup_steps,
    )
    init_optimizer = create_muon_adamw_optimizer(init_optim_cfg)
    template_opt_state = init_optimizer.init(template_params)

    loaded = load_latest_checkpoint(
        ckpt_root=args.ckpt_dir,
        params_template=template_params,
        opt_state_template=template_opt_state,
        hf_repo_id=args.hf_model_repo,
        hf_token=args.hf_token,
    )
    if loaded is not None:
        params, opt_state, meta = loaded
        resumed_stage = int(meta["stage_id"])
        resumed_step = int(meta["step"])
        tokens_seen = int(meta["tokens_seen"])
        prev_num_loops = int(meta["model_config"]["num_loops"])
        print(
            f"Resumed checkpoint: stage={resumed_stage}, step={resumed_step}, "
            f"tokens_seen={tokens_seen:,}, loops={prev_num_loops}"
        )
    else:
        params = template_params
        opt_state = template_opt_state
        print("Starting fresh from random initialization.")

    pending_upload: Future | None = None

    for stage in DEFAULT_CURRICULUM:
        if stage.stage_id < max(args.start_stage, resumed_stage) or stage.stage_id > args.end_stage:
            continue

        if args.seq_len_override is not None:
            stage = StageConfig(**{**stage.__dict__, "seq_len": args.seq_len_override})
        if args.english_only:
            stage = stage.with_english_only()

        start_step = resumed_step if stage.stage_id == resumed_stage else 0
        total_steps = args.max_steps_per_stage or stage.total_steps
        if start_step >= total_steps:
            continue

        global_tok = args.global_batch_tokens or stage.global_batch_tokens
        default_micro = 2 if args.tiny else int(spec["default_microbatch"])
        micro_bs = args.microbatch or default_micro
        micro_tok = micro_bs * stage.seq_len
        if args.grad_accum is not None:
            grad_accum = args.grad_accum
        elif args.global_batch_tokens is not None or args.microbatch is not None:
            grad_accum = max(1, global_tok // micro_tok)
        else:
            grad_accum = 1 if args.tiny else int(spec["default_grad_accum"])

        stage_model_cfg = base_cfg.with_stage(
            seq_len=stage.seq_len,
            num_loops=stage.num_loops,
            use_csa2_indexer=bool(args.use_csa2_long),
        )
        stage_overrides: dict[str, Any] = {}
        if args.use_csa2_long and stage.seq_len < 16384:
            # Keep idx_* parameter shapes consistent across stages while staying dense below 16K
            stage_overrides["csa2_topk"] = stage.seq_len
        if args.tpu_preset == "v5e-1" and not args.tiny and (stage.num_loops > 1 or stage.seq_len >= 16384):
            if args.remat_moe is None and not stage_model_cfg.remat_moe:
                stage_overrides["remat_moe"] = True
            if stage.seq_len >= 16384:
                if stage.num_loops > 1 and args.remat_sublayers is None and not stage_model_cfg.remat_sublayers:
                    stage_overrides["remat_sublayers"] = True
                if args.moe_chunk_size is None and stage_model_cfg.moe_chunk_size == 0:
                    stage_overrides["moe_chunk_size"] = 4096
                if args.chunk_ce_size is None and stage_model_cfg.chunk_ce_size > 4096:
                    stage_overrides["chunk_ce_size"] = 4096
                explicit_risks = []
                if stage.num_loops > 1 and args.remat_sublayers is False:
                    explicit_risks.append("--no-remat-sublayers")
                if args.remat_moe is False:
                    explicit_risks.append("--no-remat-moe")
                if args.chunk_ce_size is not None and args.chunk_ce_size > 4096:
                    explicit_risks.append(f"--chunk-ce-size {args.chunk_ce_size}")
                if explicit_risks:
                    print(
                        f"[WARN v5e-1 @ seq={stage.seq_len}] Explicit flags disable 16 GB HBM savings: "
                        + ", ".join(explicit_risks)
                    )
        if stage_overrides:
            stage_model_cfg = LoopMoEConfig(**{**stage_model_cfg.to_dict(), **stage_overrides})

        # Smooth R=1 -> R=8 expansion entering Stage 7
        if stage.num_loops > prev_num_loops and prev_num_loops == 1:
            print(f"\n[Transition] Expanding Phase-1 (R=1) -> Phase-2 (R={stage.num_loops}) with step-0 identity!")
            rng_key, exp_key = jax.random.split(rng_key)
            params = expand_checkpoint_1_to_8_loops(params, exp_key)
            opt_state = None  # Fresh warmup for newly opened loop parameters

        optim_cfg = OptimConfig(
            muon_lr=args.muon_lr,
            adam_lr=args.adam_lr,
            warmup_steps=stage.warmup_steps,
        )
        optimizer = create_muon_adamw_optimizer(optim_cfg)
        if opt_state is None:
            opt_state = optimizer.init(params)

        train_step = make_train_step(stage_model_cfg, optimizer, grad_accum, seed=args.seed)
        step_flops = estimate_step_flops(stage_model_cfg, micro_bs, stage.seq_len) * grad_accum
        loader = _build_loader(stage, args, stage_model_cfg, tokenizer, micro_tok)

        dummy_x = jnp.zeros((micro_bs, stage.seq_len), dtype=jnp.int32)
        dummy_y = jnp.zeros((micro_bs, stage.seq_len), dtype=jnp.int32)
        autotune_ctx = maybe_tokamax_autotune(
            stage_model_cfg, params, dummy_x, dummy_y, enabled=args.autotune
        )
        del dummy_x, dummy_y

        print(
            f"\n=== Stage {stage.stage_id}: {stage.name} | seq={stage.seq_len} loops={stage.num_loops} "
            f"micro={micro_bs}x{stage.seq_len} accum={grad_accum} steps={start_step}..{total_steps} "
            f"mix=EN:{stage.web_en_ratio:.2f}/ZH:{stage.web_zh_ratio:.2f}/Code:{stage.code_ratio:.2f}/Math:{stage.math_ratio:.2f} ==="
        )

        try:
            with autotune_ctx:
                t_log = time.perf_counter()
                profiling_active = False
                prof_end_step = args.profile_start_step + args.profile_steps
                for step in range(start_step, total_steps):
                    if args.profile_dir and not profiling_active and step == args.profile_start_step:
                        Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
                        jax.profiler.start_trace(args.profile_dir)
                        profiling_active = True

                    xs_list = []
                    ys_list = []
                    for _ in range(grad_accum):
                        x_np, y_np = loader.next_batch()
                        xs_list.append(x_np)
                        ys_list.append(y_np)

                    batch_xs = jnp.asarray(np.stack(xs_list, axis=0))
                    batch_ys = jnp.asarray(np.stack(ys_list, axis=0))

                    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
                        params, opt_state, metrics = train_step(params, opt_state, batch_xs, batch_ys)
                        if profiling_active:
                            jax.block_until_ready(metrics["loss"])

                    if profiling_active and (step + 1) >= prof_end_step:
                        jax.profiler.stop_trace()
                        profiling_active = False
                        print(f"[Profiler] Saved trace to {args.profile_dir}")

                    # Synchronize on log/ckpt boundaries for accurate MFU timing
                    tokens_seen += grad_accum * micro_tok

                    if (step + 1) % args.log_every == 0 or step == start_step:
                        jax.block_until_ready(metrics["loss"])
                        dt = max(time.perf_counter() - t_log, 1e-6)
                        steps_elapsed = 1 if step == start_step else args.log_every
                        sec_per_step = dt / steps_elapsed
                        tok_per_sec = (grad_accum * micro_tok) / sec_per_step
                        mfu = 100.0 * (step_flops / sec_per_step) / (peak_tflops * 1e12)
                        tag = " (compile)" if step == start_step else ""
                        dyn_str = (
                            f" exp_r={float(metrics['expected_loops']):.2f}"
                            f" h_ent={float(metrics['halt_entropy']):.2f}"
                            if stage.num_loops > 1
                            else ""
                        )
                        print(
                            f"[S{stage.stage_id} step {step + 1}/{total_steps}]{tag} "
                            f"loss={float(metrics['loss']):.4f} ce={float(metrics['ce_loss']):.4f} "
                            f"eng_g={float(metrics['engram_gate']):.3f} exp_cv={float(metrics['expert_cv']):.3f} "
                            f"mmr={float(metrics['router_mmr']):.2f}{dyn_str} "
                            f"| {tok_per_sec:,.0f} tok/s | MFU={mfu:.1f}% | total_tok={tokens_seen/1e9:.3f}B"
                        )
                        t_log = time.perf_counter()

                    if (step + 1) % stage.checkpoint_every_steps == 0 or (step + 1) == total_steps:
                        if pending_upload is not None:
                            pending_upload.result()
                        ckpt_dir, pending_upload = save_checkpoint(
                            ckpt_root=args.ckpt_dir,
                            stage_id=stage.stage_id,
                            step=step + 1,
                            tokens_seen=tokens_seen,
                            params=params,
                            opt_state=opt_state,
                            cfg=stage_model_cfg,
                            optim_cfg=optim_cfg,
                            hf_repo_id=args.hf_model_repo,
                            hf_token=args.hf_token,
                            keep_last_k=args.keep_last_k,
                            async_upload=True,
                            async_write=True,
                        )
                        print(f"[Checkpoint] Queued async save {ckpt_dir}")
        finally:
            if profiling_active:
                jax.profiler.stop_trace()
                print(f"[Profiler] Saved trace to {args.profile_dir}")
            loader.close()

        prev_num_loops = stage.num_loops
        resumed_step = 0

    if pending_upload is not None:
        pending_upload.result()
    print("\nTraining curriculum complete!")


if __name__ == "__main__":
    main()
