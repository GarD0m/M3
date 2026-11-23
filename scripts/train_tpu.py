"""High-MFU single-chip TPU v6e-1 curriculum trainer for 1B 1-Layer DeepSeek-V4.1-Flash LoopMoE."""

from __future__ import annotations

import argparse
from concurrent.futures import Future
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
        x = self.rng.integers(64, self.vocab_size, size=(self.micro_bs, self.seq_len), dtype=np.int32)
        y = np.roll(x, -1, axis=1)
        return x, y

    def close(self) -> None:
        return None


def make_train_step(
    model_cfg: LoopMoEConfig,
    optimizer: optax.GradientTransformation,
    grad_accum_steps: int,
):
    """Build a donated JIT-compiled global optimizer step over `grad_accum_steps` microbatches."""

    def _loss_fn(params: dict[str, Any], x: jax.Array, y: jax.Array):
        return forward_model(params, x, y, model_cfg)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

    @jax.jit(donate_argnums=(0, 1))
    def train_step(
        params: dict[str, Any],
        opt_state: Any,
        batch_xs: jax.Array,
        batch_ys: jax.Array,
    ) -> tuple[dict[str, Any], Any, dict[str, jax.Array]]:
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        init_counts = jnp.zeros((model_cfg.num_loops, model_cfg.n_routed_experts), dtype=jnp.float32)
        init_carry = (
            zero_grads,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            init_counts,
        )

        inv_g = 1.0 / float(grad_accum_steps)

        def accum_body(carry, microbatch):
            g_acc, loss_acc, ce_acc, gate_acc, cnt_acc = carry
            x_mb, y_mb = microbatch
            with jax.named_scope("fwd_bwd_microbatch"):
                (loss, aux), grads = grad_fn(params, x_mb, y_mb)
            g_acc = jax.tree_util.tree_map(lambda a, g: a + g, g_acc, grads)
            return (
                g_acc,
                loss_acc + loss * inv_g,
                ce_acc + aux["ce_loss"] * inv_g,
                gate_acc + aux["engram_gate_mean"] * inv_g,
                cnt_acc + aux["expert_counts"] * inv_g,
            ), None

        with jax.named_scope("grad_accum_scan"):
            (grads_sum, loss_mean, ce_mean, gate_mean, counts_mean), _ = jax.lax.scan(
                accum_body, init_carry, (batch_xs, batch_ys)
            )
        grads_mean = jax.tree_util.tree_map(
            lambda g: g * jnp.asarray(inv_g, dtype=g.dtype), grads_sum
        )

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
            microbatch_tokens=microbatch_tokens,
        )
    return ShardBatchPrefetcher(
        stage=stage,
        shard_dir=args.shard_dir,
        hf_dataset_repo=args.hf_dataset_repo,
        hf_token=args.hf_token,
        seed=args.seed,
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
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer_64k/tokenizer.json")
    parser.add_argument("--data-mode", choices=["shards", "stream", "synthetic"], default="stream")

    # Stage & optimization overrides
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=7)
    parser.add_argument("--max-steps-per-stage", type=int, default=None, help="Optional cap for smoke testing.")
    parser.add_argument("--microbatch-tokens", type=int, default=None, help="Override microbatch tokens.")
    parser.add_argument("--global-batch-tokens", type=int, default=None, help="Override global batch tokens.")
    parser.add_argument("--seq-len-override", type=int, default=None, help="Optional seq_len override for debugging.")
    parser.add_argument("--muon-lr", type=float, default=0.015)
    parser.add_argument("--adam-lr", type=float, default=1.5e-3)
    parser.add_argument("--peak-tflops", type=float, default=918.0, help="TPU v6e-1 BF16 peak TFLOPS.")
    parser.add_argument("--use-csa2-long", action="store_true", help="Enable CSA2 indexer on >=16K stages.")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model config for local debugging.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")

    base_cfg = LoopMoEConfig.tiny() if args.tiny else LoopMoEConfig()
    if args.use_csa2_long:
        base_cfg = LoopMoEConfig(**{**base_cfg.to_dict(), "use_csa2_indexer": True})
    print(summarize_model(base_cfg))

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

        start_step = resumed_step if stage.stage_id == resumed_stage else 0
        total_steps = args.max_steps_per_stage or stage.total_steps
        if start_step >= total_steps:
            continue

        global_tok = args.global_batch_tokens or stage.global_batch_tokens
        micro_tok = args.microbatch_tokens or stage.microbatch_tokens
        if micro_tok % stage.seq_len != 0:
            micro_tok = stage.seq_len
        grad_accum = max(1, global_tok // micro_tok)
        micro_bs = micro_tok // stage.seq_len

        stage_model_cfg = base_cfg.with_stage(
            seq_len=stage.seq_len,
            num_loops=stage.num_loops,
            use_csa2_indexer=bool(args.use_csa2_long),
        )
        if args.use_csa2_long and stage.seq_len < 16384:
            # Keep idx_* parameter shapes consistent across stages while staying dense below 16K
            stage_model_cfg = LoopMoEConfig(
                **{**stage_model_cfg.to_dict(), "csa2_topk": stage.seq_len}
            )

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

        train_step = make_train_step(stage_model_cfg, optimizer, grad_accum)
        step_flops = estimate_step_flops(stage_model_cfg, micro_bs, stage.seq_len) * grad_accum
        loader = _build_loader(stage, args, stage_model_cfg, tokenizer, micro_tok)

        print(
            f"\n=== Stage {stage.stage_id}: {stage.name} | seq={stage.seq_len} loops={stage.num_loops} "
            f"micro={micro_bs}x{stage.seq_len} accum={grad_accum} steps={start_step}..{total_steps} ==="
        )

        try:
            t_log = time.perf_counter()
            for step in range(start_step, total_steps):
                xs_list = []
                ys_list = []
                for _ in range(grad_accum):
                    x_np, y_np = loader.next_batch()
                    xs_list.append(x_np)
                    ys_list.append(y_np)

                batch_xs = jnp.asarray(np.stack(xs_list, axis=0))
                batch_ys = jnp.asarray(np.stack(ys_list, axis=0))

                params, opt_state, metrics = train_step(params, opt_state, batch_xs, batch_ys)
                # Synchronize on log/ckpt boundaries for accurate MFU timing
                tokens_seen += grad_accum * micro_tok

                if (step + 1) % args.log_every == 0 or step == start_step:
                    jax.block_until_ready(metrics["loss"])
                    dt = max(time.perf_counter() - t_log, 1e-6)
                    steps_elapsed = 1 if step == start_step else args.log_every
                    sec_per_step = dt / steps_elapsed
                    tok_per_sec = (grad_accum * micro_tok) / sec_per_step
                    mfu = 100.0 * (step_flops / sec_per_step) / (args.peak_tflops * 1e12)
                    tag = " (compile)" if step == start_step else ""
                    print(
                        f"[S{stage.stage_id} step {step + 1}/{total_steps}]{tag} "
                        f"loss={float(metrics['loss']):.4f} ce={float(metrics['ce_loss']):.4f} "
                        f"eng_g={float(metrics['engram_gate']):.3f} exp_cv={float(metrics['expert_cv']):.3f} "
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
                    )
                    print(f"[Checkpoint] Saved {ckpt_dir}")
        finally:
            loader.close()

        prev_num_loops = stage.num_loops
        resumed_step = 0

    if pending_upload is not None:
        pending_upload.result()
    print("\nTraining curriculum complete!")


if __name__ == "__main__":
    main()
