"""TPU v6e-1 MFU, throughput, compile-time, and HBM benchmark suite (Test branch)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import jax
import jax.numpy as jnp
import numpy as np

from loopmoe.config import LoopMoEConfig, estimate_step_flops
from loopmoe.model import init_model_params, summarize_model
from loopmoe.optim import OptimConfig, create_muon_adamw_optimizer
from train_tpu import make_train_step


def _probe_tpu_kernels() -> dict[str, bool]:
    info = {
        "on_tpu": jax.default_backend() == "tpu",
        "tokamax_installed": False,
        "tokamax_ragged_dot": False,
        "tokamax_dpa": False,
        "pallas_splash_attention": False,
    }
    try:
        import tokamax  # type: ignore

        info["tokamax_installed"] = True
        info["tokamax_ragged_dot"] = hasattr(tokamax, "ragged_dot")
        info["tokamax_dpa"] = hasattr(tokamax, "dot_product_attention")
    except Exception:
        pass
    try:
        from jax.experimental.pallas.ops.tpu import splash_attention  # noqa: F401

        info["pallas_splash_attention"] = True
    except Exception:
        pass
    return info


def _peak_hbm_gb() -> float | None:
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            return float(stats["peak_bytes_in_use"]) / (1024**3)
    except Exception:
        pass
    return None


def _run_single_point(
    base_cfg: LoopMoEConfig,
    seq_len: int,
    num_loops: int,
    microbatch_tokens: int,
    grad_accum: int,
    warmup_steps: int,
    bench_steps: int,
    peak_tflops: float,
    use_csa2: bool,
    seed: int,
) -> dict[str, Any]:
    micro_tok = max(seq_len, (microbatch_tokens // seq_len) * seq_len)
    micro_bs = micro_tok // seq_len

    cfg = base_cfg.with_stage(
        seq_len=seq_len,
        num_loops=num_loops,
        use_csa2_indexer=use_csa2,
    )
    if use_csa2 and seq_len < 16384:
        cfg = LoopMoEConfig(**{**cfg.to_dict(), "csa2_topk": seq_len})

    params = init_model_params(jax.random.PRNGKey(seed), cfg)
    opt = create_muon_adamw_optimizer(OptimConfig(warmup_steps=100))
    opt_state = opt.init(params)
    train_step = make_train_step(cfg, opt, grad_accum)

    rng = np.random.default_rng(seed)
    x_np = rng.integers(64, cfg.vocab_size, size=(grad_accum, micro_bs, seq_len), dtype=np.int32)
    y_np = np.roll(x_np, -1, axis=-1)
    bx = jnp.asarray(x_np)
    by = jnp.asarray(y_np)

    # Compile + first step
    t0 = time.perf_counter()
    params, opt_state, metrics = train_step(params, opt_state, bx, by)
    jax.block_until_ready(metrics["loss"])
    compile_s = time.perf_counter() - t0

    # Extra warmup
    for _ in range(max(0, warmup_steps - 1)):
        params, opt_state, metrics = train_step(params, opt_state, bx, by)
        jax.block_until_ready(metrics["loss"])

    # Timed steady-state benchmark
    t_start = time.perf_counter()
    for _ in range(bench_steps):
        params, opt_state, metrics = train_step(params, opt_state, bx, by)
    jax.block_until_ready(metrics["loss"])
    elapsed = max(time.perf_counter() - t_start, 1e-6)

    sec_per_step = elapsed / float(bench_steps)
    step_tokens = grad_accum * micro_tok
    tok_per_sec = float(step_tokens) / sec_per_step
    step_flops = estimate_step_flops(cfg, micro_bs, seq_len) * grad_accum
    tflops_achieved = (step_flops / sec_per_step) / 1e12
    mfu_pct = 100.0 * tflops_achieved / peak_tflops
    hbm_gb = _peak_hbm_gb()

    return {
        "seq_len": seq_len,
        "num_loops": num_loops,
        "micro_bs": micro_bs,
        "grad_accum": grad_accum,
        "step_tokens": step_tokens,
        "compile_s": round(compile_s, 2),
        "step_ms": round(sec_per_step * 1000.0, 2),
        "tok_per_sec": round(tok_per_sec, 1),
        "tflops": round(tflops_achieved, 2),
        "mfu_pct": round(mfu_pct, 2),
        "peak_hbm_gb": round(hbm_gb, 2) if hbm_gb is not None else None,
        "loss": round(float(metrics["loss"]), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LoopMoE TPU v6e-1 MFU and HBM.")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to benchmark for R=1.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        nargs="+",
        default=[1],
        help="Loop counts to sweep across --seq-lens (plus optional 4K R=8 point).",
    )
    parser.add_argument("--include-4k-8loop", action="store_true", help="Also benchmark Stage 7 (4K, R=8).")
    parser.add_argument("--microbatch-tokens", type=int, default=32_768)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--bench-steps", type=int, default=5)
    parser.add_argument("--peak-tflops", type=float, default=918.0)
    parser.add_argument("--use-csa2-long", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernels = _probe_tpu_kernels()
    print(f"Backend: {jax.default_backend()} | Devices: {jax.devices()}")
    print(f"Kernels: {json.dumps(kernels)}")

    base_cfg = LoopMoEConfig.tiny() if args.tiny else LoopMoEConfig()
    if args.use_csa2_long:
        base_cfg = LoopMoEConfig(**{**base_cfg.to_dict(), "use_csa2_indexer": True})
    print(summarize_model(base_cfg))

    points: list[tuple[int, int]] = [(s, r) for r in args.loops for s in args.seq_lens]
    if args.include_4k_8loop and (4096, 8) not in points:
        points.append((4096, 8))

    results: list[dict[str, Any]] = []
    print("\n| SeqLen | Loops | Batch | Compile(s) | Step(ms) | Tok/s | TFLOPS | MFU(%) | PeakHBM(GB) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for seq_len, loops in points:
        try:
            res = _run_single_point(
                base_cfg=base_cfg,
                seq_len=seq_len,
                num_loops=loops,
                microbatch_tokens=args.microbatch_tokens,
                grad_accum=args.grad_accum,
                warmup_steps=args.warmup_steps,
                bench_steps=args.bench_steps,
                peak_tflops=args.peak_tflops,
                use_csa2=args.use_csa2_long,
                seed=args.seed,
            )
            results.append(res)
            hbm_str = f"{res['peak_hbm_gb']:.2f}" if res["peak_hbm_gb"] is not None else "N/A"
            print(
                f"| {res['seq_len']} | {res['num_loops']} | {res['micro_bs']}x{res['grad_accum']} "
                f"| {res['compile_s']:.2f} | {res['step_ms']:.2f} | {res['tok_per_sec']:,.0f} "
                f"| {res['tflops']:.1f} | {res['mfu_pct']:.1f}% | {hbm_str} |"
            )
        except Exception as exc:
            print(f"| {seq_len} | {loops} | ERR | - | - | - | - | - | {type(exc).__name__}: {exc} |")
            results.append({"seq_len": seq_len, "num_loops": loops, "error": str(exc)})

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"kernels": kernels, "results": results}, indent=2), encoding="utf-8")
        print(f"\nSaved benchmark report to {out_path}")


if __name__ == "__main__":
    main()
