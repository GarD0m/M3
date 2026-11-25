"""TPU v5e-1 / v6e-1 MFU, XProf profiler, and XLA roofline benchmark (Test branch)."""

from __future__ import annotations

import argparse
import contextlib
import gc
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

from loopmoe.config import TPU_SPECS, LoopMoEConfig, estimate_step_flops
from loopmoe.model import init_model_params, summarize_model
from loopmoe.optim import OptimConfig, create_muon_adamw_optimizer
from train_tpu import make_train_step, maybe_tokamax_autotune


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


def _extract_xla_analysis(compiled: Any, step_tokens: int, ridge_fpb: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        mem = compiled.memory_analysis()
        if mem is not None:
            arg_b = int(getattr(mem, "argument_size_in_bytes", 0))
            out_b = int(getattr(mem, "output_size_in_bytes", 0))
            tmp_b = int(getattr(mem, "temp_size_in_bytes", 0))
            alias_b = int(getattr(mem, "alias_size_in_bytes", 0))
            total_b = max(0, arg_b + out_b + tmp_b - alias_b)
            out["xla_arg_gb"] = round(arg_b / (1024**3), 3)
            out["xla_temp_gb"] = round(tmp_b / (1024**3), 3)
            out["xla_alias_gb"] = round(alias_b / (1024**3), 3)
            out["xla_est_total_gb"] = round(total_b / (1024**3), 3)
    except Exception as exc:
        out["memory_analysis_err"] = str(exc)

    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, list) and cost:
            cost = cost[0]
        if isinstance(cost, dict):
            flops = float(cost.get("flops", 0.0))
            bytes_acc = float(cost.get("bytes accessed", cost.get("bytes_accessed", 0.0)))
            out["xla_gflops"] = round(flops / 1e9, 2)
            out["xla_bytes_gb"] = round(bytes_acc / (1024**3), 3)
            if bytes_acc > 0.0 and flops > 0.0:
                ai = flops / bytes_acc
                out["arithmetic_intensity"] = round(ai, 2)
                out["bound_regime"] = "COMPUTE-BOUND" if ai >= ridge_fpb else "MEMORY-BOUND"
                out["xla_flops_per_token"] = round(flops / max(1, step_tokens), 1)
    except Exception as exc:
        out["cost_analysis_err"] = str(exc)

    return out


@contextlib.contextmanager
def _maybe_profile(profile_dir: str | None):
    if not profile_dir:
        yield
        return
    pdir = Path(profile_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(pdir))
    try:
        yield
    finally:
        jax.profiler.stop_trace()


def _run_single_point(
    base_cfg: LoopMoEConfig,
    seq_len: int,
    num_loops: int,
    microbatch: int,
    grad_accum: int,
    warmup_steps: int,
    bench_steps: int,
    peak_tflops: float,
    ridge_fpb: float,
    use_csa2: bool,
    profile_dir: str | None,
    seed: int,
    autotune: bool = True,
) -> dict[str, Any]:
    micro_bs = max(1, microbatch)
    micro_tok = micro_bs * seq_len

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
    x_np = rng.integers(64, cfg.vocab_size, size=(grad_accum, micro_bs, seq_len), dtype=np.uint16)
    y_np = np.roll(x_np, -1, axis=-1)
    bx = jnp.asarray(x_np)
    by = jnp.asarray(y_np)

    autotune_ctx = maybe_tokamax_autotune(cfg, params, bx[0], by[0], enabled=autotune)

    # Explicit AOT lower + compile to collect memory_analysis & cost_analysis
    with autotune_ctx:
        t0 = time.perf_counter()
        compiled = train_step.lower(params, opt_state, bx, by).compile()
        compile_s = time.perf_counter() - t0

        step_tokens = grad_accum * micro_tok
        xla_stats = _extract_xla_analysis(compiled, step_tokens, ridge_fpb)

        # First run + warmup
        params, opt_state, metrics = compiled(params, opt_state, bx, by)
        jax.block_until_ready(metrics["loss"])
        for _ in range(max(0, warmup_steps - 1)):
            params, opt_state, metrics = compiled(params, opt_state, bx, by)
            jax.block_until_ready(metrics["loss"])

        # Unified timed steady-state benchmark & optional XProf trace
        with _maybe_profile(profile_dir):
            t_start = time.perf_counter()
            for s_idx in range(bench_steps):
                with jax.profiler.StepTraceAnnotation("train_step", step_num=s_idx):
                    params, opt_state, metrics = compiled(params, opt_state, bx, by)
            jax.block_until_ready(metrics["loss"])
            elapsed = max(time.perf_counter() - t_start, 1e-6)

    sec_per_step = elapsed / float(bench_steps)
    tok_per_sec = float(step_tokens) / sec_per_step
    step_flops = estimate_step_flops(cfg, micro_bs, seq_len) * grad_accum
    tflops_achieved = (step_flops / sec_per_step) / 1e12
    mfu_pct = 100.0 * tflops_achieved / peak_tflops
    hbm_gb = _peak_hbm_gb()
    loss_val = round(float(metrics["loss"]), 4)

    del params, opt_state, metrics, bx, by, compiled, train_step
    gc.collect()
    jax.clear_caches()

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
        "loss": loss_val,
        **xla_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark & Profile LoopMoE on TPU v5e-1 / v6e-1.")
    parser.add_argument("--tpu-preset", choices=["v5e-1", "v6e-1"], default="v5e-1")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None)
    parser.add_argument("--loops", type=int, nargs="+", default=[1])
    parser.add_argument("--include-4k-8loop", action="store_true", help="Include Stage 7 (4K, R=8).")
    parser.add_argument("--microbatch", type=int, default=None, help="Microbatch size in sequences per step (default: 4 on v5e-1, 8 on v6e-1, 2 on --tiny).")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps (default: 16 on v5e-1, 8 on v6e-1, 1 on --tiny).")
    parser.add_argument("--chunk-ce-size", type=int, default=None)
    parser.add_argument("--remat-sublayers", action="store_true", default=None)
    parser.add_argument("--no-remat-sublayers", action="store_false", dest="remat_sublayers")
    parser.add_argument(
        "--remat-moe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint only the MoE sublayer (saves ~3 GB HBM at microbatch>=8 without recomputing MLA/mHC).",
    )
    parser.add_argument(
        "--autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in slow Tokamax autotuning sweep on TPU (explicit tile configs are already set in moe.py/mla.py).",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--bench-steps", type=int, default=5)
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Optional output dir to capture XProf/Perfetto traces during --bench-steps.",
    )
    parser.add_argument("--use-csa2-long", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = TPU_SPECS[args.tpu_preset]
    peak_tflops = args.peak_tflops or spec["peak_tflops_bf16"]
    ridge_fpb = spec["ridge_flops_per_byte"]
    micro_bs = args.microbatch or (2 if args.tiny else int(spec["default_microbatch"]))
    grad_accum = args.grad_accum or (1 if args.tiny else int(spec["default_grad_accum"]))
    chunk_ce = args.chunk_ce_size or (16 if args.tiny else int(spec["default_chunk_ce_size"]))
    seq_lens = args.seq_lens or ([32] if args.tiny else [4096])

    kernels = _probe_tpu_kernels()
    print(f"Preset: {args.tpu_preset} (peak={peak_tflops:.0f} TF, BW={spec['hbm_bw_gbps']:.0f} GB/s, HBM={spec['hbm_capacity_gb']:.0f} GB, ridge={ridge_fpb:.1f} F/B)")
    print(f"Backend: {jax.default_backend()} | Devices: {jax.devices()}")
    print(f"Kernels: {json.dumps(kernels)}")

    base_cfg = LoopMoEConfig.tiny() if args.tiny else LoopMoEConfig()
    remat_sub = False if args.remat_sublayers is None else args.remat_sublayers
    remat_moe = False if args.remat_moe is None else args.remat_moe
    base_cfg = LoopMoEConfig(
        **{
            **base_cfg.to_dict(),
            "chunk_ce_size": chunk_ce,
            "remat_sublayers": remat_sub,
            "remat_moe": remat_moe,
            "use_csa2_indexer": bool(args.use_csa2_long),
        }
    )
    print(summarize_model(base_cfg))
    print(
        f"Execution config: microbatch={micro_bs} (seq={seq_lens[0]} -> {micro_bs * seq_lens[0]} tok), "
        f"grad_accum={grad_accum}, chunk_ce={chunk_ce}, remat_sublayers={remat_sub}, remat_moe={remat_moe}, autotune={args.autotune}"
    )

    points: list[tuple[int, int]] = [(s, r) for r in args.loops for s in seq_lens]
    if args.include_4k_8loop:
        target_4k = 32 if args.tiny else 4096
        if (target_4k, 8) not in points:
            points.append((target_4k, 8))

    print("\n| SeqLen | Loops | Batch | Compile(s) | Step(ms) | Tok/s | TFLOPS | MFU(%) | PeakHBM(GB) | XLA_AI | Regime |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for seq_len, loops in points:
        point_cfg = base_cfg
        if args.tpu_preset == "v5e-1" and not args.tiny and seq_len >= 16384:
            auto_overrides: dict[str, Any] = {}
            if args.remat_moe is None and not base_cfg.remat_moe:
                auto_overrides["remat_moe"] = True
            if args.chunk_ce_size is None and base_cfg.chunk_ce_size > 4096:
                auto_overrides["chunk_ce_size"] = 4096
            if auto_overrides:
                point_cfg = LoopMoEConfig(**{**base_cfg.to_dict(), **auto_overrides})
                print(
                    f"[Auto-HBM v5e-1 @ seq={seq_len}] "
                    + ", ".join(f"{k}={v}" for k, v in auto_overrides.items())
                )
        trace_here = (
            None
            if not args.profile_dir
            else (
                args.profile_dir
                if len(points) == 1
                else str(Path(args.profile_dir) / f"s{seq_len}_r{loops}")
            )
        )
        try:
            res = _run_single_point(
                base_cfg=point_cfg,
                seq_len=seq_len,
                num_loops=loops,
                microbatch=micro_bs,
                grad_accum=grad_accum,
                warmup_steps=args.warmup_steps,
                bench_steps=args.bench_steps,
                peak_tflops=peak_tflops,
                ridge_fpb=ridge_fpb,
                use_csa2=args.use_csa2_long,
                profile_dir=trace_here,
                seed=args.seed,
                autotune=args.autotune,
            )
            hbm_str = f"{res['peak_hbm_gb']:.2f}" if res["peak_hbm_gb"] is not None else "N/A"
            ai_str = f"{res['arithmetic_intensity']:.1f}" if "arithmetic_intensity" in res else "N/A"
            reg_str = res.get("bound_regime", "N/A")
            print(
                f"| {res['seq_len']} | {res['num_loops']} | {res['micro_bs']}x{res['grad_accum']} "
                f"| {res['compile_s']:.2f} | {res['step_ms']:.2f} | {res['tok_per_sec']:,.0f} "
                f"| {res['tflops']:.1f} | {res['mfu_pct']:.1f}% | {hbm_str} | {ai_str} | {reg_str} |"
            )
        except Exception as exc:
            print(f"| {seq_len} | {loops} | ERR | - | - | - | - | - | - | - | {type(exc).__name__}: {exc} |")


if __name__ == "__main__":
    main()
