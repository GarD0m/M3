"""TPU v5e-1 / v6e-1 MFU, XProf profiler, XLA roofline, and bottleneck diagnostics (Test branch)."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from loopmoe.config import TPU_SPECS, LoopMoEConfig, estimate_step_flops
from loopmoe.engram import apply_engram_step, prefetch_engram_embeddings
from loopmoe.iter_adaln import compute_iter_adaln_modulations
from loopmoe.mhc import collapse_streams, mhc_compute_gates, mhc_post_merge
from loopmoe.mla import apply_mla
from loopmoe.model import (
    chunked_cross_entropy_loss,
    forward_model,
    get_lm_head_weight,
    init_model_params,
    summarize_model,
)
from loopmoe.moe import apply_moe
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


def _bench_op(fn: Callable[..., Any], args: tuple[Any, ...], warmup: int = 2, iters: int = 5) -> float:
    """JIT-compile and measure steady-state execution latency in milliseconds."""
    jitted = jax.jit(fn)
    out = jitted(*args)
    jax.block_until_ready(out)
    for _ in range(max(0, warmup - 1)):
        out = jitted(*args)
        jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = jitted(*args)
    jax.block_until_ready(out)
    return ((time.perf_counter() - t0) / float(iters)) * 1000.0


def run_module_breakdown(
    cfg: LoopMoEConfig,
    seq_len: int,
    micro_bs: int,
    warmup: int = 2,
    iters: int = 5,
    seed: int = 42,
    autotune_ctx: Any = None,
) -> dict[str, Any]:
    """Micro-profile each architectural component individually to expose exact bottlenecks."""
    params = init_model_params(jax.random.PRNGKey(seed), cfg)
    opt = create_muon_adamw_optimizer(OptimConfig(warmup_steps=100))
    opt_state = opt.init(params)

    rng = np.random.default_rng(seed)
    x_ids = jnp.asarray(rng.integers(64, cfg.vocab_size, size=(micro_bs, seq_len), dtype=np.int32))
    y_ids = jnp.roll(x_ids, -1, axis=-1)
    dt = getattr(jnp, cfg.dtype_name)
    h_2d = jax.random.normal(jax.random.PRNGKey(seed + 1), (micro_bs, seq_len, cfg.d_model), dtype=dt)
    x_streams = jax.random.normal(
        jax.random.PRNGKey(seed + 2),
        (micro_bs, seq_len, cfg.mhc_streams if cfg.use_mhc else 1, cfg.d_model),
        dtype=dt,
    )
    engram_e = prefetch_engram_embeddings(params["engram"], x_ids, cfg) if cfg.use_engram else None
    lm_w = get_lm_head_weight(params, cfg)

    timings_ms: dict[str, float] = {}
    ctx = autotune_ctx if autotune_ctx is not None else contextlib.nullcontext()
    with ctx:
        if cfg.use_engram:
            timings_ms["engram_prefetch"] = round(
                _bench_op(lambda p, ids: prefetch_engram_embeddings(p, ids, cfg), (params["engram"], x_ids), warmup, iters),
                3,
            )
            timings_ms["engram_step"] = round(
                _bench_op(lambda p, h, e: apply_engram_step(p, h, e, cfg), (params["engram"], h_2d, engram_e), warmup, iters),
                3,
            )

        timings_ms["iter_adaln"] = round(
            _bench_op(
                lambda p, xs: compute_iter_adaln_modulations(p, collapse_streams(xs), 0),
                (params["iter_adaln"], x_streams),
                warmup,
                iters,
            ),
            3,
        )

        if cfg.use_mhc:
            def _mhc_full(p, xs, out_h):
                h_in, _, h_post, h_res = mhc_compute_gates(p, xs, cfg.mhc_sinkhorn_iters, cfg.rms_norm_eps)
                merged = mhc_post_merge(xs, out_h, h_post, h_res, 1.0)
                return h_in, merged

            timings_ms["mhc_gates_and_merge"] = round(
                _bench_op(_mhc_full, (params["mhc"]["attn"], x_streams, h_2d), warmup, iters), 3
            )

        timings_ms["mla"] = round(
            _bench_op(lambda p, h: apply_mla(p, h, cfg), (params["mla"], h_2d), warmup, iters), 3
        )
        timings_ms["moe_total"] = round(
            _bench_op(lambda p, h: apply_moe(p, h, 0, cfg), (params["moe"], h_2d), warmup, iters), 3
        )

        # Isolate MoE dispatch sort/gather overhead vs shared expert
        def _moe_sort_only(p, h):
            x_flat = h.reshape(-1, cfg.d_model)
            scores = jax.nn.sigmoid((x_flat @ p["router_w"]).astype(jnp.float32))
            _, topk_idx = jax.lax.top_k(scores + p["router_bias"][0][None, :], cfg.top_k)
            flat_e = topk_idx.reshape(-1)
            order = jnp.argsort(flat_e)
            return x_flat[order // cfg.top_k], jnp.bincount(flat_e, length=cfg.n_routed_experts)

        def _moe_shared_only(p, h):
            xf = h.reshape(-1, cfg.d_model)
            g, u = jnp.split(xf @ p["shared_w_gate_up"], 2, axis=-1)
            return (jax.nn.silu(g) * u) @ p["shared_w_down"]

        timings_ms["moe_sort_dispatch_only"] = round(_bench_op(_moe_sort_only, (params["moe"], h_2d), warmup, iters), 3)
        timings_ms["moe_shared_only"] = round(_bench_op(_moe_shared_only, (params["moe"], h_2d), warmup, iters), 3)

        timings_ms["chunked_ce_loss"] = round(
            _bench_op(
                lambda h, w, t: chunked_cross_entropy_loss(h, w, t, chunk_size=cfg.chunk_ce_size),
                (h_2d, lm_w, y_ids),
                warmup,
                iters,
            ),
            3,
        )

        # Free activation scratch before full fwd+bwd and donated optimizer step to fit 16 GB HBM
        del x_streams, h_2d, engram_e, lm_w
        gc.collect()

        timings_ms["full_fwd"] = round(
            _bench_op(lambda p, xi, yi: forward_model(p, xi, yi, cfg), (params, x_ids, y_ids), warmup, iters),
            3,
        )
        val_grad_fn = jax.value_and_grad(lambda p, xi, yi: forward_model(p, xi, yi, cfg), has_aux=True)
        timings_ms["full_fwd_bwd"] = round(_bench_op(val_grad_fn, (params, x_ids, y_ids), warmup, iters), 3)

        (_, _), sample_grads = jax.jit(val_grad_fn)(params, x_ids, y_ids)
        jax.block_until_ready(sample_grads)
        del val_grad_fn
        gc.collect()

    @jax.jit(donate_argnums=(0, 1))
    def _opt_step(p, s, g):
        upd, ns = opt.update(g, s, p)
        return optax.apply_updates(p, upd), ns

    params, opt_state = _opt_step(params, opt_state, sample_grads)
    jax.block_until_ready(params)
    for _ in range(max(0, warmup - 1)):
        params, opt_state = _opt_step(params, opt_state, sample_grads)
        jax.block_until_ready(params)
    t0 = time.perf_counter()
    for _ in range(iters):
        params, opt_state = _opt_step(params, opt_state, sample_grads)
    jax.block_until_ready(params)
    timings_ms["optimizer_muon_adamw"] = round(
        ((time.perf_counter() - t0) / float(iters)) * 1000.0, 3
    )

    del params, opt_state, sample_grads, x_ids, y_ids, _opt_step
    gc.collect()
    jax.clear_caches()

    bwd_ms = max(0.0, timings_ms["full_fwd_bwd"] - timings_ms["full_fwd"])
    timings_ms["bwd_only_est"] = round(bwd_ms, 3)
    timings_ms["bwd_to_fwd_ratio"] = round(bwd_ms / max(timings_ms["full_fwd"], 1e-6), 2)
    return timings_ms


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
    profile_steps: int,
    seed: int,
    autotune: bool = True,
) -> tuple[dict[str, Any], Any]:
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
    x_np = rng.integers(64, cfg.vocab_size, size=(grad_accum, micro_bs, seq_len), dtype=np.int32)
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

        # Optional JAX Profiler XProf / Perfetto trace capture
        if profile_dir:
            pdir = Path(profile_dir)
            pdir.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(str(pdir))
            try:
                for s_idx in range(profile_steps):
                    with jax.profiler.StepTraceAnnotation("train_step", step_num=s_idx):
                        params, opt_state, metrics = compiled(params, opt_state, bx, by)
                        jax.block_until_ready(metrics["loss"])
            finally:
                jax.profiler.stop_trace()

        # Timed steady-state benchmark
        t_start = time.perf_counter()
        for _ in range(bench_steps):
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

    return (
        {
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
        },
        autotune_ctx,
    )


def diagnose_bottlenecks(
    cfg: LoopMoEConfig,
    preset: str,
    kernels: dict[str, bool],
    results: list[dict[str, Any]],
    breakdown: dict[str, Any] | None,
    microbatch: int,
    seq_len: int,
) -> list[str]:
    """Generate actionable, prioritized bottleneck diagnostics from measured profiler telemetry."""
    spec = TPU_SPECS[preset]
    mxu = int(spec["mxu_size"])
    microbatch_tokens = microbatch * seq_len
    notes: list[str] = []

    if not kernels["on_tpu"]:
        notes.append("[Env] Running on non-TPU backend; re-run on Colab TPU v5e-1/v6e-1 for true MXU MFU.")
    elif not kernels["tokamax_ragged_dot"]:
        notes.append("[MoE Kernel] tokamax.ragged_dot unavailable; falling back to jax.lax.ragged_dot or gather-einsum.")

    avg_tok_per_expert = (microbatch_tokens * cfg.top_k) / float(cfg.n_routed_experts)
    if avg_tok_per_expert < mxu:
        notes.append(
            f"[MXU Tiling] At microbatch={microbatch} ({microbatch_tokens} tok), avg tokens/expert={avg_tok_per_expert:.1f} < MXU={mxu}. "
            f"Experts are under-tiled (memory-bound); raise --microbatch (if HBM allows) or use larger d_ff_expert."
        )
    elif avg_tok_per_expert < 2 * mxu:
        notes.append(
            f"[MXU Tiling] Avg tokens/expert={avg_tok_per_expert:.1f} (~{avg_tok_per_expert/mxu:.2f}x MXU={mxu}). "
            f"Load imbalance can drop busy experts below 1 full tile; watch exp_cv and tune router_bias_update_rate."
        )

    valid_res = [r for r in results if "error" not in r]
    if valid_res:
        max_hbm = max((r.get("peak_hbm_gb") or 0.0) for r in valid_res)
        cap_gb = spec["hbm_capacity_gb"]
        if max_hbm > 0.85 * cap_gb:
            notes.append(
                f"[HBM Pressure] Peak HBM={max_hbm:.2f} GB / {cap_gb:.0f} GB (>85%). "
                f"Reduce --microbatch, lower --chunk-ce-size (e.g. 2048), or enable --remat-sublayers."
            )
        elif max_hbm > 0.0 and max_hbm < 0.65 * cap_gb and cfg.remat_sublayers:
            notes.append(
                f"[Remat Opportunity] Peak HBM={max_hbm:.2f} GB / {cap_gb:.0f} GB (<65%) with remat_sublayers=True. "
                f"Try --no-remat-sublayers or raising --microbatch to trade spare HBM for ~20-30% higher MFU."
            )
        for r in valid_res:
            if r.get("bound_regime") == "MEMORY-BOUND":
                notes.append(
                    f"[Roofline @ seq={r['seq_len']},R={r['num_loops']}] XLA arithmetic intensity="
                    f"{r.get('arithmetic_intensity')} FLOPs/B < {preset} ridge ({spec['ridge_flops_per_byte']:.0f}); memory-bandwidth bound."
                )

    if breakdown:
        fwd = max(breakdown.get("full_fwd", 0.0), 1e-6)
        tot_step = fwd + breakdown.get("bwd_only_est", 0.0) + breakdown.get("optimizer_muon_adamw", 0.0)
        moe_pct = 100.0 * breakdown.get("moe_total", 0.0) / fwd
        sort_pct_of_moe = 100.0 * breakdown.get("moe_sort_dispatch_only", 0.0) / max(
            breakdown.get("moe_total", 1e-6), 1e-6
        )
        ce_pct = 100.0 * breakdown.get("chunked_ce_loss", 0.0) / fwd
        opt_pct = 100.0 * breakdown.get("optimizer_muon_adamw", 0.0) / max(tot_step, 1e-6)
        mhc_pct = 100.0 * breakdown.get("mhc_gates_and_merge", 0.0) / fwd

        if moe_pct > 45.0:
            notes.append(f"[MoE Dominance] MoE takes {moe_pct:.1f}% of forward pass (sort/dispatch is {sort_pct_of_moe:.1f}% of MoE).")
        if sort_pct_of_moe > 20.0:
            notes.append(
                f"[MoE Dispatch Overhead] argsort/token gather is {sort_pct_of_moe:.1f}% of MoE time; "
                f"consider padded/capacity-factor routing or Pallas fused scatter."
            )
        if ce_pct > 20.0:
            notes.append(
                f"[Chunked CE Overhead] chunked_ce_loss takes {ce_pct:.1f}% of forward; "
                f"try larger --chunk-ce-size (e.g. 4096/8192) if HBM allows."
            )
        if mhc_pct > 15.0:
            notes.append(
                f"[mHC Overhead] mHC Sinkhorn+merge takes {mhc_pct:.1f}% of forward; consider fewer Sinkhorn iters (10 vs 20)."
            )
        if opt_pct > 15.0:
            notes.append(
                f"[Optimizer Overhead] Un-amortized Muon+AdamW step takes {opt_pct:.1f}% of 1-microbatch step "
                f"({cfg.n_routed_experts} experts); amortize with larger --grad-accum (e.g. 16 on v5e-1)."
            )
        if breakdown.get("bwd_to_fwd_ratio", 0.0) > 2.7:
            notes.append(
                f"[Remat Tax] bwd/fwd ratio={breakdown['bwd_to_fwd_ratio']}x (>2.7x); heavy recomputation detected."
            )

    return notes


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
        "--autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in slow Tokamax autotuning sweep on TPU (explicit tile configs are already set in moe.py/mla.py).",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--bench-steps", type=int, default=5)
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--breakdown", action="store_true", help="Run per-module micro-profiler.")
    parser.add_argument("--breakdown-seq-len", type=int, default=4096)
    parser.add_argument("--profile-dir", type=str, default=None, help="Output dir for JAX XProf/Perfetto trace.")
    parser.add_argument("--profile-seq-len", type=int, default=None)
    parser.add_argument("--profile-loops", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--use-csa2-long", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--out-json", type=str, default=None)
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
    base_cfg = LoopMoEConfig(
        **{
            **base_cfg.to_dict(),
            "chunk_ce_size": chunk_ce,
            "remat_sublayers": remat_sub,
            "use_csa2_indexer": bool(args.use_csa2_long),
        }
    )
    print(summarize_model(base_cfg))
    print(
        f"Execution config: microbatch={micro_bs} (seq={seq_lens[0]} -> {micro_bs * seq_lens[0]} tok), "
        f"grad_accum={grad_accum}, chunk_ce={chunk_ce}, remat_sublayers={remat_sub}, autotune={args.autotune}"
    )

    points: list[tuple[int, int]] = [(s, r) for r in args.loops for s in seq_lens]
    if args.include_4k_8loop:
        target_4k = 32 if args.tiny else 4096
        if (target_4k, 8) not in points:
            points.append((target_4k, 8))

    prof_target_seq = args.profile_seq_len or (seq_lens[0] if args.tiny else (4096 if 4096 in seq_lens else seq_lens[0]))
    breakdown_target_seq = seq_lens[0] if args.tiny else args.breakdown_seq_len
    breakdown_autotune_ctx: Any = contextlib.nullcontext()
    results: list[dict[str, Any]] = []
    print("\n| SeqLen | Loops | Batch | Compile(s) | Step(ms) | Tok/s | TFLOPS | MFU(%) | PeakHBM(GB) | XLA_AI | Regime |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for seq_len, loops in points:
        trace_here = args.profile_dir if (args.profile_dir and seq_len == prof_target_seq and loops == args.profile_loops) else None
        try:
            res, pt_autotune_ctx = _run_single_point(
                base_cfg=base_cfg,
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
                profile_steps=args.profile_steps,
                seed=args.seed,
                autotune=args.autotune,
            )
            if seq_len == breakdown_target_seq and loops == 1:
                breakdown_autotune_ctx = pt_autotune_ctx
            results.append(res)
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
            results.append({"seq_len": seq_len, "num_loops": loops, "error": str(exc)})

    breakdown: dict[str, Any] | None = None
    if args.breakdown:
        b_cfg = base_cfg.with_stage(seq_len=breakdown_target_seq, num_loops=1, use_csa2_indexer=args.use_csa2_long)
        breakdown = run_module_breakdown(
            b_cfg,
            breakdown_target_seq,
            micro_bs,
            warmup=args.warmup_steps,
            iters=args.bench_steps,
            seed=args.seed,
            autotune_ctx=breakdown_autotune_ctx,
        )
        print(f"\n=== Module Micro-Breakdown (seq={breakdown_target_seq}, microbatch={micro_bs}, R=1) ===")
        fwd_ref = max(breakdown["full_fwd"], 1e-6)
        for k, v in breakdown.items():
            if k in ("bwd_to_fwd_ratio",):
                print(f"  {k:24s}: {v}x")
            else:
                pct = 100.0 * float(v) / fwd_ref
                print(f"  {k:24s}: {v:8.3f} ms ({pct:5.1f}% of full_fwd)")

    diagnoses = diagnose_bottlenecks(base_cfg, args.tpu_preset, kernels, results, breakdown, micro_bs, seq_lens[0])
    print("\n=== Automated Bottleneck Diagnostics ===")
    for note in diagnoses:
        print(f"- {note}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preset": args.tpu_preset,
            "kernels": kernels,
            "config": base_cfg.to_dict(),
            "results": results,
            "breakdown": breakdown,
            "diagnoses": diagnoses,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved benchmark & profiler report to {out_path}")


if __name__ == "__main__":
    main()
