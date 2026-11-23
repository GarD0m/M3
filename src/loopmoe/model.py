"""1B 1-Layer DeepSeek-V4.1-Flash LoopMoE model, chunked cross-entropy, and R=1->8 expansion."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import optax

from .config import LoopMoEConfig, count_analytic_params
from .engram import apply_engram_step, init_engram_params, prefetch_engram_embeddings
from .iter_adaln import (
    apply_adaln,
    compute_iter_adaln_modulations,
    init_iter_adaln_params,
)
from .mhc import (
    collapse_streams,
    expand_streams,
    init_mhc_params,
    mhc_compute_gates,
    mhc_post_merge,
    rms_norm,
)
from .mla import apply_mla, init_mla_params
from .moe import apply_moe, init_moe_params


def _get_dtype(dtype_name: str) -> jnp.dtype:
    return getattr(jnp, dtype_name)


def init_model_params(key: jax.Array, cfg: LoopMoEConfig) -> dict[str, Any]:
    """Initialize the complete 1B 1-Layer DeepSeek-V4.1-Flash LoopMoE parameter pytree."""
    dtype = _get_dtype(cfg.dtype_name)
    keys = jax.random.split(key, 7)

    params: dict[str, Any] = {
        "embed_tokens": jax.random.normal(keys[0], (cfg.vocab_size, cfg.d_model), dtype=dtype) * 0.02,
        "final_norm_w": jnp.ones((cfg.d_model,), dtype=dtype),
        "iter_adaln": init_iter_adaln_params(keys[1], cfg, dtype=dtype),
        "mla": init_mla_params(keys[2], cfg, dtype=dtype),
        "moe": init_moe_params(keys[3], cfg, dtype=dtype),
    }
    if not cfg.tie_word_embeddings:
        params["lm_head"] = jax.random.normal(keys[4], (cfg.d_model, cfg.vocab_size), dtype=dtype) * (
            1.0 / jnp.sqrt(float(cfg.d_model))
        )
    if cfg.use_mhc:
        params["mhc"] = init_mhc_params(keys[5], cfg, dtype=dtype)
    if cfg.use_engram:
        params["engram"] = init_engram_params(keys[6], cfg, dtype=dtype)
    return params


def count_tree_params(tree: Any) -> int:
    """Count total scalar elements in a JAX pytree."""
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(tree))


def _single_loop_step(
    params: dict[str, Any],
    x_streams: jax.Array,
    engram_e: jax.Array | None,
    loop_idx: jax.Array | int,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Execute one recurrent iteration of Engram + IterAdaLN + Single-Pass mHC(MLA) + Single-Pass mHC(MoE)."""
    with jax.named_scope("loop_step"):
        alphas = params["iter_adaln"]["loop_alpha"][loop_idx]  # [3]
        dt = x_streams.dtype
        alpha_engram = alphas[0].astype(dt)
        alpha_attn = alphas[1].astype(dt)
        alpha_moe = alphas[2].astype(dt)

        # 1. Context-aware Engram conditional memory refinement
        if cfg.use_engram:
            def _engram_sublayer(p, xs, ee, ae):
                with jax.named_scope("engram_step"):
                    h_summary = collapse_streams(xs)
                    engram_delta, gate_mean = apply_engram_step(p["engram"], h_summary, ee, cfg)
                    return xs + ae * engram_delta[:, :, None, :], gate_mean

            engram_fn = (
                jax.checkpoint(_engram_sublayer)
                if (cfg.remat_sublayers or cfg.remat_moe)
                else _engram_sublayer
            )
            x_streams, engram_gate_mean = engram_fn(params, x_streams, engram_e, alpha_engram)
        else:
            engram_gate_mean = jnp.asarray(0.0, dtype=jnp.float32)

        # 2. LoopMoE IterAdaLN modulation jointly conditioned on h_t and loop_idx
        with jax.named_scope("iter_adaln"):
            h_mod = collapse_streams(x_streams)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = compute_iter_adaln_modulations(
                params["iter_adaln"], h_mod, loop_idx
            )

        # 3. MLA sublayer inside Single-Pass mHC
        def _attn_sublayer(p, xs, sa, sca, ga, aa):
            with jax.named_scope("mhc_attn_sublayer"):
                if cfg.use_mhc:
                    h_in_a, _, h_post_a, h_res_a = mhc_compute_gates(
                        p["mhc"]["attn"], xs, cfg.mhc_sinkhorn_iters, cfg.rms_norm_eps
                    )
                else:
                    h_in_a = xs[:, :, 0, :]

                h_in_a_norm = apply_adaln(
                    h_in_a, p["iter_adaln"]["norm_attn_w"], sca, sa, cfg.rms_norm_eps
                )
                attn_out, _ = apply_mla(p["mla"], h_in_a_norm, cfg)
                attn_out = attn_out * ga.astype(dt)

                if cfg.use_mhc:
                    return mhc_post_merge(xs, attn_out, h_post_a, h_res_a, loop_gate=aa)
                return xs + aa * attn_out[:, :, None, :]

        attn_fn = jax.checkpoint(_attn_sublayer) if cfg.remat_sublayers else _attn_sublayer
        x_streams = attn_fn(params, x_streams, shift_a, scale_a, gate_a, alpha_attn)

        # 4. DeepSeek MoE sublayer inside Single-Pass mHC
        def _moe_sublayer(p, xs, sm, scm, gm, am, lidx):
            with jax.named_scope("mhc_moe_sublayer"):
                if cfg.use_mhc:
                    h_in_m, _, h_post_m, h_res_m = mhc_compute_gates(
                        p["mhc"]["moe"], xs, cfg.mhc_sinkhorn_iters, cfg.rms_norm_eps
                    )
                else:
                    h_in_m = xs[:, :, 0, :]

                h_in_m_norm = apply_adaln(
                    h_in_m, p["iter_adaln"]["norm_moe_w"], scm, sm, cfg.rms_norm_eps
                )
                if cfg.remat_moe and not cfg.remat_sublayers:
                    moe_out, ec, aux_l = jax.checkpoint(
                        lambda pm, h_in, li: apply_moe(pm, h_in, li, cfg)
                    )(p["moe"], h_in_m_norm, lidx)
                else:
                    moe_out, ec, aux_l = apply_moe(p["moe"], h_in_m_norm, lidx, cfg)
                moe_out = moe_out * gm.astype(dt)

                if cfg.use_mhc:
                    next_xs = mhc_post_merge(xs, moe_out, h_post_m, h_res_m, loop_gate=am)
                else:
                    next_xs = xs + am * moe_out[:, :, None, :]
                return next_xs, ec, aux_l

        moe_fn = jax.checkpoint(_moe_sublayer) if cfg.remat_sublayers else _moe_sublayer
        x_streams, expert_counts, seq_aux_loss = moe_fn(
            params, x_streams, shift_m, scale_m, gate_m, alpha_moe, loop_idx
        )

        return x_streams, (expert_counts, seq_aux_loss, engram_gate_mean)


def forward_hidden(
    params: dict[str, Any],
    input_ids: jax.Array,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Run embedding + single pre-loop Engram gather + R-loop LoopMoE block + final RMSNorm."""
    x = params["embed_tokens"][input_ids]  # [B, S, D]
    with jax.named_scope("engram_prefetch"):
        engram_e = (
            prefetch_engram_embeddings(params["engram"], input_ids, cfg) if cfg.use_engram else None
        )

    stream_init = params["mhc"]["stream_init"] if cfg.use_mhc else None
    x_streams = expand_streams(x, stream_init)

    def scan_body(carry: jax.Array, loop_idx: jax.Array):
        return _single_loop_step(params, carry, engram_e, loop_idx, cfg)

    use_loop_remat = cfg.remat_per_loop and (cfg.num_loops > 1)
    step_fn = jax.checkpoint(scan_body) if use_loop_remat else scan_body
    loop_indices = jnp.arange(cfg.num_loops, dtype=jnp.int32)
    x_streams, (expert_counts, seq_aux_losses, gate_means) = jax.lax.scan(
        step_fn, x_streams, loop_indices
    )
    seq_aux_loss = jnp.mean(seq_aux_losses)
    engram_gate_mean = jnp.mean(gate_means)

    hidden = rms_norm(collapse_streams(x_streams), params["final_norm_w"], eps=cfg.rms_norm_eps)
    aux = {
        "expert_counts": expert_counts,  # [num_loops, E]
        "seq_aux_loss": seq_aux_loss,
        "engram_gate_mean": engram_gate_mean,
    }
    return hidden, aux


def get_lm_head_weight(params: dict[str, Any], cfg: LoopMoEConfig) -> jax.Array:
    """Return [D, V] LM head projection matrix (tied or untied)."""
    if cfg.tie_word_embeddings:
        return params["embed_tokens"].T
    return params["lm_head"]


def forward_logits(
    params: dict[str, Any],
    input_ids: jax.Array,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute full logits [B, S, V] (intended for small eval batches / unit tests)."""
    hidden, aux = forward_hidden(params, input_ids, cfg)
    logits = (hidden @ get_lm_head_weight(params, cfg)).astype(jnp.float32)
    return logits, aux


def _ce_chunk_loss_and_lse(
    logits_bf16: jax.Array,
    tc: jax.Array,
    mc: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    lse = jax.scipy.special.logsumexp(logits_bf16.astype(jnp.float32), axis=-1)
    z_t = (
        jnp.take_along_axis(logits_bf16, tc[:, None], axis=-1)
        .squeeze(-1)
        .astype(jnp.float32)
    )
    return jnp.sum((lse - z_t) * mc), lse


def _ce_chunk_dlogits(
    logits_bf16: jax.Array,
    lse_c: jax.Array,
    tc: jax.Array,
    mc: jax.Array,
    vocab_idx: jax.Array,
    scale_factor: jax.Array,
    out_dtype: jnp.dtype,
) -> jax.Array:
    # Pure elementwise pipeline fuses exp -> where -> scale -> bf16 in VPU without FP32 [chunk, V] HBM spills
    probs = jnp.exp(logits_bf16.astype(jnp.float32) - lse_c[:, None])
    diff = jnp.where(vocab_idx == tc[:, None], probs - 1.0, probs)
    return (diff * (mc * scale_factor)[:, None]).astype(out_dtype)


def _run_scan_ce_fwd(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate scalar CE loss over chunks while retaining only [num_chunks, chunk] LSE."""

    def _fwd_step(loss_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array]):
        hc, tc, mc = chunk_in
        logits_bf16 = hc @ lm_head_weight
        chunk_loss, lse = _ce_chunk_loss_and_lse(logits_bf16, tc, mc)
        return loss_acc + chunk_loss, lse

    total_loss, lse_chunks = jax.lax.scan(
        _fwd_step,
        jnp.asarray(0.0, dtype=jnp.float32),
        (h_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, lse_chunks


@jax.custom_vjp
def _scan_chunked_ce(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _ = _run_scan_ce_fwd(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)
    return out


def _scan_chunked_ce_fwd(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, lse_chunks = _run_scan_ce_fwd(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)
    return out, (h_chunks, lm_head_weight, t_chunks, m_chunks, lse_chunks, denom)


def _scan_chunked_ce_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    h_chunks, lm_head_weight, t_chunks, m_chunks, lse_chunks, denom = res
    vocab_size = lm_head_weight.shape[1]
    vocab_idx = jnp.arange(vocab_size, dtype=t_chunks.dtype)[None, :]
    scale_factor = g_out / denom

    def _bwd_step(
        d_lm_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        hc, tc, mc, lse_c = chunk_in
        logits_bf16 = hc @ lm_head_weight
        d_logits = _ce_chunk_dlogits(logits_bf16, lse_c, tc, mc, vocab_idx, scale_factor, hc.dtype)
        d_hc = (d_logits @ lm_head_weight.T).astype(hc.dtype)
        d_lm_acc = d_lm_acc + (hc.T @ d_logits).astype(d_lm_acc.dtype)
        return d_lm_acc, d_hc

    d_lm_w, d_h_chunks = jax.lax.scan(
        _bwd_step,
        jnp.zeros_like(lm_head_weight),
        (h_chunks, t_chunks, m_chunks, lse_chunks),
        unroll=1,
    )
    return d_h_chunks, d_lm_w, None, None, None


_scan_chunked_ce.defvjp(_scan_chunked_ce_fwd, _scan_chunked_ce_bwd)


def _run_scan_ce_fwd_vd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Layout-native [V, D] chunked CE forward avoiding [V, D] -> [D, V] transpose copy."""

    def _fwd_step(loss_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array]):
        hc, tc, mc = chunk_in
        logits_bf16 = jax.lax.dot_general(hc, lm_weight_vd, (((1,), (1,)), ((), ())))
        chunk_loss, lse = _ce_chunk_loss_and_lse(logits_bf16, tc, mc)
        return loss_acc + chunk_loss, lse

    total_loss, lse_chunks = jax.lax.scan(
        _fwd_step,
        jnp.asarray(0.0, dtype=jnp.float32),
        (h_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, lse_chunks


@jax.custom_vjp
def _scan_chunked_ce_vd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _ = _run_scan_ce_fwd_vd(h_chunks, lm_weight_vd, t_chunks, m_chunks, denom)
    return out


def _scan_chunked_ce_vd_fwd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, lse_chunks = _run_scan_ce_fwd_vd(h_chunks, lm_weight_vd, t_chunks, m_chunks, denom)
    return out, (h_chunks, lm_weight_vd, t_chunks, m_chunks, lse_chunks, denom)


def _scan_chunked_ce_vd_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    h_chunks, lm_weight_vd, t_chunks, m_chunks, lse_chunks, denom = res
    vocab_size = lm_weight_vd.shape[0]
    vocab_idx = jnp.arange(vocab_size, dtype=t_chunks.dtype)[None, :]
    scale_factor = g_out / denom

    def _bwd_step(
        d_w_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        hc, tc, mc, lse_c = chunk_in
        logits_bf16 = jax.lax.dot_general(hc, lm_weight_vd, (((1,), (1,)), ((), ())))
        d_logits = _ce_chunk_dlogits(logits_bf16, lse_c, tc, mc, vocab_idx, scale_factor, hc.dtype)
        d_hc = (d_logits @ lm_weight_vd).astype(hc.dtype)
        d_w_acc = d_w_acc + jax.lax.dot_general(
            d_logits, hc, (((0,), (0,)), ((), ()))
        ).astype(d_w_acc.dtype)
        return d_w_acc, d_hc

    d_lm_w_vd, d_h_chunks = jax.lax.scan(
        _bwd_step,
        jnp.zeros_like(lm_weight_vd),
        (h_chunks, t_chunks, m_chunks, lse_chunks),
        unroll=1,
    )
    return d_h_chunks, d_lm_w_vd, None, None, None


_scan_chunked_ce_vd.defvjp(_scan_chunked_ce_vd_fwd, _scan_chunked_ce_vd_bwd)


def _run_saved_logits_ce_fwd(
    h_chunks: jax.Array,
    lm_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
    weight_is_vd: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute scalar CE over sub-chunks while saving only BF16 logits and FP32 LSE."""

    def _fwd_step(loss_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array]):
        hc, tc, mc = chunk_in
        logits_bf16 = (
            jax.lax.dot_general(hc, lm_weight, (((1,), (1,)), ((), ())))
            if weight_is_vd
            else (hc @ lm_weight)
        )
        chunk_loss, lse = _ce_chunk_loss_and_lse(logits_bf16, tc, mc)
        return loss_acc + chunk_loss, (logits_bf16, lse)

    total_loss, (logits_chunks, lse_chunks) = jax.lax.scan(
        _fwd_step,
        jnp.asarray(0.0, dtype=jnp.float32),
        (h_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, logits_chunks, lse_chunks


def _run_saved_logits_ce_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
    weight_is_vd: bool,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    h_chunks, lm_weight, logits_chunks, t_chunks, m_chunks, lse_chunks, denom = res
    vocab_size = lm_weight.shape[0] if weight_is_vd else lm_weight.shape[1]
    vocab_idx = jnp.arange(vocab_size, dtype=t_chunks.dtype)[None, :]
    scale_factor = g_out / denom

    def _bwd_step(
        d_w_acc: jax.Array,
        chunk_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        hc, logits_c, tc, mc, lse_c = chunk_in
        d_logits = _ce_chunk_dlogits(logits_c, lse_c, tc, mc, vocab_idx, scale_factor, hc.dtype)
        if weight_is_vd:
            d_hc = (d_logits @ lm_weight).astype(hc.dtype)
            d_w_acc = d_w_acc + jax.lax.dot_general(
                d_logits, hc, (((0,), (0,)), ((), ()))
            ).astype(d_w_acc.dtype)
        else:
            d_hc = (d_logits @ lm_weight.T).astype(hc.dtype)
            d_w_acc = d_w_acc + (hc.T @ d_logits).astype(d_w_acc.dtype)
        return d_w_acc, d_hc

    d_lm_w, d_h_chunks = jax.lax.scan(
        _bwd_step,
        jnp.zeros_like(lm_weight),
        (h_chunks, logits_chunks, t_chunks, m_chunks, lse_chunks),
        unroll=1,
    )
    return d_h_chunks, d_lm_w, None, None, None


@jax.custom_vjp
def _saved_logits_ce(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _, _ = _run_saved_logits_ce_fwd(h_chunks, lm_head_weight, t_chunks, m_chunks, denom, False)
    return out


def _saved_logits_ce_fwd(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, logits_chunks, lse_chunks = _run_saved_logits_ce_fwd(
        h_chunks, lm_head_weight, t_chunks, m_chunks, denom, False
    )
    return out, (h_chunks, lm_head_weight, logits_chunks, t_chunks, m_chunks, lse_chunks, denom)


def _saved_logits_ce_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    return _run_saved_logits_ce_bwd(res, g_out, False)


_saved_logits_ce.defvjp(_saved_logits_ce_fwd, _saved_logits_ce_bwd)


@jax.custom_vjp
def _saved_logits_ce_vd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _, _ = _run_saved_logits_ce_fwd(h_chunks, lm_weight_vd, t_chunks, m_chunks, denom, True)
    return out


def _saved_logits_ce_vd_fwd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, logits_chunks, lse_chunks = _run_saved_logits_ce_fwd(
        h_chunks, lm_weight_vd, t_chunks, m_chunks, denom, True
    )
    return out, (h_chunks, lm_weight_vd, logits_chunks, t_chunks, m_chunks, lse_chunks, denom)


def _saved_logits_ce_vd_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    return _run_saved_logits_ce_bwd(res, g_out, True)


_saved_logits_ce_vd.defvjp(_saved_logits_ce_vd_fwd, _saved_logits_ce_vd_bwd)


def chunked_cross_entropy_loss(
    hidden: jax.Array,
    lm_head_weight: jax.Array,
    targets: jax.Array,
    chunk_size: int = 2048,
    loss_mask: jax.Array | None = None,
    weight_is_vd: bool = False,
) -> jax.Array:
    """Memory-efficient chunked linear projection + softmax cross-entropy.

    Supports `[D, V]` (`weight_is_vd=False`) or layout-native tied `[V, D]` (`weight_is_vd=True`)
    without transposing the full `[65536, 2048]` embedding table. When `chunk_size >= total_tokens`,
    uses BF16-saved sub-chunking (`<=8192` tok/chunk) to skip backward logits matmul recompute
    without allocating full FP32 `[N, V]` logits in HBM.
    """
    bsz, seq_len, d_model = hidden.shape
    total_tokens = bsz * seq_len
    h_flat = hidden.reshape(total_tokens, d_model)
    t_flat = targets.reshape(total_tokens)
    m_flat = (
        jnp.ones((total_tokens,), dtype=jnp.float32)
        if loss_mask is None
        else loss_mask.reshape(total_tokens).astype(jnp.float32)
    )

    denom = jnp.maximum(jnp.sum(m_flat), 1.0)
    if chunk_size >= total_tokens:
        sub_chunk = min(total_tokens, 8192)
        if total_tokens % sub_chunk != 0:
            sub_chunk = math.gcd(total_tokens, sub_chunk)
            if sub_chunk == 1 and total_tokens <= 8192:
                sub_chunk = total_tokens
        num_chunks = total_tokens // sub_chunk
        h_chunks = h_flat.reshape(num_chunks, sub_chunk, d_model)
        t_chunks = t_flat.reshape(num_chunks, sub_chunk)
        m_chunks = m_flat.reshape(num_chunks, sub_chunk)
        saved_fn = _saved_logits_ce_vd if weight_is_vd else _saved_logits_ce
        return saved_fn(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)

    effective_chunk = min(chunk_size, total_tokens)
    if total_tokens % effective_chunk != 0:
        effective_chunk = math.gcd(total_tokens, chunk_size)
        if effective_chunk == 1 and total_tokens <= 4096:
            effective_chunk = total_tokens

    num_chunks = total_tokens // effective_chunk
    h_chunks = h_flat.reshape(num_chunks, effective_chunk, d_model)
    t_chunks = t_flat.reshape(num_chunks, effective_chunk)
    m_chunks = m_flat.reshape(num_chunks, effective_chunk)
    scan_fn = _scan_chunked_ce_vd if weight_is_vd else _scan_chunked_ce
    return scan_fn(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)


def forward_model(
    params: dict[str, Any],
    input_ids: jax.Array,
    targets: jax.Array,
    cfg: LoopMoEConfig,
    loss_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Forward pass returning total loss (CE + tiny seq aux) and auxiliary telemetry."""
    hidden, aux = forward_hidden(params, input_ids, cfg)
    if cfg.tie_word_embeddings:
        lm_w = params["embed_tokens"]
        weight_is_vd = True
    else:
        lm_w = params["lm_head"]
        weight_is_vd = False
    with jax.named_scope("chunked_ce_loss"):
        ce_loss = chunked_cross_entropy_loss(
            hidden,
            lm_w,
            targets,
            chunk_size=cfg.chunk_ce_size,
            loss_mask=loss_mask,
            weight_is_vd=weight_is_vd,
        )
    total_loss = ce_loss + aux["seq_aux_loss"]
    aux["ce_loss"] = ce_loss
    return total_loss, aux


def expand_checkpoint_1_to_8_loops(
    params: dict[str, Any],
    key: jax.Array,
    noise_std: float = 1e-3,
) -> dict[str, Any]:
    """Expand a Phase-1 (R=1) trained checkpoint to Phase-2 (R=8) preserving step-0 logits."""
    new_params = jax.tree_util.tree_map(lambda x: x, params)
    adaln = dict(new_params["iter_adaln"])

    loop_emb = adaln["loop_emb"]
    max_loops = loop_emb.shape[0]
    base_emb = loop_emb[0:1]
    noise = jax.random.normal(key, (max_loops - 1, loop_emb.shape[1]), dtype=loop_emb.dtype) * noise_std
    expanded_emb = jnp.concatenate([base_emb, jnp.repeat(base_emb, max_loops - 1, axis=0) + noise], axis=0)
    adaln["loop_emb"] = expanded_emb

    # Force t=0 active (preserving Phase 1 value) and t=1..7 exact zero for step-0 identity
    loop_alpha = jnp.zeros_like(adaln["loop_alpha"]).at[0].set(adaln["loop_alpha"][0])
    adaln["loop_alpha"] = loop_alpha
    new_params["iter_adaln"] = adaln

    # Broadcast learned Phase-1 router bias to all loop steps as warm start
    moe = dict(new_params["moe"])
    base_bias = moe["router_bias"][0:1]
    moe["router_bias"] = jnp.repeat(base_bias, max_loops, axis=0)
    new_params["moe"] = moe
    return new_params


def summarize_model(cfg: LoopMoEConfig) -> str:
    """Format human-readable parameter and capacity breakdown."""
    stats = count_analytic_params(cfg)
    return (
        f"LoopMoE(d={cfg.d_model}, V={cfg.vocab_size}, R={cfg.num_loops}, "
        f"experts={cfg.n_routed_experts}x{cfg.d_ff_expert} top-{cfg.top_k}+shared) | "
        f"Total={stats['total']/1e6:.2f}M | Emb={stats['embedding']/1e6:.2f}M | "
        f"Engram={stats['engram_total']/1e6:.2f}M ({stats['engram_sparse_ratio']*100:.1f}% sparse) | "
        f"MLA={stats['mla']/1e6:.2f}M | MoE={stats['moe_total']/1e6:.2f}M | "
        f"Active/loop(non-emb)={stats['active_per_loop_non_emb']/1e6:.2f}M "
        f"(Attn:FFN = 1:{stats['attn_to_ffn_active_ratio']:.2f})"
    )
