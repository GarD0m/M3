"""1B 1-Layer DeepSeek-V4.1-Flash LoopMoE model, chunked cross-entropy, and R=1->8 expansion."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

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
    alphas = params["iter_adaln"]["loop_alpha"][loop_idx]  # [3]
    dt = x_streams.dtype
    alpha_engram = alphas[0].astype(dt)
    alpha_attn = alphas[1].astype(dt)
    alpha_moe = alphas[2].astype(dt)

    # 1. Context-aware Engram conditional memory refinement
    if cfg.use_engram:
        h_summary = collapse_streams(x_streams)
        engram_delta, engram_gate_mean = apply_engram_step(params["engram"], h_summary, engram_e, cfg)
        x_streams = x_streams + alpha_engram * engram_delta[:, :, None, :]
    else:
        engram_gate_mean = jnp.asarray(0.0, dtype=jnp.float32)

    # 2. LoopMoE IterAdaLN modulation jointly conditioned on h_t and loop_idx
    h_mod = collapse_streams(x_streams)
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = compute_iter_adaln_modulations(
        params["iter_adaln"], h_mod, loop_idx
    )

    # 3. MLA sublayer inside Single-Pass mHC
    if cfg.use_mhc:
        h_in_a, _, h_post_a, h_res_a = mhc_compute_gates(
            params["mhc"]["attn"], x_streams, cfg.mhc_sinkhorn_iters, cfg.rms_norm_eps
        )
    else:
        h_in_a = x_streams[:, :, 0, :]

    h_in_a_norm = apply_adaln(
        h_in_a, params["iter_adaln"]["norm_attn_w"], scale_a, shift_a, cfg.rms_norm_eps
    )
    attn_out, _ = apply_mla(params["mla"], h_in_a_norm, cfg)
    attn_out = attn_out * gate_a.astype(dt)

    if cfg.use_mhc:
        x_streams = mhc_post_merge(x_streams, attn_out, h_post_a, h_res_a, loop_gate=alpha_attn)
    else:
        x_streams = x_streams + alpha_attn * attn_out[:, :, None, :]

    # 4. DeepSeek MoE sublayer inside Single-Pass mHC
    if cfg.use_mhc:
        h_in_m, _, h_post_m, h_res_m = mhc_compute_gates(
            params["mhc"]["moe"], x_streams, cfg.mhc_sinkhorn_iters, cfg.rms_norm_eps
        )
    else:
        h_in_m = x_streams[:, :, 0, :]

    h_in_m_norm = apply_adaln(
        h_in_m, params["iter_adaln"]["norm_moe_w"], scale_m, shift_m, cfg.rms_norm_eps
    )
    moe_out, expert_counts, seq_aux_loss = apply_moe(params["moe"], h_in_m_norm, loop_idx, cfg)
    moe_out = moe_out * gate_m.astype(dt)

    if cfg.use_mhc:
        x_streams = mhc_post_merge(x_streams, moe_out, h_post_m, h_res_m, loop_gate=alpha_moe)
    else:
        x_streams = x_streams + alpha_moe * moe_out[:, :, None, :]

    return x_streams, (expert_counts, seq_aux_loss, engram_gate_mean)


def forward_hidden(
    params: dict[str, Any],
    input_ids: jax.Array,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Run embedding + single pre-loop Engram gather + R-loop LoopMoE block + final RMSNorm."""
    x = params["embed_tokens"][input_ids]  # [B, S, D]
    engram_e = (
        prefetch_engram_embeddings(params["engram"], input_ids, cfg) if cfg.use_engram else None
    )

    stream_init = params["mhc"]["stream_init"] if cfg.use_mhc else None
    x_streams = expand_streams(x, stream_init)

    def scan_body(carry: jax.Array, loop_idx: jax.Array):
        return _single_loop_step(params, carry, engram_e, loop_idx, cfg)

    step_fn = jax.checkpoint(scan_body) if (cfg.remat_per_loop and cfg.num_loops > 1) else scan_body
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


def chunked_cross_entropy_loss(
    hidden: jax.Array,
    lm_head_weight: jax.Array,
    targets: jax.Array,
    chunk_size: int = 2048,
    loss_mask: jax.Array | None = None,
) -> jax.Array:
    """Memory-efficient chunked linear projection + softmax cross-entropy.

    Avoids materializing full [B*S, vocab_size] FP32 logits in HBM (critical for 16K/32K on 32GB TPU v6e).
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

    effective_chunk = min(chunk_size, total_tokens)
    if total_tokens % effective_chunk != 0:
        effective_chunk = math.gcd(total_tokens, chunk_size)
        if effective_chunk == 1 and total_tokens <= 4096:
            effective_chunk = total_tokens

    num_chunks = total_tokens // effective_chunk
    h_chunks = h_flat.reshape(num_chunks, effective_chunk, d_model)
    t_chunks = t_flat.reshape(num_chunks, effective_chunk)
    m_chunks = m_flat.reshape(num_chunks, effective_chunk)

    def _chunk_loss(chunk_in: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        hc, tc, mc = chunk_in
        logits = (hc @ lm_head_weight).astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_nll = -jnp.take_along_axis(log_probs, tc[:, None], axis=-1).squeeze(-1)
        return jnp.sum(token_nll * mc)

    remat_chunk_loss = jax.checkpoint(_chunk_loss)
    chunk_losses = jax.lax.map(remat_chunk_loss, (h_chunks, t_chunks, m_chunks))
    denom = jnp.maximum(jnp.sum(m_flat), 1.0)
    return jnp.sum(chunk_losses) / denom


def forward_model(
    params: dict[str, Any],
    input_ids: jax.Array,
    targets: jax.Array,
    cfg: LoopMoEConfig,
    loss_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Forward pass returning total loss (CE + tiny seq aux) and auxiliary telemetry."""
    hidden, aux = forward_hidden(params, input_ids, cfg)
    lm_w = get_lm_head_weight(params, cfg)
    ce_loss = chunked_cross_entropy_loss(
        hidden,
        lm_w,
        targets,
        chunk_size=cfg.chunk_ce_size,
        loss_mask=loss_mask,
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
