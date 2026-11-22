"""DeepSeek-V4.1-Flash Engram Conditional Memory (O(1) multi-head N-gram lookup, arXiv:2601.07372)."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .config import LoopMoEConfig
from .mhc import rms_norm


def _build_hash_constants(cfg: LoopMoEConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically generate odd hash multipliers, seeds, table sizes, and offsets."""
    rng = np.random.default_rng(cfg.engram_seed)
    table_sizes = np.asarray(cfg.engram_table_sizes, dtype=np.uint32)
    offsets = np.zeros((len(table_sizes),), dtype=np.int32)
    if len(table_sizes) > 1:
        offsets[1:] = np.cumsum(table_sizes[:-1], dtype=np.int32)

    max_order = max(cfg.engram_orders)
    num_tables = len(table_sizes)
    multipliers = np.zeros((num_tables, max_order), dtype=np.uint32)
    seeds = np.zeros((num_tables,), dtype=np.uint32)
    orders_arr = np.zeros((num_tables,), dtype=np.int32)

    idx = 0
    for order in cfg.engram_orders:
        for _ in range(cfg.engram_hash_heads):
            orders_arr[idx] = order
            seeds[idx] = np.uint32(rng.integers(1, 2**31 - 1, dtype=np.uint64) * 2 + 1)
            for pos in range(order):
                # Force odd multiplier so Z_{2^32} multiplication is bijective
                multipliers[idx, pos] = np.uint32(
                    rng.integers(1, 2**31 - 1, dtype=np.uint64) * 2 + 1
                )
            idx += 1

    return table_sizes, offsets, multipliers, seeds


def init_engram_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize contiguous prime-bucketed Engram tables and context gating projections."""
    if not cfg.use_engram:
        return {}

    k_tab, k_wk, k_wv, k_conv = jax.random.split(key, 4)
    total_slots = sum(cfg.engram_table_sizes)
    d = cfg.d_model
    d_engram = cfg.d_engram

    return {
        "tables": jax.random.normal(k_tab, (total_slots, cfg.engram_head_dim), dtype=dtype) * 0.02,
        "w_k": jax.random.normal(k_wk, (d_engram, d), dtype=dtype) * (1.0 / jnp.sqrt(float(d_engram))),
        "w_v": jax.random.normal(k_wv, (d_engram, d), dtype=dtype) * (1.0 / jnp.sqrt(float(d_engram))),
        "norm_q_w": jnp.ones((d,), dtype=dtype),
        "norm_k_w": jnp.ones((d,), dtype=dtype),
        "norm_conv_w": jnp.ones((d,), dtype=dtype),
        # Depthwise causal conv initialized near zero so residual stream starts clean
        "conv_w": jax.random.normal(k_conv, (cfg.engram_conv_kernel, d), dtype=dtype) * 1e-3,
        "conv_b": jnp.zeros((d,), dtype=dtype),
    }


def prefetch_engram_embeddings(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    cfg: LoopMoEConfig,
) -> jax.Array | None:
    """Gather all N-gram conditional memory vectors ONCE prior to the recurrent loop."""
    if not cfg.use_engram:
        return None

    table_sizes_np, offsets_np, mult_np, seeds_np = _build_hash_constants(cfg)
    max_order = max(cfg.engram_orders)
    bsz, seq_len = input_ids.shape

    # Shift IDs by +1 so token 0 stays distinct from left-padding 0
    ids_u32 = input_ids.astype(jnp.uint32) + jnp.uint32(1)
    padded = jnp.pad(ids_u32, ((0, 0), (max_order - 1, 0)), constant_values=jnp.uint32(0))

    # Build shifted token windows [max_order, B, S]
    windows = jnp.stack(
        [padded[:, max_order - 1 - pos : max_order - 1 - pos + seq_len] for pos in range(max_order)],
        axis=0,
    )

    mult_j = jnp.asarray(mult_np, dtype=jnp.uint32)  # [T, max_order]
    seeds_j = jnp.asarray(seeds_np, dtype=jnp.uint32)  # [T]
    sizes_j = jnp.asarray(table_sizes_np, dtype=jnp.uint32)  # [T]
    offsets_j = jnp.asarray(offsets_np, dtype=jnp.int32)  # [T]

    # Multiplicative-XOR over all orders/heads vectorized: mult_j[:, j] == 0 for unused positions >= order
    # Result shape: [B, S, num_tables]
    acc = jnp.broadcast_to(seeds_j[None, None, :], (bsz, seq_len, len(table_sizes_np)))
    for pos in range(max_order):
        tok_pos = windows[pos][:, :, None]  # [B, S, 1]
        acc = jnp.bitwise_xor(acc, tok_pos * mult_j[None, None, :, pos])

    local_idx = (acc % sizes_j[None, None, :]).astype(jnp.int32)
    global_idx = local_idx + offsets_j[None, None, :]

    gathered = params["tables"][global_idx]  # [B, S, num_tables, engram_head_dim]
    return gathered.reshape(bsz, seq_len, cfg.d_engram)


def _causal_depthwise_conv1d(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    dilation: int,
) -> jax.Array:
    """Apply causal 1D depthwise convolution over sequence axis=1 in native dtype."""
    kernel_size = weight.shape[0]
    pad_len = (kernel_size - 1) * dilation
    seq_len = x.shape[1]
    padded = jnp.pad(x, ((0, 0), (pad_len, 0), (0, 0)))

    acc = jnp.zeros(x.shape, dtype=x.dtype)
    w_cast = weight.astype(x.dtype)
    for k in range(kernel_size):
        start = pad_len - k * dilation
        sl = padded[:, start : start + seq_len, :]
        acc = acc + sl * w_cast[k][None, None, :]
    acc = acc + bias[None, None, :].astype(x.dtype)
    return jax.nn.silu(acc)


def apply_engram_step(
    params: dict[str, jax.Array],
    h_t: jax.Array,
    engram_e: jax.Array | None,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, jax.Array]:
    """Compute context-aware Engram gating and causal depthwise conv refinement for one loop."""
    if not cfg.use_engram or engram_e is None:
        return jnp.zeros_like(h_t), jnp.asarray(0.0, dtype=jnp.float32)

    k = engram_e @ params["w_k"]
    v = engram_e @ params["w_v"]

    h_norm = rms_norm(h_t, params["norm_q_w"], eps=cfg.rms_norm_eps).astype(jnp.float32)
    k_norm = rms_norm(k, params["norm_k_w"], eps=cfg.rms_norm_eps).astype(jnp.float32)

    gate_logits = jnp.sum(h_norm * k_norm, axis=-1, keepdims=True) / jnp.sqrt(float(cfg.d_model))
    gate = jax.nn.sigmoid(gate_logits)
    u = gate.astype(h_t.dtype) * v

    u_norm = rms_norm(u, params["norm_conv_w"], eps=cfg.rms_norm_eps)
    conv_out = _causal_depthwise_conv1d(
        u_norm,
        params["conv_w"],
        params["conv_b"],
        dilation=cfg.engram_conv_dilation,
    )
    delta = u + conv_out
    return delta, jnp.mean(gate)
