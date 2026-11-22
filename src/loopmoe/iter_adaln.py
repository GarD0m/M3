"""LoopMoE IterAdaLN (Iteration-Aware Adaptive Layer Normalization, arXiv:2606.04438)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import LoopMoEConfig
from .mhc import rms_norm


def init_iter_adaln_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize IterAdaLN bottleneck MLP, loop embeddings, and per-loop gates."""
    k_emb, k_wh, k_wt, k_out = jax.random.split(key, 4)
    d = cfg.d_model
    b = cfg.iter_adaln_bottleneck

    # loop_alpha[:, 0]=engram, [:, 1]=attn, [:, 2]=moe
    # t=0 starts active (1.0); t>=1 start at 0.0 so R=1 -> R=8 preserves logits at step 0.
    loop_alpha = jnp.zeros((cfg.max_loops, 3), dtype=jnp.float32).at[0].set(1.0)

    return {
        "loop_emb": jax.random.normal(k_emb, (cfg.max_loops, d), dtype=dtype) * 0.02,
        "w_h": jax.random.normal(k_wh, (d, b), dtype=dtype) * (1.0 / jnp.sqrt(float(d))),
        "w_t": jax.random.normal(k_wt, (d, b), dtype=dtype) * (1.0 / jnp.sqrt(float(d))),
        # Zero-init w_out so dynamic modulation starts cleanly around identity at init
        "w_out": jnp.zeros((b, 6 * d), dtype=dtype),
        "b_out": jnp.zeros((6 * d,), dtype=dtype),
        "norm_attn_w": jnp.ones((d,), dtype=dtype),
        "norm_moe_w": jnp.ones((d,), dtype=dtype),
        "loop_alpha": loop_alpha,
    }


def compute_iter_adaln_modulations(
    params: dict[str, jax.Array],
    h_summary: jax.Array,
    loop_idx: jax.Array | int,
) -> tuple[jax.Array, ...]:
    """Compute (shift_attn, scale_attn, gate_attn, shift_moe, scale_moe, gate_moe) in native dtype."""
    dt = h_summary.dtype
    t_emb = params["loop_emb"][loop_idx].astype(dt)  # [D]
    proj = jax.nn.silu(
        (h_summary @ params["w_h"].astype(dt)) + (t_emb @ params["w_t"].astype(dt))[None, None, :]
    )
    mod = (proj @ params["w_out"].astype(dt)) + params["b_out"].astype(dt)
    shift_attn, scale_attn, gate_attn, shift_moe, scale_moe, gate_moe = jnp.split(mod, 6, axis=-1)
    one = jnp.asarray(1.0, dtype=dt)
    return (
        shift_attn,
        one + scale_attn,
        one + gate_attn,
        shift_moe,
        one + scale_moe,
        one + gate_moe,
    )


def apply_adaln(
    x: jax.Array,
    norm_weight: jax.Array,
    scale: jax.Array,
    shift: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """Apply RMSNorm followed by IterAdaLN affine modulation in native dtype."""
    x_norm = rms_norm(x, norm_weight, eps=eps)
    return x_norm * scale.astype(x.dtype) + shift.astype(x.dtype)
