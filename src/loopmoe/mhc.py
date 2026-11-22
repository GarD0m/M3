"""DeepSeek-V4.1-Flash Single-Pass Manifold-Constrained Hyper-Connections (mHC)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import LoopMoEConfig


def rms_norm(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
) -> jax.Array:
    """Compute RMSNorm in float32 for numerical stability and cast back to input dtype."""
    x_f32 = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
    y = (x_f32 * inv_rms).astype(x.dtype)
    if weight is not None:
        y = y * weight.astype(x.dtype)
    return y


def sinkhorn_knopp(logits: jax.Array, num_iters: int = 20, eps: float = 1e-6) -> jax.Array:
    """Project log-space square matrices (..., N, N) onto the Birkhoff polytope."""
    x = jnp.clip(logits.astype(jnp.float32), -20.0, 20.0)
    # Subtract max per row before exp to prevent overflow while preserving Sinkhorn fixed point
    x = x - jnp.max(x, axis=-1, keepdims=True)
    m = jnp.exp(x)

    def body_fn(_: int, mat: jax.Array) -> jax.Array:
        mat = mat / (jnp.sum(mat, axis=-1, keepdims=True) + eps)
        mat = mat / (jnp.sum(mat, axis=-2, keepdims=True) + eps)
        return mat

    return jax.lax.fori_loop(0, num_iters, body_fn, m)


def init_mhc_sublayer_params(
    key: jax.Array,
    d_model: int,
    n_streams: int,
    alpha_init: float = 0.01,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize parameters for one Single-Pass mHC sublayer wrapper."""
    in_dim = n_streams * d_model
    k_pre, k_post, k_res = jax.random.split(key, 3)
    scale = 1.0 / jnp.sqrt(float(in_dim))

    # Initialize b_res near identity so initial H_res ~ I after Sinkhorn-Knopp
    b_res_init = jnp.full((n_streams, n_streams), -6.0, dtype=jnp.float32)
    b_res_init = b_res_init.at[jnp.arange(n_streams), jnp.arange(n_streams)].set(0.0)

    return {
        "norm_w": jnp.ones((in_dim,), dtype=dtype),
        "phi_pre": jax.random.normal(k_pre, (in_dim, n_streams), dtype=dtype) * scale,
        "phi_post": jax.random.normal(k_post, (in_dim, n_streams), dtype=dtype) * scale,
        "phi_res": jax.random.normal(k_res, (in_dim, n_streams * n_streams), dtype=dtype) * scale,
        "b_pre": jnp.full((n_streams,), -jnp.log(float(n_streams - 1) if n_streams > 1 else 1.0), dtype=jnp.float32),
        "b_post": jnp.zeros((n_streams,), dtype=jnp.float32),
        "b_res": b_res_init,
        "alpha_pre": jnp.asarray(alpha_init, dtype=jnp.float32),
        "alpha_post": jnp.asarray(alpha_init, dtype=jnp.float32),
        "alpha_res": jnp.asarray(alpha_init, dtype=jnp.float32),
    }


def init_mhc_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, dict[str, jax.Array] | jax.Array]:
    """Initialize all Single-Pass mHC parameters for the 1-layer LoopMoE block."""
    k1, k2 = jax.random.split(key)
    return {
        "stream_init": jnp.ones((cfg.mhc_streams,), dtype=dtype),
        "attn": init_mhc_sublayer_params(
            k1, cfg.d_model, cfg.mhc_streams, cfg.mhc_alpha_init, dtype=dtype
        ),
        "moe": init_mhc_sublayer_params(
            k2, cfg.d_model, cfg.mhc_streams, cfg.mhc_alpha_init, dtype=dtype
        ),
    }


def expand_streams(x: jax.Array, stream_init: jax.Array | None = None) -> jax.Array:
    """Expand [B, S, D] -> [B, S, n_streams, D] residual streams."""
    if stream_init is None:
        return x[:, :, None, :]
    return x[:, :, None, :] * stream_init[None, None, :, None].astype(x.dtype)


def collapse_streams(x_streams: jax.Array) -> jax.Array:
    """Collapse [B, S, n_streams, D] -> [B, S, D] via mean projection."""
    return jnp.mean(x_streams, axis=2)


def mhc_compute_gates(
    params: dict[str, jax.Array],
    x_streams: jax.Array,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute pre-aggregated input and mHC mixing matrices (H_pre, H_post, H_res)."""
    bsz, seq_len, n_streams, d_model = x_streams.shape
    dt = x_streams.dtype
    x_flat = x_streams.reshape(bsz, seq_len, n_streams * d_model)
    x_norm = rms_norm(x_flat, params["norm_w"], eps=eps)

    # Fused linear projections in input dtype, Sinkhorn manifold projection in float32
    pre_dyn = (x_norm @ params["phi_pre"]).astype(jnp.float32)
    post_dyn = (x_norm @ params["phi_post"]).astype(jnp.float32)
    res_dyn = (x_norm @ params["phi_res"]).astype(jnp.float32).reshape(
        bsz, seq_len, n_streams, n_streams
    )

    h_pre = jax.nn.sigmoid(params["alpha_pre"] * pre_dyn + params["b_pre"])
    h_pre = (h_pre / (jnp.sum(h_pre, axis=-1, keepdims=True) + eps)).astype(dt)

    h_post = (2.0 * jax.nn.sigmoid(params["alpha_post"] * post_dyn + params["b_post"])).astype(dt)
    h_res_logits = params["alpha_res"] * res_dyn + params["b_res"]
    h_res = sinkhorn_knopp(h_res_logits, num_iters=sinkhorn_iters, eps=eps).astype(dt)

    h_in = jnp.einsum("bsn,bsnd->bsd", h_pre, x_streams)
    return h_in, h_pre, h_post, h_res


def mhc_post_merge(
    x_streams: jax.Array,
    sublayer_out: jax.Array,
    h_post: jax.Array,
    h_res: jax.Array,
    loop_gate: jax.Array | float = 1.0,
) -> jax.Array:
    """Single-Pass mHC post-residual update with smooth identity gating for loop expansion."""
    dt = x_streams.dtype
    mixed = jnp.einsum("bsij,bsjd->bsid", h_res.astype(dt), x_streams) + (
        h_post[:, :, :, None].astype(dt) * sublayer_out[:, :, None, :].astype(dt)
    )
    gate = jnp.asarray(loop_gate, dtype=dt)
    one = jnp.asarray(1.0, dtype=dt)
    return (one - gate) * x_streams + gate * mixed
