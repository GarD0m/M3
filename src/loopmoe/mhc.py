"""DeepSeek-V4.1-Flash Single-Pass Manifold-Constrained Hyper-Connections (mHC)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .config import LoopMoEConfig


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _rms_norm_weighted(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    inv_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True, dtype=jnp.float32) + eps
    ).astype(x.dtype)
    return (x * inv_rms) * weight.astype(x.dtype)


def _rms_norm_weighted_fwd(
    x: jax.Array, weight: jax.Array, eps: float
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    inv_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True, dtype=jnp.float32) + eps
    ).astype(x.dtype)
    y = (x * inv_rms) * weight.astype(x.dtype)
    return y, (x, weight, inv_rms)


def _rms_norm_weighted_bwd(
    eps: float,
    res: tuple[jax.Array, jax.Array, jax.Array],
    dy: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    del eps
    x, weight, inv_rms = res
    x_hat = x * inv_rms
    sum_axes = tuple(range(x.ndim - 1))
    dw = jnp.sum(dy * x_hat, axis=sum_axes, dtype=jnp.float32).astype(weight.dtype)
    g = dy * weight.astype(x.dtype)
    dot_mean = jnp.mean(g * x_hat, axis=-1, keepdims=True, dtype=jnp.float32).astype(x.dtype)
    dx = inv_rms * (g - x_hat * dot_mean)
    return dx, dw


_rms_norm_weighted.defvjp(_rms_norm_weighted_fwd, _rms_norm_weighted_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _rms_norm_unweighted(x: jax.Array, eps: float) -> jax.Array:
    inv_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True, dtype=jnp.float32) + eps
    ).astype(x.dtype)
    return x * inv_rms


def _rms_norm_unweighted_fwd(
    x: jax.Array, eps: float
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    inv_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True, dtype=jnp.float32) + eps
    ).astype(x.dtype)
    y = x * inv_rms
    return y, (x, inv_rms)


def _rms_norm_unweighted_bwd(
    eps: float,
    res: tuple[jax.Array, jax.Array],
    dy: jax.Array,
) -> tuple[jax.Array]:
    del eps
    x, inv_rms = res
    x_hat = x * inv_rms
    dot_mean = jnp.mean(dy * x_hat, axis=-1, keepdims=True, dtype=jnp.float32).astype(x.dtype)
    dx = inv_rms * (dy - x_hat * dot_mean)
    return (dx,)


_rms_norm_unweighted.defvjp(_rms_norm_unweighted_fwd, _rms_norm_unweighted_bwd)


def rms_norm(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
) -> jax.Array:
    """Compute RMSNorm in native dtype with FP32 scalar variance reduction and lean custom VJP."""
    if weight is not None:
        return _rms_norm_weighted(x, weight, eps)
    return _rms_norm_unweighted(x, eps)


def sinkhorn_knopp(logits: jax.Array, num_iters: int = 10, eps: float = 1e-6) -> jax.Array:
    """Project log-space square matrices (..., N, N) onto the Birkhoff polytope."""
    with jax.named_scope("mhc_sinkhorn"):
        x = jnp.clip(logits.astype(jnp.float32), -20.0, 20.0)
        # Subtract max per row before exp to prevent overflow while preserving Sinkhorn fixed point
        x = x - jnp.max(x, axis=-1, keepdims=True)
        m = jnp.exp(x)
        # Python unroll keeps tiny (N, N) matrices in VMEM registers instead of fori_loop HBM slices
        for _ in range(num_iters):
            m = m / (jnp.sum(m, axis=-1, keepdims=True) + eps)
            m = m / (jnp.sum(m, axis=-2, keepdims=True) + eps)
        return m


def init_mhc_sublayer_params(
    key: jax.Array,
    d_model: int,
    n_streams: int,
    alpha_init: float = 0.01,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize parameters for one Single-Pass mHC sublayer wrapper."""
    in_dim = n_streams * d_model
    res_dim = 1 if n_streams == 2 else n_streams * n_streams
    out_dim = 2 * n_streams + res_dim
    scale = 1.0 / jnp.sqrt(float(in_dim))

    if n_streams == 2:
        # Closed-form 2x2 Birkhoff: H_res = [[p, 1-p], [1-p, p]] with p = sigmoid(6.0) ~ 0.9975
        b_res_init = jnp.asarray([6.0], dtype=jnp.float32)
    else:
        # Initialize b_res near identity so initial H_res ~ I after Sinkhorn-Knopp
        b_res_init = jnp.full((n_streams, n_streams), -6.0, dtype=jnp.float32)
        b_res_init = b_res_init.at[jnp.arange(n_streams), jnp.arange(n_streams)].set(0.0)

    return {
        "norm_w": jnp.ones((in_dim,), dtype=dtype),
        "phi_all": jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale,
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
    """Collapse [B, S, n_streams, D] -> [B, S, D] in native dtype."""
    dt = x_streams.dtype
    if x_streams.shape[2] == 2:
        return jnp.asarray(0.5, dtype=dt) * (x_streams[:, :, 0, :] + x_streams[:, :, 1, :])
    return jnp.mean(x_streams, axis=2, dtype=dt)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _streams_inv_rms(x_streams: jax.Array, eps: float) -> jax.Array:
    """Compute [B, S, 1] inverse RMS in native dtype across [n_streams, D] without [B, S, n_streams*D] reshape."""
    n_streams = x_streams.shape[2]
    sq_acc = jnp.mean(jnp.square(x_streams[:, :, 0, :]), axis=-1, keepdims=True, dtype=jnp.float32)
    for s in range(1, n_streams):
        sq_acc = sq_acc + jnp.mean(
            jnp.square(x_streams[:, :, s, :]), axis=-1, keepdims=True, dtype=jnp.float32
        )
    return jax.lax.rsqrt(sq_acc * (1.0 / float(n_streams)) + eps).astype(x_streams.dtype)


def _streams_inv_rms_fwd(
    x_streams: jax.Array, eps: float
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    inv_rms = _streams_inv_rms(x_streams, eps)
    return inv_rms, (x_streams, inv_rms)


def _streams_inv_rms_bwd(
    eps: float,
    res: tuple[jax.Array, jax.Array],
    d_inv_rms: jax.Array,
) -> tuple[jax.Array]:
    del eps
    x_streams, inv_rms = res
    n_streams = x_streams.shape[2]
    d_model = x_streams.shape[3]
    inv_rms_f32 = inv_rms.astype(jnp.float32)
    scale_dt = (
        (-1.0 / float(n_streams * d_model))
        * d_inv_rms.astype(jnp.float32)
        * (inv_rms_f32 * inv_rms_f32 * inv_rms_f32)
    ).astype(x_streams.dtype)
    dx_streams = jnp.stack(
        [x_streams[:, :, s, :] * scale_dt for s in range(n_streams)],
        axis=2,
    )
    return (dx_streams,)


_streams_inv_rms.defvjp(_streams_inv_rms_fwd, _streams_inv_rms_bwd)


def mhc_compute_gates(
    params: dict[str, jax.Array],
    x_streams: jax.Array,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute pre-aggregated input and mHC mixing matrices (H_pre, H_post, H_res) in native dtype."""
    with jax.named_scope("mhc_compute_gates"):
        bsz, seq_len, n_streams, d_model = x_streams.shape
        dt = x_streams.dtype
        inv_rms = _streams_inv_rms(x_streams, eps)
        w_eff = (params["norm_w"].astype(dt)[:, None] * params["phi_all"].astype(dt)).reshape(
            n_streams, d_model, -1
        )
        streams = [x_streams[:, :, s, :] for s in range(n_streams)]
        proj = streams[0] @ w_eff[0]
        for s in range(1, n_streams):
            proj = proj + (streams[s] @ w_eff[s])
        all_dyn = proj * inv_rms

        pre_dyn = all_dyn[..., :n_streams]
        post_dyn = all_dyn[..., n_streams : 2 * n_streams]

        h_pre = jax.nn.sigmoid(
            params["alpha_pre"].astype(dt) * pre_dyn + params["b_pre"].astype(dt)
        )
        h_pre = h_pre / (jnp.sum(h_pre, axis=-1, keepdims=True) + jnp.asarray(eps, dtype=dt))

        h_post = jnp.asarray(2.0, dtype=dt) * jax.nn.sigmoid(
            params["alpha_post"].astype(dt) * post_dyn + params["b_post"].astype(dt)
        )

        if n_streams == 2:
            # Closed-form exact 2x2 Birkhoff polytope projection (zero Sinkhorn iterations)
            res_dyn = all_dyn[..., 4:5]
            p = jax.nn.sigmoid(params["alpha_res"].astype(dt) * res_dyn + params["b_res"].astype(dt))
            q = jnp.asarray(1.0, dtype=dt) - p
            h_res = jnp.stack(
                [jnp.concatenate([p, q], axis=-1), jnp.concatenate([q, p], axis=-1)],
                axis=-2,
            )
            h_in = h_pre[:, :, 0:1] * streams[0] + h_pre[:, :, 1:2] * streams[1]
            return h_in, h_pre, h_post, h_res

        res_dyn = all_dyn[..., 2 * n_streams :].reshape(bsz, seq_len, n_streams, n_streams)
        h_res_logits = params["alpha_res"].astype(dt) * res_dyn + params["b_res"].astype(dt)
        h_res = sinkhorn_knopp(h_res_logits, num_iters=sinkhorn_iters, eps=eps).astype(dt)

        h_in = h_pre[:, :, 0, None] * streams[0]
        for i in range(1, n_streams):
            h_in = h_in + h_pre[:, :, i, None] * streams[i]
        return h_in, h_pre, h_post, h_res


def mhc_post_merge(
    x_streams: jax.Array,
    sublayer_out: jax.Array,
    h_post: jax.Array,
    h_res: jax.Array,
    loop_gate: jax.Array | float = 1.0,
) -> jax.Array:
    """Single-Pass mHC post-residual update with smooth identity gating for loop expansion."""
    with jax.named_scope("mhc_post_merge"):
        dt = x_streams.dtype
        n_streams = x_streams.shape[2]
        gate = jnp.asarray(loop_gate, dtype=dt)
        one_minus_gate = jnp.asarray(1.0, dtype=dt) - gate
        h_post_eff = gate * h_post.astype(dt)
        out_dt = sublayer_out.astype(dt)

        if n_streams == 2:
            x0 = x_streams[:, :, 0, :]
            x1 = x_streams[:, :, 1, :]
            h_res_dt = h_res.astype(dt)
            r00 = one_minus_gate + gate * h_res_dt[:, :, 0, 0]
            r01 = gate * h_res_dt[:, :, 0, 1]
            r10 = gate * h_res_dt[:, :, 1, 0]
            r11 = one_minus_gate + gate * h_res_dt[:, :, 1, 1]
            y0 = r00[..., None] * x0 + r01[..., None] * x1 + h_post_eff[:, :, 0, None] * out_dt
            y1 = r10[..., None] * x0 + r11[..., None] * x1 + h_post_eff[:, :, 1, None] * out_dt
            return jnp.stack([y0, y1], axis=2)

        eye = jnp.eye(n_streams, dtype=dt)[None, None, :, :]
        h_res_eff = one_minus_gate * eye + gate * h_res.astype(dt)
        streams = [x_streams[:, :, j, :] for j in range(n_streams)]
        out_streams = []
        for i in range(n_streams):
            acc = h_res_eff[:, :, i, 0, None] * streams[0]
            for j in range(1, n_streams):
                acc = acc + h_res_eff[:, :, i, j, None] * streams[j]
            out_streams.append(acc + h_post_eff[:, :, i, None] * out_dt)
        return jnp.stack(out_streams, axis=2)
