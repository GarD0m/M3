"""DeepSeek-V4.1-Flash Fine-Grained MoE with Tokamax Megablox ragged_dot and Aux-Loss-Free balancing."""

from __future__ import annotations

import functools
import inspect
import math

import jax
import jax.numpy as jnp

from .config import LoopMoEConfig
from .mhc import rms_norm
from .pallas_moe import can_use_pallas_moe, pallas_moe_combine, pallas_moe_gather


def init_moe_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize routed experts (3D tensors for per-expert Muon), shared expert, and cyclic routers."""
    keys = jax.random.split(key, 5)
    d = cfg.d_model
    e = cfg.n_routed_experts
    df = cfg.d_ff_expert
    ds = cfg.n_shared_experts * cfg.d_ff_shared
    n_router_banks = min(cfg.max_loops, max(1, cfg.router_weight_banks))
    router_shape = (n_router_banks, d, e) if cfg.untie_router_per_loop else (d, e)

    params: dict[str, jax.Array] = {
        "router_w": jax.random.normal(keys[0], router_shape, dtype=dtype)
        * (1.0 / jnp.sqrt(float(d))),
        "router_bias": jnp.zeros((cfg.max_loops, e), dtype=jnp.float32),
        "w_gate_up": jax.random.normal(keys[1], (e, d, 2 * df), dtype=dtype)
        * (1.0 / jnp.sqrt(float(d))),
        "w_down": jax.random.normal(keys[2], (e, df, d), dtype=dtype) * (1.0 / jnp.sqrt(float(df))),
        "shared_w_gate_up": jax.random.normal(keys[3], (d, 2 * ds), dtype=dtype)
        * (1.0 / jnp.sqrt(float(d))),
        "shared_w_down": jax.random.normal(keys[4], (ds, d), dtype=dtype)
        * (1.0 / jnp.sqrt(float(ds))),
    }
    if cfg.use_post_moe_rmsnorm:
        params["post_norm_w"] = jnp.ones((d,), dtype=dtype)
    return params


def _grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    sorted_experts: jax.Array,
    use_tokamax: bool = True,
) -> jax.Array:
    """Compute grouped matrix multiply: lhs [M, K], rhs [E, K, N] -> [M, N].

    On TPU, prefers OpenXLA Tokamax Megablox (`ragged_dot` with `mosaic_tpu_v2`),
    then `jax.lax.ragged_dot`, and falls back cleanly to pure JAX gather-matmul on CPU.
    """
    on_tpu = jax.default_backend() == "tpu"

    if on_tpu and use_tokamax:
        try:
            from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2 as _tpu_rd  # type: ignore

            k_dim = int(rhs.shape[1])
            n_dim = int(rhs.shape[2])
            fwd_cfg = _tpu_rd.Config(
                tile_m=256,
                tile_k=min(k_dim, 2048),
                tile_n=min(n_dim, 1024),
            )
            dlhs_cfg = _tpu_rd.Config(
                tile_m=256,
                tile_k=min(n_dim, 2048),
                tile_n=min(k_dim, 2048),
            )
            drhs_cfg = _tpu_rd.Config(
                tile_m=256,
                tile_k=min(k_dim, 2048),
                tile_n=min(n_dim, 2048),
            )
            op = _tpu_rd.PallasMosaicTpuV2RaggedDot(
                config=fwd_cfg,
                dlhs_config=dlhs_cfg,
                drhs_config=drhs_cfg,
            )
            return op(lhs, rhs, group_sizes=group_sizes)
        except Exception:
            pass

        try:
            import tokamax  # type: ignore

            fn = getattr(tokamax, "ragged_dot", None)
            if fn is not None:
                sig = inspect.signature(fn)
                if "implementation" in sig.parameters:
                    try:
                        return fn(lhs, rhs, group_sizes=group_sizes, implementation="mosaic_tpu_v2")
                    except Exception:
                        return fn(lhs, rhs, group_sizes=group_sizes)
                return fn(lhs, rhs, group_sizes=group_sizes)
        except Exception:
            pass

    if on_tpu and hasattr(jax.lax, "ragged_dot"):
        try:
            return jax.lax.ragged_dot(lhs, rhs, group_sizes=group_sizes)
        except Exception:
            pass

    # Pure JAX portable fallback (also verified differentiable on CPU/GPU)
    gathered_w = rhs[sorted_experts]  # [M, K, N]
    return jnp.einsum("mk,mkn->mn", lhs, gathered_w)


def _gather_sum_topk(arr: jax.Array, inv_order: jax.Array, top_k: int) -> jax.Array:
    """Stream K TPU-aligned [N, D] gathers directly in native dtype (avoids FP32 casts & retile copies)."""
    num_tokens = inv_order.shape[0] // top_k
    inv_2d = inv_order.reshape(num_tokens, top_k)
    acc = arr[inv_2d[:, 0]]
    for k in range(1, top_k):
        acc = acc + arr[inv_2d[:, k]]
    return acc


def _dispatch_gather(x_flat: jax.Array, token_indices: jax.Array, top_k: int) -> jax.Array:
    """Gather token rows using pipelined Mosaic TPU Pallas DMA when aligned, else native JAX gather."""
    if can_use_pallas_moe(x_flat.shape[0], token_indices.shape[0], x_flat.shape[1], top_k):
        try:
            return pallas_moe_gather(x_flat, token_indices)
        except Exception:
            pass
    return x_flat[token_indices]


def _combine_gather_sum(arr: jax.Array, inv_order: jax.Array, top_k: int) -> jax.Array:
    """Gather+sum top_k rows per token using double-buffered Pallas VMEM combine on TPU when aligned."""
    n_rows = inv_order.shape[0] // top_k
    if can_use_pallas_moe(n_rows, arr.shape[0], arr.shape[1], top_k):
        try:
            return pallas_moe_combine(arr, inv_order, top_k)
        except Exception:
            pass
    return _gather_sum_topk(arr, inv_order, top_k)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _moe_dispatch(
    x_flat: jax.Array,
    token_indices: jax.Array,
    inv_order: jax.Array,
    top_k: int,
) -> jax.Array:
    """Gather token embeddings into expert-sorted order; backward uses conflict-free inv_order reduce."""
    del inv_order
    return _dispatch_gather(x_flat, token_indices, top_k)


def _moe_dispatch_fwd(
    x_flat: jax.Array,
    token_indices: jax.Array,
    inv_order: jax.Array,
    top_k: int,
) -> tuple[jax.Array, tuple[jax.Array]]:
    return _dispatch_gather(x_flat, token_indices, top_k), (inv_order,)


def _moe_dispatch_bwd(
    top_k: int,
    res: tuple[jax.Array],
    d_sorted_x: jax.Array,
) -> tuple[jax.Array, None, None]:
    (inv_order,) = res
    return (_combine_gather_sum(d_sorted_x, inv_order, top_k), None, None)


_moe_dispatch.defvjp(_moe_dispatch_fwd, _moe_dispatch_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _moe_combine(
    down_out: jax.Array,
    token_indices: jax.Array,
    inv_order: jax.Array,
    top_k: int,
) -> jax.Array:
    """Combine expert-sorted outputs per token via conflict-free inv_order gather+sum (zero scatter-add)."""
    del token_indices
    return _combine_gather_sum(down_out, inv_order, top_k)


def _moe_combine_fwd(
    down_out: jax.Array,
    token_indices: jax.Array,
    inv_order: jax.Array,
    top_k: int,
) -> tuple[jax.Array, tuple[jax.Array]]:
    return _combine_gather_sum(down_out, inv_order, top_k), (token_indices,)


def _moe_combine_bwd(
    top_k: int,
    res: tuple[jax.Array],
    d_routed_out: jax.Array,
) -> tuple[jax.Array, None, None]:
    (token_indices,) = res
    return (_dispatch_gather(d_routed_out, token_indices, top_k), None, None)


_moe_combine.defvjp(_moe_combine_fwd, _moe_combine_bwd)


@jax.custom_vjp
def _permute_1d(arr: jax.Array, order: jax.Array, inv_order: jax.Array) -> jax.Array:
    """Gather 1D tensor by permutation `order`; backward uses inverse permutation (zero scatter-add)."""
    del inv_order
    if arr.dtype != jnp.float32:
        return arr.astype(jnp.float32)[order].astype(arr.dtype)
    return arr[order]


def _permute_1d_fwd(
    arr: jax.Array, order: jax.Array, inv_order: jax.Array
) -> tuple[jax.Array, tuple[jax.Array]]:
    out = arr.astype(jnp.float32)[order].astype(arr.dtype) if arr.dtype != jnp.float32 else arr[order]
    return out, (inv_order,)


def _permute_1d_bwd(
    res: tuple[jax.Array], d_out: jax.Array
) -> tuple[jax.Array, None, None]:
    (inv_order,) = res
    if d_out.dtype != jnp.float32:
        return (d_out.astype(jnp.float32)[inv_order].astype(d_out.dtype), None, None)
    return (d_out[inv_order], None, None)


_permute_1d.defvjp(_permute_1d_fwd, _permute_1d_bwd)


@jax.custom_vjp
def _sorted_weights_from_sort(
    topk_weights_flat: jax.Array,
    sorted_weights_f32: jax.Array,
    sort_order: jax.Array,
) -> jax.Array:
    """Return pre-sorted routing weights in forward (zero 1D gather) and un-permute grads via bitonic sort."""
    del sort_order
    return sorted_weights_f32.astype(topk_weights_flat.dtype)


def _sorted_weights_from_sort_fwd(
    topk_weights_flat: jax.Array,
    sorted_weights_f32: jax.Array,
    sort_order: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array]]:
    return sorted_weights_f32.astype(topk_weights_flat.dtype), (sort_order,)


def _sorted_weights_from_sort_bwd(
    res: tuple[jax.Array], d_sorted: jax.Array
) -> tuple[jax.Array, None, None]:
    (sort_order,) = res
    _, d_unsorted_f32 = jax.lax.sort((sort_order, d_sorted.astype(jnp.float32)), num_keys=1)
    return (d_unsorted_f32.astype(d_sorted.dtype), None, None)


_sorted_weights_from_sort.defvjp(_sorted_weights_from_sort_fwd, _sorted_weights_from_sort_bwd)


def _extract_topk_scores(scores: jax.Array, topk_idx: jax.Array, n_experts: int) -> jax.Array:
    """Extract [N, K] routing scores from [N, E] via K VPU masked reductions (zero 1D gather/scatter)."""
    top_k = topk_idx.shape[-1]
    zero = jnp.asarray(0.0, dtype=scores.dtype)
    e_idx = jnp.arange(n_experts, dtype=topk_idx.dtype)[None, :]  # [1, E]
    cols = [
        jnp.sum(jnp.where(e_idx == topk_idx[:, k : k + 1], scores, zero), axis=-1)
        for k in range(top_k)
    ]
    return jnp.stack(cols, axis=-1)


def apply_moe(
    params: dict[str, jax.Array],
    x: jax.Array,
    loop_idx: jax.Array | int,
    cfg: LoopMoEConfig,
    keep_scores: jax.Array | None = None,
    capacity_factor: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run DeepSeek-V4.1-Flash fine-grained MoE with optional training early-exit token dropping.

    Returns:
        out: [B, S, D] MoE output tensor (zeros for early-exited tokens when `keep_scores` drops them).
        expert_counts: [E] token assignment counts (float32) for Aux-Loss-Free bias updates.
        seq_aux_loss: scalar sequence-wise balance regularization loss.
        router_mmr: scalar Foil Median Margin Ratio diagnostic.
    """
    bsz, seq_len, d_model = x.shape
    num_tokens = bsz * seq_len
    x_flat = x.reshape(num_tokens, d_model)
    dt = x_flat.dtype

    active_idx: jax.Array | None = None
    active_bsz, active_seq = bsz, seq_len
    if keep_scores is not None and 0.0 < capacity_factor < 1.0 and num_tokens > 1:
        cap = max(1, int(math.ceil(num_tokens * float(capacity_factor))))
        if num_tokens >= 128:
            cap = min(num_tokens, ((cap + 127) // 128) * 128)
        if cap < num_tokens:
            _, active_idx = jax.lax.top_k(keep_scores.reshape(num_tokens), cap)
            x_flat = x_flat[active_idx]
            num_tokens = cap
            active_bsz, active_seq = 1, cap

    # 1. Sigmoid routing + Aux-Loss-Free bias (bias excluded from gradient flow)
    with jax.named_scope("moe_router"):
        router_w_t = (
            params["router_w"][loop_idx % params["router_w"].shape[0]]
            if params["router_w"].ndim == 3
            else params["router_w"]
        )
        router_logits = x_flat @ router_w_t.astype(dt)
        # Foil Eq. 11 Median Margin Ratio diagnostic: exp(mean(z_(1) - z_(E/2)))
        z_sub = jax.lax.stop_gradient(
            (router_logits[::8] if num_tokens > 4096 else router_logits).astype(jnp.float32)
        )
        z_sorted = jnp.sort(z_sub, axis=-1)
        mid_idx = cfg.n_routed_experts - (cfg.n_routed_experts // 2)
        router_mmr = jnp.exp(jnp.clip(jnp.mean(z_sorted[:, -1] - z_sorted[:, mid_idx]), -20.0, 20.0))

        scores = jax.nn.sigmoid(router_logits)
        bias_f32 = jax.lax.stop_gradient(params["router_bias"][loop_idx])
        biased_scores = jax.lax.stop_gradient(scores) + bias_f32.astype(dt)[None, :]

        _, topk_idx = jax.lax.top_k(biased_scores, cfg.top_k)  # [N, K]
        topk_scores = _extract_topk_scores(scores, topk_idx, cfg.n_routed_experts)
        topk_weights = topk_scores / (
            jnp.sum(topk_scores, axis=-1, keepdims=True) + jnp.asarray(1e-6, dtype=dt)
        )

    # 2. Sort token-expert pairs + routing weights once; reuse sorted_experts for counts and inv_order for combine/bwd
    with jax.named_scope("moe_sort_dispatch"):
        flat_experts = topk_idx.reshape(-1)
        arange_idx = jnp.arange(flat_experts.shape[0], dtype=jnp.int32)
        topk_weights_flat = topk_weights.reshape(-1)
        sorted_experts, sort_order, sorted_weights_f32 = jax.lax.sort(
            (flat_experts, arange_idx, jax.lax.stop_gradient(topk_weights_flat).astype(jnp.float32)),
            num_keys=1,
        )
        token_indices = sort_order // cfg.top_k
        _, inv_order = jax.lax.sort((sort_order, arange_idx), num_keys=1)
        edges = jnp.arange(cfg.n_routed_experts + 1, dtype=sorted_experts.dtype)
        expert_counts = jnp.diff(jnp.searchsorted(sorted_experts, edges, side="left")).astype(
            jnp.float32
        )

        sorted_weights = _sorted_weights_from_sort(topk_weights_flat, sorted_weights_f32, sort_order)
        sorted_x = _moe_dispatch(x_flat, token_indices, inv_order, cfg.top_k)  # [N*K, D]
        group_sizes = expert_counts.astype(jnp.int32)

    if cfg.seq_aux_loss_weight > 0.0:
        scores_norm = scores / (
            jnp.sum(scores, axis=-1, keepdims=True) + jnp.asarray(1e-6, dtype=dt)
        )
        scores_seq = scores_norm.reshape(active_bsz, active_seq, cfg.n_routed_experts)
        idx_seq = topk_idx.reshape(active_bsz, active_seq, cfg.top_k)
        seq_counts = jax.vmap(
            lambda idx: jnp.diff(jnp.searchsorted(jnp.sort(idx.reshape(-1)), edges, side="left"))
        )(idx_seq).astype(jnp.float32)
        ce = seq_counts * (float(cfg.n_routed_experts) / float(active_seq * cfg.top_k))  # [B, E]
        pi = jnp.mean(scores_seq, axis=1, dtype=jnp.float32)  # [B, E]
        seq_aux_loss = cfg.seq_aux_loss_weight * jnp.mean(jnp.sum(ce * pi, axis=-1))
    else:
        seq_aux_loss = jnp.asarray(0.0, dtype=jnp.float32)

    with jax.named_scope("moe_gmm_gate_up"):
        gate_up = _grouped_matmul(
            sorted_x, params["w_gate_up"], group_sizes, sorted_experts, use_tokamax=cfg.moe_use_tokamax
        )
        gate_act, up_act = jnp.split(gate_up, 2, axis=-1)
    scale_w = sorted_weights[:, None] * jnp.asarray(cfg.routed_scaling_factor, dtype=dt)
    hidden = (jax.nn.silu(gate_act) * up_act) * scale_w
    with jax.named_scope("moe_gmm_down"):
        down_out = _grouped_matmul(
            hidden, params["w_down"], group_sizes, sorted_experts, use_tokamax=cfg.moe_use_tokamax
        )

    with jax.named_scope("moe_scatter_combine"):
        routed_out = _moe_combine(down_out, token_indices, inv_order, cfg.top_k)

    # 4. Shared expert path (native BF16)
    with jax.named_scope("moe_shared_expert"):
        shared_gate_up = x_flat @ params["shared_w_gate_up"]
        shared_gate, shared_up = jnp.split(shared_gate_up, 2, axis=-1)
        shared_out = (jax.nn.silu(shared_gate) * shared_up) @ params["shared_w_down"]

    total_flat = routed_out + shared_out
    if cfg.use_post_moe_rmsnorm:
        total_flat = rms_norm(total_flat, params["post_norm_w"], eps=cfg.rms_norm_eps)
    if active_idx is not None:
        scattered = jnp.zeros((bsz * seq_len, d_model), dtype=total_flat.dtype).at[active_idx].set(
            total_flat
        )
        return scattered.reshape(bsz, seq_len, d_model), expert_counts, seq_aux_loss, router_mmr
    return total_flat.reshape(bsz, seq_len, d_model), expert_counts, seq_aux_loss, router_mmr


def update_router_biases(
    router_bias: jax.Array,
    mean_expert_counts: jax.Array,
    num_loops: int,
    update_rate: float = 1e-3,
    bias_clip: float = 16.0,
) -> jax.Array:
    """Apply DeepSeek Auxiliary-Loss-Free dynamic bias update for active loop indices."""
    active_counts = mean_expert_counts[:num_loops]  # [R, E]
    avg_load = jnp.mean(active_counts, axis=-1, keepdims=True)
    violation = jnp.sign(avg_load - active_counts)
    updated_slice = jnp.clip(
        router_bias[:num_loops] + float(update_rate) * violation,
        -bias_clip,
        bias_clip,
    )
    return router_bias.at[:num_loops].set(updated_slice)
