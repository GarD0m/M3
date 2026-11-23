"""DeepSeek-V4.1-Flash Fine-Grained MoE with Tokamax Megablox ragged_dot and Aux-Loss-Free balancing."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp

from .config import LoopMoEConfig


def init_moe_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize routed experts (3D tensors for per-expert Muon), shared expert, and router."""
    keys = jax.random.split(key, 5)
    d = cfg.d_model
    e = cfg.n_routed_experts
    df = cfg.d_ff_expert
    ds = cfg.n_shared_experts * cfg.d_ff_shared

    return {
        "router_w": jax.random.normal(keys[0], (d, e), dtype=dtype) * (1.0 / jnp.sqrt(float(d))),
        "router_bias": jnp.zeros((cfg.max_loops, e), dtype=jnp.float32),
        "w_gate_up": jax.random.normal(keys[1], (e, d, 2 * df), dtype=dtype)
        * (1.0 / jnp.sqrt(float(d))),
        "w_down": jax.random.normal(keys[2], (e, df, d), dtype=dtype) * (1.0 / jnp.sqrt(float(df))),
        "shared_w_gate_up": jax.random.normal(keys[3], (d, 2 * ds), dtype=dtype)
        * (1.0 / jnp.sqrt(float(d))),
        "shared_w_down": jax.random.normal(keys[4], (ds, d), dtype=dtype)
        * (1.0 / jnp.sqrt(float(ds))),
    }


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
            op = _tpu_rd.PallasMosaicTpuV2RaggedDot(
                config=fwd_cfg,
                dlhs_config=_tpu_rd.Config(),
                drhs_config=_tpu_rd.Config(),
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


def apply_moe(
    params: dict[str, jax.Array],
    x: jax.Array,
    loop_idx: jax.Array | int,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run DeepSeek-V4.1-Flash fine-grained MoE for one loop iteration.

    Returns:
        out: [B, S, D] MoE output tensor.
        expert_counts: [E] token assignment counts (float32) for Aux-Loss-Free bias updates.
        seq_aux_loss: scalar sequence-wise balance regularization loss.
    """
    bsz, seq_len, d_model = x.shape
    num_tokens = bsz * seq_len
    x_flat = x.reshape(num_tokens, d_model)

    # 1. Sigmoid routing + Aux-Loss-Free bias (bias excluded from gradient flow)
    with jax.named_scope("moe_router"):
        router_logits = (x_flat @ params["router_w"]).astype(jnp.float32)
        scores = jax.nn.sigmoid(router_logits)
        bias_t = jax.lax.stop_gradient(params["router_bias"][loop_idx])
        biased_scores = scores + bias_t[None, :]

        _, topk_idx = jax.lax.top_k(biased_scores, cfg.top_k)  # [N, K]
        topk_scores = jnp.take_along_axis(scores, topk_idx, axis=-1)
        topk_weights = topk_scores / (jnp.sum(topk_scores, axis=-1, keepdims=True) + 1e-20)

        # 2. Sequence-wise balance loss (optional tiny regularizer) & token load counts
        expert_counts = jnp.bincount(topk_idx.reshape(-1), length=cfg.n_routed_experts).astype(
            jnp.float32
        )

        if cfg.seq_aux_loss_weight > 0.0:
            scores_norm = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores_seq = scores_norm.reshape(bsz, seq_len, cfg.n_routed_experts)
            idx_seq = topk_idx.reshape(bsz, seq_len, cfg.top_k)
            # vmap bincount avoids materializing a huge [B, S, K, E] FP32 one_hot tensor
            seq_counts = jax.vmap(
                lambda idx: jnp.bincount(idx.reshape(-1), length=cfg.n_routed_experts)
            )(idx_seq).astype(jnp.float32)
            ce = seq_counts * (float(cfg.n_routed_experts) / float(seq_len * cfg.top_k))  # [B, E]
            pi = jnp.mean(scores_seq, axis=1)  # [B, E]
            seq_aux_loss = cfg.seq_aux_loss_weight * jnp.mean(jnp.sum(ce * pi, axis=-1))
        else:
            seq_aux_loss = jnp.asarray(0.0, dtype=jnp.float32)

    # 3. Sort token-expert pairs by expert index for dropless ragged_dot
    with jax.named_scope("moe_sort_dispatch"):
        flat_experts = topk_idx.reshape(-1)
        sort_order = jnp.argsort(flat_experts)
        sorted_experts = flat_experts[sort_order]
        token_indices = sort_order // cfg.top_k
        sorted_weights = topk_weights.reshape(-1)[sort_order]

        sorted_x = x_flat[token_indices]  # [N*K, D]
        group_sizes = expert_counts.astype(jnp.int32)

    with jax.named_scope("moe_gmm_gate_up"):
        gate_up = _grouped_matmul(
            sorted_x, params["w_gate_up"], group_sizes, sorted_experts, use_tokamax=cfg.moe_use_tokamax
        )
        gate_act, up_act = jnp.split(gate_up, 2, axis=-1)
    dt = x_flat.dtype
    scale_w = (sorted_weights[:, None] * float(cfg.routed_scaling_factor)).astype(dt)
    hidden = (jax.nn.silu(gate_act) * up_act) * scale_w
    with jax.named_scope("moe_gmm_down"):
        down_out = _grouped_matmul(
            hidden, params["w_down"], group_sizes, sorted_experts, use_tokamax=cfg.moe_use_tokamax
        )

    with jax.named_scope("moe_scatter_combine"):
        routed_out = jnp.zeros((num_tokens, d_model), dtype=dt).at[token_indices].add(down_out)

    # 4. Shared expert path (native BF16)
    with jax.named_scope("moe_shared_expert"):
        shared_gate_up = x_flat @ params["shared_w_gate_up"]
        shared_gate, shared_up = jnp.split(shared_gate_up, 2, axis=-1)
        shared_out = (jax.nn.silu(shared_gate) * shared_up) @ params["shared_w_down"]

    total_out = (routed_out + shared_out).reshape(bsz, seq_len, d_model)
    return total_out, expert_counts, seq_aux_loss


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
