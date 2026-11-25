"""Muon + Auxiliary AdamW optimizer with per-expert 3D vmap orthogonalization and Warmup-Stable schedule (NO DECAY)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import optax
from optax.contrib import MuonDimensionNumbers, muon as optax_muon
from optax.contrib._muon import (
    _compute_muon_reshape,
    _is_weight_dim_nums,
    scale_by_shape,
)

# Cubic5 schedule (Hua Huang, "How Much Orthogonalization Does Muon Need?", l_0=7e-3):
# X_{t+1} = a_t X_t + b_t (X_t X_t^T) X_t -> 2 dominant matmuls/step (10 total vs 15 for 5-step quintic).
CUBIC5_NS_COEFFS: tuple[tuple[float, float], ...] = (
    (3.3656576, -3.3420992),
    (2.5744352, -1.4957376),
    (2.5368962, -1.4312570),
    (2.4418906, -1.2764040),
    (2.2230472, -0.9630650),
)


@dataclass(frozen=True)
class OptimConfig:
    """Hyperparameters for Muon + Auxiliary AdamW Warmup-Stable optimization."""

    muon_lr: float = 0.015
    adam_lr: float = 1.5e-3
    warmup_steps: int = 1000
    muon_momentum: float = 0.95
    muon_ns_mode: str = "cubic5"
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.01
    muon_consistent_rms: float = 0.2
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    eps: float = 1e-7


def make_warmup_stable_schedule(peak_lr: float, warmup_steps: int) -> optax.Schedule:
    """Create a Warmup-Stable schedule with NO learning-rate decay."""
    if warmup_steps <= 0:
        return optax.constant_schedule(peak_lr)
    init_lr = peak_lr / float(warmup_steps)
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=init_lr,
                end_value=peak_lr,
                transition_steps=max(1, warmup_steps - 1),
            ),
            optax.constant_schedule(peak_lr),
        ],
        boundaries=[max(1, warmup_steps - 1)],
    )


def _path_str(path: tuple[Any, ...]) -> str:
    parts = []
    for k in path:
        if hasattr(k, "key"):
            parts.append(str(k.key))
        elif hasattr(k, "name"):
            parts.append(str(k.name))
        else:
            parts.append(str(k))
    return "/".join(parts)


def classify_param(path_str: str, arr: Any) -> str:
    """Classify each parameter into frozen | muon | engram_row_adam | adam_decay | adam_no_decay."""
    if not hasattr(arr, "ndim"):
        return "frozen"
    if path_str.endswith("router_bias"):
        return "frozen"
    if path_str == "engram/tables":
        return "engram_row_adam"

    leaf = path_str.split("/")[-1]

    # Scalar gates, mHC biases, and norm weights (including 2D per-loop MLA norms) -> AdamW without decay
    if leaf in ("loop_alpha", "b_res", "b_pre", "b_post") or leaf.endswith("_norm_w") or leaf == "norm_w":
        return "adam_no_decay"

    # 3D (or split 4D) MoE expert tensors -> per-expert vmap Muon
    if path_str.startswith("moe/") and leaf in ("w_gate_up", "w_gate", "w_up", "w_down") and arr.ndim in (3, 4):
        return "muon"

    # MLA projection matrices (2D tied or 3D per-loop untied) -> Muon
    if path_str.startswith("mla/") and arr.ndim in (2, 3):
        return "muon"

    # 2D hidden linear matrices -> standard 2D Muon
    if arr.ndim == 2:
        if path_str.startswith("moe/") and leaf in ("shared_w_gate_up", "shared_w_gate", "shared_w_up", "shared_w_down"):
            return "muon"
        if path_str.startswith("engram/") and leaf in ("w_kv", "w_k", "w_v"):
            return "muon"
        if path_str.startswith("iter_adaln/") and leaf in ("w_h", "w_t", "w_out"):
            return "muon"

    # Scalar/1D tensors -> AdamW without weight decay
    if arr.ndim < 2:
        return "adam_no_decay"

    # Embeddings, Engram lookup tables, router weights (2D/3D), mHC dynamic matrices -> AdamW with decay
    return "adam_decay"


def make_param_labels(params: Any) -> Any:
    """Return pytree of optimizer group labels matching `params`."""
    return jax.tree_util.tree_map_with_path(
        lambda path, arr: classify_param(_path_str(path), arr),
        params,
    )


def _muon_dim_numbers_fn(params: Any) -> Any:
    """Specify per-expert (1, 3) for split 4D w_gate_up, (1, 2) for 3D MoE experts, and (0, 1) for 2D."""
    def _spec(path: tuple[Any, ...], arr: Any):
        if not hasattr(arr, "ndim"):
            return None
        p = _path_str(path)
        label = classify_param(p, arr)
        if label != "muon":
            return None
        if arr.ndim == 4:
            return MuonDimensionNumbers(reduction_axis=1, output_axis=3)
        if arr.ndim == 3:
            return MuonDimensionNumbers(reduction_axis=1, output_axis=2)
        return MuonDimensionNumbers(reduction_axis=0, output_axis=1)

    return jax.tree_util.tree_map_with_path(_spec, params)


def _split_gate_up_tree(tree: Any) -> Any:
    """Split moe/w_gate_up (E, D, 2*df) into two contiguous 3D leaves w_gate & w_up (E, D, df)."""
    if not isinstance(tree, dict) or "moe" not in tree or not isinstance(tree["moe"], dict):
        return tree
    moe = dict(tree["moe"])
    w_gu = moe.get("w_gate_up")
    if not hasattr(w_gu, "ndim") or w_gu.ndim != 3:
        return tree
    w_gate, w_up = jnp.split(w_gu, 2, axis=-1)
    moe.pop("w_gate_up")
    moe["w_gate"] = w_gate
    moe["w_up"] = w_up
    return {**tree, "moe": moe}


def _merge_gate_up_tree(tree: Any) -> Any:
    """Merge split 3D leaves moe/w_gate & moe/w_up (E, D, df) back into (E, D, 2*df)."""
    if not isinstance(tree, dict) or "moe" not in tree or not isinstance(tree["moe"], dict):
        return tree
    moe = dict(tree["moe"])
    if "w_gate" not in moe or "w_up" not in moe:
        return tree
    w_gate = moe.pop("w_gate")
    w_up = moe.pop("w_up")
    moe["w_gate_up"] = jnp.concatenate([w_gate, w_up], axis=-1)
    return {**tree, "moe": moe}


def _wrap_split_gate_up_muon(tx: optax.GradientTransformation) -> optax.GradientTransformation:
    """Wrap Muon to orthogonalize fused gate/up weights independently without touching forward layout."""

    def init_fn(params: Any) -> optax.OptState:
        return tx.init(_split_gate_up_tree(params))

    def update_fn(updates: Any, state: optax.OptState, params: Any | None = None):
        split_params = None if params is None else _split_gate_up_tree(params)
        new_updates, new_state = tx.update(_split_gate_up_tree(updates), state, split_params)
        return _merge_gate_up_tree(new_updates), new_state

    return optax.GradientTransformation(init_fn, update_fn)


class CubicMuonState(NamedTuple):
    """Optimizer state for Cubic5 Muon."""

    count: jax.Array
    mu: Any


def orthogonalize_via_cubic_ns(
    x: jax.Array,
    coeffs: Sequence[tuple[float, float]] = CUBIC5_NS_COEFFS,
    eps: float = 1e-7,
    dimension_numbers: MuonDimensionNumbers | None = None,
) -> jax.Array:
    """Unrolled cubic Newton-Schulz orthogonalization (2 matmuls/step, zero A@A or while-loop copies)."""
    if x.ndim == 2 and dimension_numbers is None:
        dimension_numbers = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
    if dimension_numbers is None:
        raise ValueError(f"dimension_numbers required for tensor of shape {x.shape}")

    def _orth(mat: jax.Array) -> jax.Array:
        dt = mat.dtype
        norm = jnp.sqrt(jnp.sum(jnp.square(mat), dtype=jnp.float32)) + jnp.asarray(eps, dtype=jnp.float32)
        inv_norm = (jnp.asarray(1.0, dtype=jnp.float32) / norm).astype(dt)
        mat = mat * inv_norm
        if mat.shape[0] <= mat.shape[1]:
            for a_val, b_val in coeffs:
                gram = mat @ mat.T
                mat = jnp.asarray(a_val, dtype=dt) * mat + jnp.asarray(b_val, dtype=dt) * (gram @ mat)
        else:
            for a_val, b_val in coeffs:
                gram = mat.T @ mat
                mat = jnp.asarray(a_val, dtype=dt) * mat + jnp.asarray(b_val, dtype=dt) * (mat @ gram)
        return mat

    reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
    return inverse_fn(jax.vmap(_orth)(reshape_fn(x)))


def scale_by_cubic_muon(
    coeffs: Sequence[tuple[float, float]] = CUBIC5_NS_COEFFS,
    beta: float = 0.95,
    eps: float = 1e-7,
    mu_dtype: jnp.dtype | None = jnp.bfloat16,
    nesterov: bool = True,
    weight_dimension_numbers: Callable[[Any], Any] | MuonDimensionNumbers | None = None,
) -> optax.GradientTransformation:
    """Momentum + Nesterov + Cubic5 Newton-Schulz orthogonalization."""
    mu_dtype_can = optax._src.utils.canonicalize_dtype(mu_dtype)

    def init_fn(params: Any) -> CubicMuonState:
        return CubicMuonState(
            count=jnp.zeros([], dtype=jnp.int32),
            mu=optax.tree.zeros_like(params, dtype=mu_dtype_can),
        )

    def update_fn(updates: Any, state: CubicMuonState, params: Any | None = None):
        del params
        resolved_dim_nums = (
            weight_dimension_numbers(updates)
            if callable(weight_dimension_numbers)
            else weight_dimension_numbers
        )
        def _bias_corr_native(tree: Any, step_count: jax.Array) -> Any:
            inv_bias_corr = jnp.asarray(1.0, dtype=jnp.float32) / (
                jnp.asarray(1.0, dtype=jnp.float32)
                - jnp.asarray(beta, dtype=jnp.float32) ** step_count.astype(jnp.float32)
            )
            return jax.tree_util.tree_map(lambda x: x * inv_bias_corr.astype(x.dtype), tree)

        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count_inc = optax.tree.safe_increment(state.count) if hasattr(optax.tree, "safe_increment") else (state.count + jnp.asarray(1, dtype=jnp.int32))
        if nesterov:
            mu_corr = _bias_corr_native(mu, count_inc + jnp.asarray(1, dtype=jnp.int32))
            g_corr = _bias_corr_native(updates, count_inc)
            mu_hat = jax.tree_util.tree_map(
                lambda m, g: jnp.asarray(beta, dtype=m.dtype) * m
                + jnp.asarray(1.0 - beta, dtype=g.dtype) * g,
                mu_corr,
                g_corr,
            )
        else:
            mu_hat = _bias_corr_native(mu, count_inc)

        is_leaf = lambda v: v is None or _is_weight_dim_nums(v)
        ortho_updates = jax.tree_util.tree_map(
            lambda arr, dn: arr if dn is None else orthogonalize_via_cubic_ns(
                arr, coeffs=coeffs, eps=eps, dimension_numbers=dn
            ),
            mu_hat,
            resolved_dim_nums,
            is_leaf=is_leaf,
        )
        mu_cast = optax.tree.cast(mu, mu_dtype_can)
        return ortho_updates, CubicMuonState(count=count_inc, mu=mu_cast)

    return optax.GradientTransformation(init_fn, update_fn)


class RowAdamState(NamedTuple):
    """Optimizer state for Row-Wise Factored AdamW (BF16 mu + 1 FP32 scalar v_row per N-gram slot)."""

    count: jax.Array
    mu: Any
    v_row: Any


def scale_by_row_adam(
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: jnp.dtype | None = jnp.bfloat16,
) -> optax.GradientTransformation:
    """Factored row-wise Adam preserving Zipfian per-slot step scaling at ~33% less HBM & bandwidth."""
    mu_dtype_can = optax._src.utils.canonicalize_dtype(mu_dtype)

    def init_fn(params: Any) -> RowAdamState:
        return RowAdamState(
            count=jnp.zeros([], dtype=jnp.int32),
            mu=optax.tree.zeros_like(params, dtype=mu_dtype_can),
            v_row=jax.tree_util.tree_map(
                lambda a: jnp.zeros(a.shape[:-1] + (1,), dtype=jnp.float32),
                params,
            ),
        )

    def update_fn(updates: Any, state: RowAdamState, params: Any | None = None):
        del params
        count_inc = (
            optax.tree.safe_increment(state.count)
            if hasattr(optax.tree, "safe_increment")
            else (state.count + jnp.asarray(1, dtype=jnp.int32))
        )
        c_f32 = count_inc.astype(jnp.float32)
        bc1 = jnp.asarray(1.0, dtype=jnp.float32) - jnp.asarray(b1, dtype=jnp.float32) ** c_f32
        bc2 = jnp.asarray(1.0, dtype=jnp.float32) - jnp.asarray(b2, dtype=jnp.float32) ** c_f32

        def _step_leaf(g: jax.Array, m: jax.Array, v: jax.Array):
            g_f32 = g.astype(jnp.float32)
            new_m_f32 = jnp.asarray(b1, dtype=jnp.float32) * m.astype(jnp.float32) + (
                jnp.asarray(1.0 - b1, dtype=jnp.float32) * g_f32
            )
            row_sq = jnp.mean(jnp.square(g_f32), axis=-1, keepdims=True)
            new_v = jnp.asarray(b2, dtype=jnp.float32) * v + (
                jnp.asarray(1.0 - b2, dtype=jnp.float32) * row_sq
            )
            upd = (new_m_f32 / bc1) / (jnp.sqrt(new_v / bc2) + jnp.asarray(eps, dtype=jnp.float32))
            return upd.astype(g.dtype), new_m_f32.astype(mu_dtype_can), new_v

        treedef = jax.tree_util.tree_structure(updates)
        u_leaves = jax.tree_util.tree_leaves(updates)
        m_leaves = jax.tree_util.tree_leaves(state.mu)
        v_leaves = jax.tree_util.tree_leaves(state.v_row)
        out_u, out_m, out_v = [], [], []
        for g, m, v in zip(u_leaves, m_leaves, v_leaves, strict=True):
            ug, nm, nv = _step_leaf(g, m, v)
            out_u.append(ug)
            out_m.append(nm)
            out_v.append(nv)
        return jax.tree_util.tree_unflatten(treedef, out_u), RowAdamState(
            count=count_inc,
            mu=jax.tree_util.tree_unflatten(treedef, out_m),
            v_row=jax.tree_util.tree_unflatten(treedef, out_v),
        )

    return optax.GradientTransformation(init_fn, update_fn)


def create_muon_adamw_optimizer(
    optim_cfg: OptimConfig,
    params_template: Any | None = None,
) -> optax.GradientTransformation:
    """Build clipped Muon (Cubic5) + Row-Factored Engram AdamW + BF16-moment AdamW optimizer."""
    del params_template  # labels & dim numbers resolve dynamically inside optax.partition
    muon_lr = make_warmup_stable_schedule(optim_cfg.muon_lr, optim_cfg.warmup_steps)
    adam_lr = make_warmup_stable_schedule(optim_cfg.adam_lr, optim_cfg.warmup_steps)

    if optim_cfg.muon_ns_mode == "cubic5":
        coeffs = CUBIC5_NS_COEFFS[: max(1, min(optim_cfg.muon_ns_steps, len(CUBIC5_NS_COEFFS)))]
        muon_tx = _wrap_split_gate_up_muon(
            optax.chain(
                scale_by_cubic_muon(
                    coeffs=coeffs,
                    beta=optim_cfg.muon_momentum,
                    eps=optim_cfg.eps,
                    mu_dtype=jnp.bfloat16,
                    nesterov=True,
                    weight_dimension_numbers=_muon_dim_numbers_fn,
                ),
                scale_by_shape(
                    weight_dimension_numbers=_muon_dim_numbers_fn,
                    consistent_rms=optim_cfg.muon_consistent_rms,
                ),
                optax.add_decayed_weights(optim_cfg.muon_weight_decay),
                optax.scale_by_learning_rate(muon_lr),
            )
        )
    elif optim_cfg.muon_ns_mode == "quintic":
        muon_tx = _wrap_split_gate_up_muon(
            optax_muon(
                learning_rate=muon_lr,
                ns_steps=optim_cfg.muon_ns_steps,
                beta=optim_cfg.muon_momentum,
                eps=optim_cfg.eps,
                weight_decay=optim_cfg.muon_weight_decay,
                mu_dtype=jnp.bfloat16,
                nesterov=True,
                muon_weight_dimension_numbers=_muon_dim_numbers_fn,
                consistent_rms=optim_cfg.muon_consistent_rms,
            )
        )
    else:
        raise ValueError(f"Unsupported muon_ns_mode: {optim_cfg.muon_ns_mode!r}")

    engram_row_adam_tx = optax.chain(
        scale_by_row_adam(
            b1=optim_cfg.adam_b1,
            b2=optim_cfg.adam_b2,
            eps=1e-8,
            mu_dtype=jnp.bfloat16,
        ),
        optax.add_decayed_weights(optim_cfg.adam_weight_decay),
        optax.scale_by_learning_rate(adam_lr),
    )

    adam_decay_tx = optax.adamw(
        learning_rate=adam_lr,
        b1=optim_cfg.adam_b1,
        b2=optim_cfg.adam_b2,
        eps=1e-8,
        weight_decay=optim_cfg.adam_weight_decay,
        mu_dtype=jnp.bfloat16,
    )

    adam_no_decay_tx = optax.adamw(
        learning_rate=adam_lr,
        b1=optim_cfg.adam_b1,
        b2=optim_cfg.adam_b2,
        eps=1e-8,
        weight_decay=0.0,
        mu_dtype=jnp.bfloat16,
    )

    partitioned = optax.partition(
        transforms={
            "muon": muon_tx,
            "engram_row_adam": engram_row_adam_tx,
            "adam_decay": adam_decay_tx,
            "adam_no_decay": adam_no_decay_tx,
            "frozen": optax.set_to_zero(),
        },
        param_labels=make_param_labels,
    )

    if optim_cfg.grad_clip_norm > 0.0:
        return optax.chain(
            optax.clip_by_global_norm(optim_cfg.grad_clip_norm),
            partitioned,
        )
    return partitioned


def per_expert_newton_schulz(
    tensor_3d: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-7,
    mode: str = "cubic5",
) -> jax.Array:
    """Explicit helper applying 2D Newton-Schulz orthogonalization independently per expert via vmap."""
    dim_nums = MuonDimensionNumbers(reduction_axis=1, output_axis=2)
    if mode == "cubic5":
        coeffs = CUBIC5_NS_COEFFS[: max(1, min(ns_steps, len(CUBIC5_NS_COEFFS)))]
        return orthogonalize_via_cubic_ns(
            tensor_3d,
            coeffs=coeffs,
            eps=eps,
            dimension_numbers=dim_nums,
        )
    coeffs_q = jnp.asarray((3.4445, -4.775, 2.0315), dtype=tensor_3d.dtype)
    from optax.contrib._muon import orthogonalize_via_newton_schulz

    return orthogonalize_via_newton_schulz(
        tensor_3d,
        ns_coeffs=coeffs_q,
        ns_steps=ns_steps,
        eps=eps,
        dimension_numbers=dim_nums,
    )
