"""Muon + Auxiliary AdamW optimizer with per-expert 3D vmap orthogonalization and Warmup-Stable schedule (NO DECAY)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from optax.contrib import MuonDimensionNumbers, muon as optax_muon


@dataclass(frozen=True)
class OptimConfig:
    """Hyperparameters for Muon + Auxiliary AdamW Warmup-Stable optimization."""

    muon_lr: float = 0.015
    adam_lr: float = 1.5e-3
    warmup_steps: int = 1000
    muon_momentum: float = 0.95
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
    """Classify each parameter into frozen | muon | adam_decay | adam_no_decay."""
    if not hasattr(arr, "ndim"):
        return "frozen"
    if path_str.endswith("router_bias"):
        return "frozen"

    leaf = path_str.split("/")[-1]

    # 3D MoE expert tensors (E, in, out) -> per-expert vmap Muon
    if path_str.startswith("moe/") and leaf in ("w_gate_up", "w_gate", "w_up", "w_down") and arr.ndim == 3:
        return "muon"

    # 2D hidden linear matrices -> standard 2D Muon
    if arr.ndim == 2:
        if path_str.startswith("mla/"):
            return "muon"
        if path_str.startswith("moe/") and leaf in ("shared_w_gate_up", "shared_w_gate", "shared_w_up", "shared_w_down"):
            return "muon"
        if path_str.startswith("engram/") and leaf in ("w_k", "w_v"):
            return "muon"
        if path_str.startswith("iter_adaln/") and leaf in ("w_h", "w_t", "w_out"):
            return "muon"

    # Scalar gates, mHC biases, and 1D norms/biases -> AdamW without weight decay
    if leaf in ("loop_alpha", "b_res", "b_pre", "b_post") or arr.ndim < 2:
        return "adam_no_decay"

    # Embeddings, Engram lookup tables, router weights, mHC dynamic matrices -> AdamW with decay
    return "adam_decay"


def make_param_labels(params: Any) -> Any:
    """Return pytree of optimizer group labels matching `params`."""
    return jax.tree_util.tree_map_with_path(
        lambda path, arr: classify_param(_path_str(path), arr),
        params,
    )


def _muon_dim_numbers_fn(params: Any) -> Any:
    """Specify per-expert (1, 2) axes for 3D MoE experts and (0, 1) for 2D matrices."""
    def _spec(path: tuple[Any, ...], arr: Any):
        if not hasattr(arr, "ndim"):
            return None
        p = _path_str(path)
        label = classify_param(p, arr)
        if label != "muon":
            return None
        if arr.ndim == 3:
            return MuonDimensionNumbers(reduction_axis=1, output_axis=2)
        return MuonDimensionNumbers(reduction_axis=0, output_axis=1)

    return jax.tree_util.tree_map_with_path(_spec, params)


def create_muon_adamw_optimizer(
    optim_cfg: OptimConfig,
    params_template: Any | None = None,
) -> optax.GradientTransformation:
    """Build clipped Muon + BF16-moment AdamW optimizer with Warmup-Stable (no-decay) LR."""
    del params_template  # labels & dim numbers resolve dynamically inside optax.partition
    muon_lr = make_warmup_stable_schedule(optim_cfg.muon_lr, optim_cfg.warmup_steps)
    adam_lr = make_warmup_stable_schedule(optim_cfg.adam_lr, optim_cfg.warmup_steps)

    muon_tx = optax_muon(
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
) -> jax.Array:
    """Explicit helper applying 2D Newton-Schulz orthogonalization independently per expert via vmap."""
    dim_nums = MuonDimensionNumbers(reduction_axis=1, output_axis=2)
    coeffs = jnp.asarray((3.4445, -4.775, 2.0315), dtype=tensor_3d.dtype)
    from optax.contrib._muon import orthogonalize_via_newton_schulz

    return orthogonalize_via_newton_schulz(
        tensor_3d,
        ns_coeffs=coeffs,
        ns_steps=ns_steps,
        eps=eps,
        dimension_numbers=dim_nums,
    )
