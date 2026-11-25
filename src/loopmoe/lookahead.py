"""Explicit file-driven config validation and Lookahead Time-Efficiency curriculum scheduler."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .config import LoopMoEConfig, StageConfig
from .optim import OptimConfig

REQUIRED_MODEL_KEYS: tuple[str, ...] = (
    "vocab_size",
    "d_model",
    "num_layers",
    "max_loops",
    "tie_word_embeddings",
    "dtype_name",
    "rms_norm_eps",
    "chunk_ce_size",
    "rope_theta",
    "base_seq_len",
    "max_seq_len",
    "yarn_beta_fast",
    "yarn_beta_slow",
    "yarn_mscale",
    "use_mhc",
    "mhc_streams",
    "mhc_sinkhorn_iters",
    "mhc_alpha_init",
    "use_engram",
    "engram_orders",
    "engram_hash_heads",
    "engram_head_dim",
    "engram_total_slots",
    "engram_slot_alignment",
    "engram_conv_kernel",
    "engram_conv_dilation",
    "engram_seed",
    "engram_pre_loop_only",
    "n_heads",
    "d_q_lora",
    "d_kv_lora",
    "d_qk_nope",
    "d_qk_rope",
    "d_v",
    "untie_mla_per_loop",
    "mla_weight_banks",
    "reuse_kv_across_loops",
    "attn_use_tokamax",
    "use_csa2_indexer",
    "csa2_num_heads",
    "csa2_head_dim",
    "csa2_topk",
    "n_routed_experts",
    "d_ff_expert",
    "n_shared_experts",
    "d_ff_shared",
    "top_k",
    "routed_scaling_factor",
    "router_bias_update_rate",
    "seq_aux_loss_weight",
    "untie_router_per_loop",
    "router_weight_banks",
    "use_post_moe_rmsnorm",
    "moe_use_tokamax",
    "moe_chunk_size",
    "iter_adaln_bottleneck",
    "embed_reinject_lambda",
    "use_dynamic_loop",
    "halt_balance_weight",
    "halt_threshold",
    "dynamic_loop_ce_subsample",
    "dynamic_loop_ce_sample_depths",
    "train_loop_capacity_factor",
    "train_loop_explore_noise",
    "bptt_window",
    "remat_per_loop",
    "remat_sublayers",
    "remat_moe",
)

REQUIRED_OPTIMIZER_KEYS: tuple[str, ...] = (
    "muon_lr",
    "adam_lr",
    "warmup_steps",
    "muon_momentum",
    "muon_ns_steps",
    "muon_ns_mode",
    "muon_consistent_rms",
    "muon_weight_decay",
    "adam_b1",
    "adam_b2",
    "adam_weight_decay",
    "grad_clip_norm",
    "eps",
)

REQUIRED_TRAINING_KEYS: tuple[str, ...] = (
    "peak_tflops",
    "seed",
    "log_every",
    "keep_last_k",
    "tokenizer",
    "data_mode",
)

REQUIRED_LOOKAHEAD_KEYS: tuple[str, ...] = (
    "enabled",
    "main_steps",
    "probe_steps",
    "edge_window",
    "metric",
    "switch_margin",
)

REQUIRED_STAGE_KEYS: tuple[str, ...] = (
    "stage_id",
    "name",
    "seq_len",
    "num_loops",
    "microbatch",
    "grad_accum",
    "warmup_steps",
    "remat_moe",
    "remat_sublayers",
    "chunk_ce_size",
    "target_tokens",
    "checkpoint_every_steps",
    "web_en_ratio",
    "web_zh_ratio",
    "code_ratio",
    "math_ratio",
    "web_tier",
    "code_tier",
    "math_tier",
)


def _require_keys(section_name: str, data: Any, required: Sequence[str]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section_name}' must be a JSON object.")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"Config section '{section_name}' is missing required keys (no silent defaults allowed): {missing}"
        )
    return data


@dataclass(frozen=True)
class LookaheadConfig:
    """Configuration for training-metric time-efficiency Lookahead stage transitions."""

    enabled: bool
    main_steps: int
    probe_steps: int
    edge_window: int
    metric: str
    switch_margin: float


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    """Explicit runtime configuration loaded from file (hardware peak, seed, logging, data)."""

    peak_tflops: float
    seed: int
    log_every: int
    keep_last_k: int
    tokenizer: str
    data_mode: str


@dataclass(frozen=True)
class ExplicitStageConfig:
    """Explicit per-stage execution, HBM, curriculum, and data-mix configuration."""

    stage_id: int
    name: str
    seq_len: int
    num_loops: int
    microbatch: int
    grad_accum: int
    warmup_steps: int
    remat_moe: bool
    remat_sublayers: bool
    chunk_ce_size: int
    target_tokens: int
    checkpoint_every_steps: int
    web_en_ratio: float
    web_zh_ratio: float
    code_ratio: float
    math_ratio: float
    web_tier: str
    code_tier: str
    math_tier: str
    moe_chunk_size: int = 0
    use_csa2_indexer: bool | None = None

    @property
    def microbatch_tokens(self) -> int:
        return self.microbatch * self.seq_len

    @property
    def global_batch_tokens(self) -> int:
        return self.microbatch * self.grad_accum * self.seq_len

    def to_stage_config(self, base_csa2: bool) -> StageConfig:
        return StageConfig(
            stage_id=self.stage_id,
            name=self.name,
            seq_len=self.seq_len,
            target_tokens=self.target_tokens,
            num_loops=self.num_loops,
            global_batch_tokens=self.global_batch_tokens,
            microbatch=self.microbatch,
            warmup_steps=self.warmup_steps,
            web_en_ratio=self.web_en_ratio,
            web_zh_ratio=self.web_zh_ratio,
            code_ratio=self.code_ratio,
            math_ratio=self.math_ratio,
            web_tier=self.web_tier,
            code_tier=self.code_tier,
            math_tier=self.math_tier,
            use_csa2_indexer=(
                base_csa2 if self.use_csa2_indexer is None else bool(self.use_csa2_indexer)
            ),
            checkpoint_every_steps=self.checkpoint_every_steps,
        )

    def build_model_config(self, base_model_cfg: LoopMoEConfig) -> LoopMoEConfig:
        overrides: dict[str, Any] = {
            "num_loops": int(self.num_loops),
            "yarn_scale": base_model_cfg.yarn_scale_for_seq_len(self.seq_len),
            "remat_moe": bool(self.remat_moe),
            "remat_sublayers": bool(self.remat_sublayers),
            "chunk_ce_size": int(self.chunk_ce_size),
            "moe_chunk_size": int(self.moe_chunk_size),
        }
        if self.use_csa2_indexer is not None:
            overrides["use_csa2_indexer"] = bool(self.use_csa2_indexer)
        eff_csa2 = bool(overrides.get("use_csa2_indexer", base_model_cfg.use_csa2_indexer))
        if eff_csa2 and self.seq_len < 16384:
            overrides["csa2_topk"] = self.seq_len
        return LoopMoEConfig(**{**base_model_cfg.to_dict(), **overrides})


@dataclass(frozen=True)
class TrainFileConfig:
    """Complete explicitly-declared training configuration loaded from JSON."""

    model: LoopMoEConfig
    optimizer: OptimConfig
    training: TrainingRuntimeConfig
    lookahead: LookaheadConfig
    stages: tuple[ExplicitStageConfig, ...]


@dataclass
class LookaheadState:
    """Mutable state tracking previous stage-B probe tail loss across blocks."""

    prev_probe_tail_loss: float | None = None
    probe_count: int = 0


@dataclass(frozen=True)
class LookaheadDecision:
    """Result of comparing Stage A vs Stage B time efficiency strictly via Stage B loss."""

    should_switch: bool
    reason: str
    probe_idx: int
    head_loss: float
    tail_loss: float
    prev_tail_loss: float | None
    eff_a: float | None
    eff_b: float | None


class LookaheadTracker:
    """Compare training time efficiency on next-stage loss without standalone evaluation.

    Workflow across steady-state blocks after stage warmup completes:
      1. Run a large block on stage A (`main_steps`, duration `t_a_sec`) -> probe stage B (`probe_steps`, duration `t_b_sec`).
         The first probe only establishes `prev_probe_tail_loss` (`should_switch=False`).
      2. After the next large block on stage A -> second probe on stage B:
         - `eff_A = (prev_probe_tail_loss - head_loss_B) / t_a_sec`
         - `eff_B = (head_loss_B - tail_loss_B) / t_b_eff`
         Switch to stage B iff `eff_B > eff_A + switch_margin`.
    """

    def __init__(self, cfg: LookaheadConfig, prev_probe_tail_loss: float | None = None):
        self.cfg = cfg
        self.state = LookaheadState(prev_probe_tail_loss=prev_probe_tail_loss, probe_count=0)

    def should_probe(self, stage_step: int, warmup_steps: int, has_next_stage: bool) -> bool:
        if not self.cfg.enabled or not has_next_stage:
            return False
        return int(stage_step) >= int(warmup_steps)

    def reset(self) -> None:
        self.state.prev_probe_tail_loss = None
        self.state.probe_count = 0

    def record_probe(
        self,
        t_a_sec: float,
        t_b_sec: float,
        b_losses: Sequence[float],
    ) -> LookaheadDecision:
        if not b_losses:
            raise ValueError("b_losses must be non-empty when recording a lookahead probe.")
        n = len(b_losses)
        w = min(self.cfg.edge_window, max(1, n // 2)) if n > 1 else 1
        head_loss = float(sum(float(v) for v in b_losses[:w]) / float(w))
        tail_loss = float(sum(float(v) for v in b_losses[-w:]) / float(w))
        t_b_eff = (
            float(t_b_sec) * (float(max(1, n - w)) / float(n)) if n > 1 else float(t_b_sec)
        )

        self.state.probe_count += 1
        probe_idx = self.state.probe_count
        prev_tail = self.state.prev_probe_tail_loss

        if prev_tail is None:
            self.state.prev_probe_tail_loss = tail_loss
            return LookaheadDecision(
                should_switch=False,
                reason="first_probe_baseline",
                probe_idx=probe_idx,
                head_loss=head_loss,
                tail_loss=tail_loss,
                prev_tail_loss=None,
                eff_a=None,
                eff_b=None,
            )

        eff_a = (prev_tail - head_loss) / max(float(t_a_sec), 1e-6)
        eff_b = (head_loss - tail_loss) / max(t_b_eff, 1e-6)
        should_switch = eff_b > (eff_a + float(self.cfg.switch_margin))
        if not should_switch:
            self.state.prev_probe_tail_loss = tail_loss
            reason = "current_stage_more_efficient"
        else:
            reason = "next_stage_more_efficient"

        return LookaheadDecision(
            should_switch=should_switch,
            reason=reason,
            probe_idx=probe_idx,
            head_loss=head_loss,
            tail_loss=tail_loss,
            prev_tail_loss=prev_tail,
            eff_a=eff_a,
            eff_b=eff_b,
        )


def load_train_config(path: str | Path) -> TrainFileConfig:
    """Load and strictly validate a JSON training config file with zero silent defaults."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config file not found: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Top-level training config must be a JSON object.")

    for sec in ("model", "optimizer", "training", "lookahead", "stages"):
        if sec not in raw:
            raise ValueError(f"Training config is missing required top-level section '{sec}'.")

    m_raw = dict(_require_keys("model", raw["model"], REQUIRED_MODEL_KEYS))
    if isinstance(m_raw["engram_orders"], list):
        m_raw["engram_orders"] = tuple(int(x) for x in m_raw["engram_orders"])
    # Ignore optional comments if any, construct LoopMoEConfig strictly with known fields
    valid_model_fields = set(LoopMoEConfig.__dataclass_fields__.keys())
    model_cfg = LoopMoEConfig(**{k: v for k, v in m_raw.items() if k in valid_model_fields})

    o_raw = _require_keys("optimizer", raw["optimizer"], REQUIRED_OPTIMIZER_KEYS)
    optim_cfg = OptimConfig(
        muon_lr=float(o_raw["muon_lr"]),
        adam_lr=float(o_raw["adam_lr"]),
        warmup_steps=int(o_raw["warmup_steps"]),
        muon_momentum=float(o_raw["muon_momentum"]),
        muon_ns_steps=int(o_raw["muon_ns_steps"]),
        muon_ns_mode=str(o_raw["muon_ns_mode"]),
        muon_consistent_rms=float(o_raw["muon_consistent_rms"]),
        muon_weight_decay=float(o_raw["muon_weight_decay"]),
        adam_b1=float(o_raw["adam_b1"]),
        adam_b2=float(o_raw["adam_b2"]),
        adam_weight_decay=float(o_raw["adam_weight_decay"]),
        grad_clip_norm=float(o_raw["grad_clip_norm"]),
        eps=float(o_raw["eps"]),
    )

    t_raw = _require_keys("training", raw["training"], REQUIRED_TRAINING_KEYS)
    if str(t_raw["data_mode"]) not in ("shards", "stream", "synthetic"):
        raise ValueError(f"Invalid training.data_mode: {t_raw['data_mode']!r}")
    training_cfg = TrainingRuntimeConfig(
        peak_tflops=float(t_raw["peak_tflops"]),
        seed=int(t_raw["seed"]),
        log_every=int(t_raw["log_every"]),
        keep_last_k=int(t_raw["keep_last_k"]),
        tokenizer=str(t_raw["tokenizer"]),
        data_mode=str(t_raw["data_mode"]),
    )

    l_raw = _require_keys("lookahead", raw["lookahead"], REQUIRED_LOOKAHEAD_KEYS)
    lookahead_cfg = LookaheadConfig(
        enabled=bool(l_raw["enabled"]),
        main_steps=int(l_raw["main_steps"]),
        probe_steps=int(l_raw["probe_steps"]),
        edge_window=int(l_raw["edge_window"]),
        metric=str(l_raw["metric"]),
        switch_margin=float(l_raw["switch_margin"]),
    )
    if lookahead_cfg.main_steps < 1 or lookahead_cfg.probe_steps < 1 or lookahead_cfg.edge_window < 1:
        raise ValueError("lookahead main_steps, probe_steps, and edge_window must all be >= 1.")
    if lookahead_cfg.metric not in ("ce_loss", "loss"):
        raise ValueError(f"Unsupported lookahead.metric {lookahead_cfg.metric!r}; use 'ce_loss' or 'loss'.")

    stages_raw = raw["stages"]
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("Config section 'stages' must be a non-empty list.")

    stages: list[ExplicitStageConfig] = []
    seen_ids: set[int] = set()
    for idx, st in enumerate(stages_raw):
        s_raw = _require_keys(f"stages[{idx}]", st, REQUIRED_STAGE_KEYS)
        stage_id = int(s_raw["stage_id"])
        if stage_id in seen_ids:
            raise ValueError(f"Duplicate stage_id={stage_id} in stages[].")
        seen_ids.add(stage_id)
        st_cfg = ExplicitStageConfig(
            stage_id=stage_id,
            name=str(s_raw["name"]),
            seq_len=int(s_raw["seq_len"]),
            num_loops=int(s_raw["num_loops"]),
            microbatch=int(s_raw["microbatch"]),
            grad_accum=int(s_raw["grad_accum"]),
            warmup_steps=int(s_raw["warmup_steps"]),
            remat_moe=bool(s_raw["remat_moe"]),
            remat_sublayers=bool(s_raw["remat_sublayers"]),
            chunk_ce_size=int(s_raw["chunk_ce_size"]),
            target_tokens=int(s_raw["target_tokens"]),
            checkpoint_every_steps=int(s_raw["checkpoint_every_steps"]),
            web_en_ratio=float(s_raw["web_en_ratio"]),
            web_zh_ratio=float(s_raw["web_zh_ratio"]),
            code_ratio=float(s_raw["code_ratio"]),
            math_ratio=float(s_raw["math_ratio"]),
            web_tier=str(s_raw["web_tier"]),
            code_tier=str(s_raw["code_tier"]),
            math_tier=str(s_raw["math_tier"]),
            moe_chunk_size=int(s_raw.get("moe_chunk_size", model_cfg.moe_chunk_size)),
            use_csa2_indexer=(
                None if "use_csa2_indexer" not in s_raw else bool(s_raw["use_csa2_indexer"])
            ),
        )
        if st_cfg.seq_len < 1 or st_cfg.num_loops < 1 or st_cfg.microbatch < 1 or st_cfg.grad_accum < 1:
            raise ValueError(f"Stage {st_cfg.name} dimensions must all be >= 1.")
        if st_cfg.num_loops > model_cfg.max_loops:
            raise ValueError(
                f"Stage {st_cfg.name} num_loops={st_cfg.num_loops} exceeds model.max_loops={model_cfg.max_loops}."
            )
        if (st_cfg.web_en_ratio + st_cfg.web_zh_ratio + st_cfg.code_ratio + st_cfg.math_ratio) <= 0.0:
            raise ValueError(f"Stage {st_cfg.name} data ratios must sum to > 0.")
        stages.append(st_cfg)

    return TrainFileConfig(
        model=model_cfg,
        optimizer=optim_cfg,
        training=training_cfg,
        lookahead=lookahead_cfg,
        stages=tuple(stages),
    )
