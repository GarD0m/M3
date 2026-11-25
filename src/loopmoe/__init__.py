"""1B 1-Layer DeepSeek-V4.1-Flash LoopMoE package."""

from .config import (
    DEFAULT_CURRICULUM,
    SPECIAL_TOKENS,
    LoopMoEConfig,
    StageConfig,
    count_analytic_params,
    estimate_step_flops,
)
from .lookahead import (
    ExplicitStageConfig,
    LookaheadConfig,
    LookaheadDecision,
    LookaheadState,
    LookaheadTracker,
    TrainFileConfig,
    TrainingRuntimeConfig,
    load_train_config,
)
from .model import (
    chunked_cross_entropy_loss,
    chunked_cross_entropy_loss_and_nll,
    count_tree_params,
    expand_checkpoint_1_to_8_loops,
    forward_dynamic_exit,
    forward_model,
    init_model_params,
)

__all__ = [
    "DEFAULT_CURRICULUM",
    "ExplicitStageConfig",
    "LookaheadConfig",
    "LookaheadDecision",
    "LookaheadState",
    "LookaheadTracker",
    "SPECIAL_TOKENS",
    "LoopMoEConfig",
    "StageConfig",
    "TrainFileConfig",
    "TrainingRuntimeConfig",
    "chunked_cross_entropy_loss",
    "chunked_cross_entropy_loss_and_nll",
    "count_analytic_params",
    "count_tree_params",
    "estimate_step_flops",
    "expand_checkpoint_1_to_8_loops",
    "forward_dynamic_exit",
    "forward_model",
    "init_model_params",
    "load_train_config",
]
