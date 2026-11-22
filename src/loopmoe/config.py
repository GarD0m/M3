"""Model configuration, 7-stage curriculum, and exact analytic parameter/FLOPs accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


BASE_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|tool_call|>",
    "</|tool_call|>",
    "<|tool_result|>",
    "</|tool_result|>",
    "<|code|>",
    "<|math|>",
)

NUM_TOTAL_SPECIAL_TOKENS = 64
SPECIAL_TOKENS: tuple[str, ...] = BASE_SPECIAL_TOKENS + tuple(
    f"<|reserved_{i}|>" for i in range(NUM_TOTAL_SPECIAL_TOKENS - len(BASE_SPECIAL_TOKENS))
)
SPECIAL_TOKEN_IDS: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}


def is_prime(n: int) -> bool:
    """Deterministic primality test for positive integers."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    limit = math.isqrt(n)
    k = 5
    while k <= limit:
        if n % k == 0 or n % (k + 2) == 0:
            return False
        k += 6
    return True


def generate_prime_table_sizes(total_slots: int, num_tables: int) -> tuple[int, ...]:
    """Generate `num_tables` distinct prime bucket sizes summing close to `total_slots`."""
    if num_tables <= 0:
        return ()
    target = max(17, total_slots // num_tables)
    primes: list[int] = []
    candidate = target if target % 2 == 1 else target + 1
    while len(primes) < num_tables:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 2
    return tuple(primes)


@dataclass(frozen=True)
class LoopMoEConfig:
    """Configuration for 1B 1-Layer DeepSeek-V4.1-Flash LoopMoE."""

    # Core backbone
    vocab_size: int = 65_536
    d_model: int = 2048
    num_layers: int = 1
    max_loops: int = 8
    num_loops: int = 1
    tie_word_embeddings: bool = True
    dtype_name: str = "bfloat16"
    rms_norm_eps: float = 1e-6
    chunk_ce_size: int = 2048

    # RoPE / YaRN long-context scaling
    rope_theta: float = 10_000.0
    base_seq_len: int = 4096
    max_seq_len: int = 32768
    yarn_scale: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0

    # DeepSeek-V4.1-Flash Single-Pass mHC
    use_mhc: bool = True
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_alpha_init: float = 0.01

    # DeepSeek-V4.1-Flash Engram Conditional Memory (~22.5% of sparse capacity)
    use_engram: bool = True
    engram_orders: tuple[int, ...] = (2, 3)
    engram_hash_heads: int = 4
    engram_head_dim: int = 128
    engram_total_slots: int = 1_460_000
    engram_conv_kernel: int = 4
    engram_conv_dilation: int = 3
    engram_seed: int = 20260112

    # MLA + optional CSA2 Lightning Indexer
    n_heads: int = 16
    d_q_lora: int = 768
    d_kv_lora: int = 512
    d_qk_nope: int = 128
    d_qk_rope: int = 64
    d_v: int = 128
    attn_use_tokamax: bool = True
    use_csa2_indexer: bool = False
    csa2_num_heads: int = 4
    csa2_head_dim: int = 64
    csa2_topk: int = 2048

    # DeepSeek fine-grained MoE + LoopMoE Capacity Balancing
    n_routed_experts: int = 416
    d_ff_expert: int = 256
    n_shared_experts: int = 1
    d_ff_shared: int = 512
    top_k: int = 12
    routed_scaling_factor: float = 2.5
    router_bias_update_rate: float = 1e-3
    seq_aux_loss_weight: float = 1e-4
    moe_use_tokamax: bool = True

    # LoopMoE IterAdaLN
    iter_adaln_bottleneck: int = 128

    # TPU execution
    remat_per_loop: bool = True
    remat_sublayers: bool = False

    @property
    def num_engram_tables(self) -> int:
        return len(self.engram_orders) * self.engram_hash_heads

    @property
    def engram_table_sizes(self) -> tuple[int, ...]:
        if not self.use_engram:
            return ()
        return generate_prime_table_sizes(self.engram_total_slots, self.num_engram_tables)

    @property
    def d_engram(self) -> int:
        return self.num_engram_tables * self.engram_head_dim

    @property
    def d_qk_head(self) -> int:
        return self.d_qk_nope + self.d_qk_rope

    def yarn_scale_for_seq_len(self, seq_len: int) -> float:
        if seq_len <= self.base_seq_len:
            return 1.0
        return float(seq_len) / float(self.base_seq_len)

    def with_stage(self, seq_len: int, num_loops: int, use_csa2_indexer: bool | None = None) -> LoopMoEConfig:
        """Return a copy adjusted for a curriculum stage's sequence length and loop count."""
        return LoopMoEConfig(
            **{
                **asdict(self),
                "num_loops": num_loops,
                "yarn_scale": self.yarn_scale_for_seq_len(seq_len),
                "use_csa2_indexer": (
                    self.use_csa2_indexer if use_csa2_indexer is None else use_csa2_indexer
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopMoEConfig:
        fixed = dict(data)
        if "engram_orders" in fixed and isinstance(fixed["engram_orders"], list):
            fixed["engram_orders"] = tuple(fixed["engram_orders"])
        return cls(**fixed)

    @classmethod
    def tiny(cls, num_loops: int = 1, use_csa2_indexer: bool = False) -> LoopMoEConfig:
        """Small fast configuration for local CPU verification tests."""
        return cls(
            vocab_size=512,
            d_model=64,
            max_loops=8,
            num_loops=num_loops,
            chunk_ce_size=16,
            base_seq_len=32,
            max_seq_len=128,
            mhc_streams=4,
            mhc_sinkhorn_iters=10,
            engram_orders=(2, 3),
            engram_hash_heads=2,
            engram_head_dim=8,
            engram_total_slots=400,
            engram_conv_kernel=4,
            engram_conv_dilation=1,
            n_heads=4,
            d_q_lora=32,
            d_kv_lora=24,
            d_qk_nope=8,
            d_qk_rope=8,
            d_v=8,
            attn_use_tokamax=False,
            use_csa2_indexer=use_csa2_indexer,
            csa2_num_heads=2,
            csa2_head_dim=8,
            csa2_topk=16,
            n_routed_experts=8,
            d_ff_expert=32,
            n_shared_experts=1,
            d_ff_shared=32,
            top_k=2,
            moe_use_tokamax=False,
            iter_adaln_bottleneck=16,
            remat_per_loop=False,
            remat_sublayers=False,
        )

    @classmethod
    def for_tpu_v5e(cls, num_loops: int = 1, use_csa2_indexer: bool = False) -> LoopMoEConfig:
        """Full 1B preset tuned for TPU v5e-1 (16 GB HBM, 197 TFLOPS, 128x128 MXU)."""
        return cls(
            num_loops=num_loops,
            chunk_ce_size=1024,
            use_csa2_indexer=use_csa2_indexer,
            remat_per_loop=True,
            remat_sublayers=True,
        )


TPU_SPECS: dict[str, dict[str, float]] = {
    "v5e-1": {
        "peak_tflops_bf16": 197.0,
        "hbm_bw_gbps": 819.0,
        "hbm_capacity_gb": 16.0,
        "mxu_size": 128.0,
        "ridge_flops_per_byte": (197.0 * 1e12) / (819.0 * 1e9),
        "default_microbatch_tokens": 8192.0,
        "default_chunk_ce_size": 1024.0,
    },
    "v6e-1": {
        "peak_tflops_bf16": 918.0,
        "hbm_bw_gbps": 1600.0,
        "hbm_capacity_gb": 32.0,
        "mxu_size": 256.0,
        "ridge_flops_per_byte": (918.0 * 1e12) / (1600.0 * 1e9),
        "default_microbatch_tokens": 32768.0,
        "default_chunk_ce_size": 2048.0,
    },
}


@dataclass(frozen=True)
class StageConfig:
    """Single curriculum stage specification."""

    stage_id: int
    name: str
    seq_len: int
    target_tokens: int
    num_loops: int
    global_batch_tokens: int = 262_144
    microbatch_tokens: int = 32_768
    warmup_steps: int = 100
    web_en_ratio: float = 0.32
    web_zh_ratio: float = 0.28
    code_ratio: float = 0.22
    math_ratio: float = 0.18
    web_tier: str = "L3"
    code_tier: str = "L3"
    math_tier: str = "L3"
    use_csa2_indexer: bool = False
    checkpoint_every_steps: int = 500

    @property
    def micro_batch_size(self) -> int:
        if self.microbatch_tokens % self.seq_len != 0:
            raise ValueError(
                f"microbatch_tokens ({self.microbatch_tokens}) must be divisible by seq_len ({self.seq_len})"
            )
        return self.microbatch_tokens // self.seq_len

    @property
    def grad_accum_steps(self) -> int:
        if self.global_batch_tokens % self.microbatch_tokens != 0:
            raise ValueError(
                f"global_batch_tokens ({self.global_batch_tokens}) must be divisible by microbatch_tokens ({self.microbatch_tokens})"
            )
        return self.global_batch_tokens // self.microbatch_tokens

    @property
    def total_steps(self) -> int:
        return math.ceil(self.target_tokens / self.global_batch_tokens)


# Researched 2025-2026 1B SLM progressive bilingual/code/math curriculum (21B tokens total, L3 tiers)
DEFAULT_CURRICULUM: tuple[StageConfig, ...] = (
    StageConfig(
        stage_id=1,
        name="stage1_1k_4B",
        seq_len=1024,
        target_tokens=4_000_000_000,
        num_loops=1,
        warmup_steps=1000,
        web_en_ratio=0.32,
        web_zh_ratio=0.28,
        code_ratio=0.22,
        math_ratio=0.18,
        web_tier="L3",
        code_tier="L3",
        math_tier="L3",
    ),
    StageConfig(
        stage_id=2,
        name="stage2_2k_4B",
        seq_len=2048,
        target_tokens=4_000_000_000,
        num_loops=1,
        warmup_steps=0,
        web_en_ratio=0.28,
        web_zh_ratio=0.24,
        code_ratio=0.25,
        math_ratio=0.23,
        web_tier="L3",
        code_tier="L3",
        math_tier="L3",
    ),
    StageConfig(
        stage_id=3,
        name="stage3_4k_4B",
        seq_len=4096,
        target_tokens=4_000_000_000,
        num_loops=1,
        warmup_steps=0,
        web_en_ratio=0.24,
        web_zh_ratio=0.20,
        code_ratio=0.28,
        math_ratio=0.28,
        web_tier="L3",
        code_tier="L3",
        math_tier="L3",
    ),
    StageConfig(
        stage_id=4,
        name="stage4_8k_4B",
        seq_len=8192,
        target_tokens=4_000_000_000,
        num_loops=1,
        warmup_steps=0,
        web_en_ratio=0.20,
        web_zh_ratio=0.16,
        code_ratio=0.36,
        math_ratio=0.28,
        web_tier="L3",
        code_tier="L3",
        math_tier="L3",
    ),
    StageConfig(
        stage_id=5,
        name="stage5_16k_2B",
        seq_len=16384,
        target_tokens=2_000_000_000,
        num_loops=1,
        warmup_steps=0,
        web_en_ratio=0.18,
        web_zh_ratio=0.14,
        code_ratio=0.40,
        math_ratio=0.28,
        web_tier="L3",
        code_tier="L3",
        math_tier="L3",
    ),
    StageConfig(
        stage_id=6,
        name="stage6_32k_2B",
        seq_len=32768,
        target_tokens=2_000_000_000,
        num_loops=1,
        warmup_steps=0,
        web_en_ratio=0.16,
        web_zh_ratio=0.14,
        code_ratio=0.42,
        math_ratio=0.28,
        web_tier="L3",
        code_tier="L3",
        math_tier="L3",
    ),
    StageConfig(
        stage_id=7,
        name="stage7_4k_8loop_1B",
        seq_len=4096,
        target_tokens=1_000_000_000,
        num_loops=8,
        warmup_steps=200,
        web_en_ratio=0.12,
        web_zh_ratio=0.10,
        code_ratio=0.35,
        math_ratio=0.43,
        web_tier="L3",
        code_tier="L3",
        math_tier="L3",
        checkpoint_every_steps=250,
    ),
)


def count_analytic_params(cfg: LoopMoEConfig) -> dict[str, float | int]:
    """Compute exact parameter counts per module and active FLOP footprint."""
    d = cfg.d_model
    emb = cfg.vocab_size * d
    lm_head = 0 if cfg.tie_word_embeddings else cfg.vocab_size * d
    final_norm = d

    # Single-Pass mHC (attn + moe sublayers) + stream expander/gate
    if cfg.use_mhc:
        s = cfg.mhc_streams
        mhc_per_sublayer = (s * d) + (s * d) * (2 * s + s * s) + (2 * s + s * s) + 3
        mhc = 2 * mhc_per_sublayer + s
    else:
        mhc = 0

    # Engram Conditional Memory
    if cfg.use_engram:
        engram_tables = sum(cfg.engram_table_sizes) * cfg.engram_head_dim
        engram_proj = 2 * cfg.d_engram * d  # W_K, W_V
        engram_norms = 2 * d + d  # query norm, key norm, conv norm
        engram_conv = cfg.engram_conv_kernel * d + d
        engram = engram_tables + engram_proj + engram_norms + engram_conv
    else:
        engram_tables = 0
        engram = 0

    # IterAdaLN
    b = cfg.iter_adaln_bottleneck
    iter_adaln = (
        cfg.max_loops * d
        + (d * b) * 2
        + b * (6 * d)
        + 6 * d
        + cfg.max_loops * 3  # per-loop residual scalars (engram, attn, moe)
        + 2 * d  # base pre-sublayer RMSNorms
    )

    # MLA
    mla = (
        d * cfg.d_q_lora
        + cfg.d_q_lora
        + cfg.d_q_lora * (cfg.n_heads * cfg.d_qk_head)
        + d * (cfg.d_kv_lora + cfg.d_qk_rope)
        + cfg.d_kv_lora
        + cfg.d_kv_lora * (cfg.n_heads * (cfg.d_qk_nope + cfg.d_v))
        + (cfg.n_heads * cfg.d_v) * d
    )
    if cfg.use_csa2_indexer:
        mla += (
            cfg.d_q_lora * (cfg.csa2_num_heads * cfg.csa2_head_dim)
            + d * cfg.csa2_head_dim
            + cfg.csa2_head_dim
            + d * cfg.csa2_num_heads
        )

    # MoE
    moe_routed = cfg.n_routed_experts * (3 * d * cfg.d_ff_expert)
    moe_shared = cfg.n_shared_experts * (3 * d * cfg.d_ff_shared)
    moe_router = d * cfg.n_routed_experts
    moe_bias = cfg.max_loops * cfg.n_routed_experts
    moe = moe_routed + moe_shared + moe_router + moe_bias

    total = emb + lm_head + final_norm + mhc + engram + iter_adaln + mla + moe
    active_ffn = cfg.top_k * (3 * d * cfg.d_ff_expert) + moe_shared
    active_engram_dense = (engram - engram_tables) if cfg.use_engram else 0
    active_per_loop_non_emb = mla + active_ffn + iter_adaln + mhc + active_engram_dense
    active_total = emb + lm_head + cfg.num_loops * active_per_loop_non_emb

    sparse_total = engram + moe
    engram_sparse_ratio = (engram / sparse_total) if sparse_total > 0 else 0.0

    return {
        "embedding": emb,
        "lm_head": lm_head,
        "final_norm": final_norm,
        "mhc": mhc,
        "engram_tables": engram_tables,
        "engram_total": engram,
        "iter_adaln": iter_adaln,
        "mla": mla,
        "moe_routed": moe_routed,
        "moe_shared": moe_shared,
        "moe_router": moe_router + moe_bias,
        "moe_total": moe,
        "total": total,
        "active_ffn_per_loop": active_ffn,
        "active_per_loop_non_emb": active_per_loop_non_emb,
        "active_total_per_token": active_total,
        "attn_to_ffn_active_ratio": active_ffn / max(1, mla),
        "engram_sparse_ratio": engram_sparse_ratio,
    }


def estimate_step_flops(cfg: LoopMoEConfig, batch_size: int, seq_len: int) -> float:
    """Estimate training FLOPs (forward + backward) per microbatch or step for MFU calculation.

    Uses standard transformer 6 * N_active * tokens + attention 12 * n_heads * seq^2 * (d_qk + d_v) * loops.
    """
    stats = count_analytic_params(cfg)
    tokens = batch_size * seq_len
    # Dense matmul FLOPs (embedding lookup is sparse; LM head projection IS a full matmul)
    active_matmul_params = (
        cfg.vocab_size * cfg.d_model  # LM head
        + cfg.num_loops * int(stats["active_per_loop_non_emb"])
    )
    linear_flops = 6.0 * float(active_matmul_params) * float(tokens)
    # Causal attention FLOPs (~6 * B * H * S^2 * (d_qk + d_v) / 2 * loops for fwd+bwd)
    attn_flops = (
        6.0
        * float(cfg.num_loops)
        * float(batch_size)
        * float(cfg.n_heads)
        * (float(seq_len) ** 2)
        * float(cfg.d_qk_head + cfg.d_v)
        * 0.5
    )
    return linear_flops + attn_flops
