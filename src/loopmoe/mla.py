"""DeepSeek-V4.1-Flash Multi-head Latent Attention (MLA), YaRN RoPE, Splash Attention, and optional CSA2 Indexer."""

from __future__ import annotations

import dataclasses
import inspect
import math
import warnings

import jax
import jax.numpy as jnp

from .config import LoopMoEConfig
from .mhc import rms_norm


def compute_yarn_freqs(
    d_rope: int,
    theta: float = 10_000.0,
    base_seq_len: int = 4096,
    scale: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[jax.Array, float]:
    """Compute YaRN inverse frequencies and attention mscale factor."""
    dim_idx = jnp.arange(0, d_rope, 2, dtype=jnp.float32)
    freq_extra = 1.0 / (theta ** (dim_idx / float(d_rope)))

    if scale <= 1.0:
        return freq_extra, 1.0

    freq_inter = 1.0 / (scale * (theta ** (dim_idx / float(d_rope))))

    def _find_correction_dim(num_rotations: float) -> float:
        return (d_rope * math.log(base_seq_len / (num_rotations * 2.0 * math.pi))) / (
            2.0 * math.log(theta)
        )

    low = max(math.floor(_find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(_find_correction_dim(beta_slow)), d_rope - 1)
    if low == high:
        high += 0.001

    half_idx = jnp.arange(d_rope // 2, dtype=jnp.float32)
    linear_func = (half_idx - float(low)) / float(high - low)
    ramp = jnp.clip(linear_func, 0.0, 1.0)
    # smooth == 1 keeps high frequency (extrapolation), 0 uses interpolated frequency
    inv_freq_mask = 1.0 - ramp
    inv_freq = freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask
    mscale = 0.1 * math.log(scale) + 1.0
    return inv_freq, mscale


def apply_rope(x: jax.Array, inv_freq: jax.Array) -> jax.Array:
    """Apply half-split (NeoX) rotary position embeddings to [B, S, H, D_rope] in native dtype."""
    seq_len = x.shape[1]
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    angles = positions * inv_freq[None, :]  # [S, D_rope // 2]
    cos = jnp.cos(angles)[None, :, None, :].astype(x.dtype)
    sin = jnp.sin(angles)[None, :, None, :].astype(x.dtype)

    x1, x2 = jnp.split(x, 2, axis=-1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return jnp.concatenate([out1, out2], axis=-1)


def is_per_loop_mla_leaf(name: str, arr: jax.Array) -> bool:
    """Return True when an MLA parameter leaf has a leading `max_loops` axis."""
    return (arr.ndim == 3) or (name.endswith("_norm_w") and arr.ndim == 2)


def _init_single_mla_qo_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize per-loop MLA query & output projection matrices."""
    keys = jax.random.split(key, 3)
    d = cfg.d_model
    h = cfg.n_heads
    return {
        "w_dq": jax.random.normal(keys[0], (d, cfg.d_q_lora), dtype=dtype)
        * (1.0 / jnp.sqrt(float(d))),
        "q_norm_w": jnp.ones((cfg.d_q_lora,), dtype=dtype),
        "w_uq": jax.random.normal(keys[1], (cfg.d_q_lora, h * cfg.d_qk_head), dtype=dtype)
        * (1.0 / jnp.sqrt(float(cfg.d_q_lora))),
        "w_o": jax.random.normal(keys[2], (h * cfg.d_v, d), dtype=dtype)
        * (1.0 / jnp.sqrt(float(h * cfg.d_v))),
    }


def _init_single_mla_kv_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize MLA KV projections and optional CSA2 Lightning Indexer (reusable across loops)."""
    keys = jax.random.split(key, 7)
    d = cfg.d_model
    h = cfg.n_heads

    params: dict[str, jax.Array] = {
        "w_dkv": jax.random.normal(keys[0], (d, cfg.d_kv_lora + cfg.d_qk_rope), dtype=dtype)
        * (1.0 / jnp.sqrt(float(d))),
        "kv_norm_w": jnp.ones((cfg.d_kv_lora,), dtype=dtype),
        "w_uk": jax.random.normal(keys[1], (cfg.d_kv_lora, h * cfg.d_qk_nope), dtype=dtype)
        * (1.0 / jnp.sqrt(float(cfg.d_kv_lora))),
        "w_uv": jax.random.normal(keys[2], (cfg.d_kv_lora, h * cfg.d_v), dtype=dtype)
        * (1.0 / jnp.sqrt(float(cfg.d_kv_lora))),
    }

    if cfg.use_csa2_indexer:
        params.update(
            {
                "idx_w_q": jax.random.normal(
                    keys[3], (cfg.d_q_lora, cfg.csa2_num_heads * cfg.csa2_head_dim), dtype=dtype
                )
                * (1.0 / jnp.sqrt(float(cfg.d_q_lora))),
                "idx_w_k": jax.random.normal(keys[4], (d, cfg.csa2_head_dim), dtype=dtype)
                * (1.0 / jnp.sqrt(float(d))),
                "idx_k_norm_w": jnp.ones((cfg.csa2_head_dim,), dtype=dtype),
                "idx_w_score": jax.random.normal(keys[5], (d, cfg.csa2_num_heads), dtype=dtype)
                * (1.0 / jnp.sqrt(float(d))),
            }
        )
    return params


def _init_single_mla_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize a single loop's complete MLA projection matrices and optional CSA2 Indexer."""
    k_qo, k_kv = jax.random.split(key, 2)
    return {
        **_init_single_mla_qo_params(k_qo, cfg, dtype),
        **_init_single_mla_kv_params(k_kv, cfg, dtype),
    }


def init_mla_params(
    key: jax.Array,
    cfg: LoopMoEConfig,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Initialize MLA weights (cyclically untied Q/O banks, sharing Loop-0 KV & CSA2 across loops)."""
    n_banks = min(cfg.max_loops, max(1, cfg.mla_weight_banks))
    if cfg.untie_mla_per_loop and cfg.reuse_kv_across_loops:
        k_qo, k_kv = jax.random.split(key, 2)
        loop_keys = jax.random.split(k_qo, n_banks)
        qo_params = jax.vmap(lambda k: _init_single_mla_qo_params(k, cfg, dtype))(loop_keys)
        kv_params = _init_single_mla_kv_params(k_kv, cfg, dtype)
        return {**qo_params, **kv_params}
    if cfg.untie_mla_per_loop:
        loop_keys = jax.random.split(key, n_banks)
        return jax.vmap(lambda k: _init_single_mla_params(k, cfg, dtype))(loop_keys)
    return _init_single_mla_params(key, cfg, dtype)


def quantize_kv_fp4_e2m1(kv: jax.Array, group_size: int = 16) -> tuple[jax.Array, jax.Array]:
    """DeepSeek-V4.1-Flash simulated FP4 (E2M1) per-16-channel KV quantization for eval/serving."""
    orig_shape = kv.shape
    last_dim = orig_shape[-1]
    pad = (group_size - (last_dim % group_size)) % group_size
    kv_f32 = kv.astype(jnp.float32)
    if pad:
        kv_f32 = jnp.pad(kv_f32, [(0, 0)] * (kv_f32.ndim - 1) + [(0, pad)])
    grouped = kv_f32.reshape(*orig_shape[:-1], -1, group_size)
    max_abs = jnp.maximum(jnp.max(jnp.abs(grouped), axis=-1, keepdims=True), 1e-8)
    scale = max_abs / 6.0  # E2M1 max magnitude = 6.0
    normed = grouped / scale

    # Nearest E2M1 grid values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    e2m1_pos = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=jnp.float32)
    signs = jnp.sign(normed)
    abs_val = jnp.abs(normed)[..., None]
    nearest_idx = jnp.argmin(jnp.abs(abs_val - e2m1_pos), axis=-1)
    q_val = signs * e2m1_pos[nearest_idx]
    dequant = (q_val * scale).reshape(*orig_shape[:-1], last_dim + pad)[..., :last_dim]
    return dequant.astype(kv.dtype), scale.squeeze(-1).astype(jnp.bfloat16)


def compute_csa2_lightning_mask(
    params: dict[str, jax.Array],
    x: jax.Array,
    c_q: jax.Array,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, jax.Array]:
    """Compute CSA2 Lightning Indexer causal Top-K mask and raw index scores."""
    bsz, seq_len, _ = x.shape
    q_idx = (c_q @ params["idx_w_q"]).reshape(bsz, seq_len, cfg.csa2_num_heads, cfg.csa2_head_dim)
    k_idx = rms_norm(x @ params["idx_w_k"], params["idx_k_norm_w"], eps=cfg.rms_norm_eps)
    weights = jax.nn.relu(x @ params["idx_w_score"])  # [B, S, H_idx] in native BF16

    # Score_{i, j} = sum_h w_{i, h} * ReLU(q_{i, h} . k_j)
    raw = jnp.einsum("bshd,btd->bsht", q_idx, k_idx)
    raw = jax.nn.relu(raw)
    scores = jnp.sum(weights[:, :, :, None] * raw, axis=2).astype(x.dtype)  # [B, S, S]

    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, :, :]
    mask_val = jnp.asarray(-1e4 if x.dtype == jnp.bfloat16 else -1e9, dtype=x.dtype)
    scores = jnp.where(causal, scores, mask_val)

    k_select = min(cfg.csa2_topk, seq_len)
    _, topk_idx = jax.lax.top_k(scores, k_select)
    mask = jnp.zeros((bsz, seq_len, seq_len), dtype=bool)
    b_idx = jnp.arange(bsz)[:, None, None]
    q_pos = jnp.arange(seq_len)[None, :, None]
    mask = mask.at[b_idx, q_pos, topk_idx].set(True) & causal
    return mask, scores


def _pad_last_dim_to_multiple(x: jax.Array, multiple: int) -> jax.Array:
    dim = x.shape[-1]
    rem = dim % multiple
    if rem == 0:
        return x
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, multiple - rem)])


def _run_causal_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    softmax_scale: float,
    use_tokamax: bool,
    sparse_mask: jax.Array | None = None,
) -> jax.Array:
    """Execute high-MFU TPU Splash/Tokamax attention with safe fallback on CPU/GPU."""
    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    seq_len = q.shape[1]
    on_tpu = jax.default_backend() == "tpu"

    def _sdpa_matched(q_in: jax.Array, k_in: jax.Array, v_in: jax.Array, mask_in=None) -> jax.Array:
        max_d = max(d_qk, d_v)
        q_m = _pad_last_dim_to_multiple(q_in, max_d) if d_qk < max_d else q_in
        k_m = _pad_last_dim_to_multiple(k_in, max_d) if d_qk < max_d else k_in
        v_m = _pad_last_dim_to_multiple(v_in, max_d) if d_v < max_d else v_in
        if mask_in is not None:
            out_m = jax.nn.dot_product_attention(
                q_m,
                k_m,
                v_m,
                mask=mask_in[:, None, :, :],
                scale=softmax_scale,
                is_causal=False,
            )
        else:
            out_m = jax.nn.dot_product_attention(
                q_m,
                k_m,
                v_m,
                is_causal=True,
                scale=softmax_scale,
            )
        return out_m[..., :d_v]

    # Sparse mask path (CSA2 Top-K) uses masked SDPA directly
    if sparse_mask is not None:
        return _sdpa_matched(q, k, v, mask_in=sparse_mask)

    if use_tokamax and on_tpu:
        target_dim = max(((max(d_qk, d_v) + 127) // 128) * 128, 128)
        q_pad = _pad_last_dim_to_multiple(q, target_dim) if d_qk != target_dim else q
        k_pad = _pad_last_dim_to_multiple(k, target_dim) if d_qk != target_dim else k
        v_pad = _pad_last_dim_to_multiple(v, target_dim) if d_v != target_dim else v

        if seq_len >= 128 and seq_len % 128 == 0:
            blk = (min(512, seq_len) // 128) * 128
            blk_kv_dkv = (min(1024, seq_len) // 128) * 128
            blk_c = min(256, blk)

            if seq_len <= 4096:
                try:
                    from jax.experimental.pallas.ops.tpu.splash_attention import QKVLayout  # type: ignore
                    from tokamax._src.ops.attention import pallas_mosaic_tpu as _tpu_attn  # type: ignore
                    from tokamax._src.ops.attention import pallas_mosaic_tpu_vjp as _tpu_attn_vjp  # type: ignore

                    vjp_op = _tpu_attn_vjp.PallasMosaicTpuFlashAttentionVjp(
                        config=_tpu_attn_vjp.Config(
                            block_q_dkv=blk,
                            block_kv_dkv=blk_kv_dkv,
                            block_kv_dkv_compute=blk_c,
                            use_base2_exp=True,
                        )
                    )
                    attn_op = _tpu_attn.PallasMosaicTpuFlashAttention(
                        config=_tpu_attn.Config(
                            block_q=blk,
                            block_kv=blk,
                            block_kv_compute=blk_c,
                            q_layout=QKVLayout.HEAD_DIM_MINOR,
                            k_layout=QKVLayout.HEAD_DIM_MINOR,
                            v_layout=QKVLayout.HEAD_DIM_MINOR,
                            use_experimental_scheduler=False,
                            use_base2_exp=True,
                        ),
                        vjp=vjp_op,
                    )
                    out = attn_op(q_pad, k_pad, v_pad, is_causal=True, logits_scale=softmax_scale)
                    return out[..., :d_v]
                except Exception as exc:
                    warnings.warn(
                        f"[MLA] PallasMosaicTpuFlashAttention failed at seq_len={seq_len}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                # For >=8K, fused backward materializes f32[B, S//1024, H, S, 128] (8.59 GB at 32K!).
                # Use unfused Splash Attention backward (separate dKV + dQ kernels) so dQ is reduced in VMEM.
                blk_dq = min(256, blk)

                def _invoke_unfused_splash(
                    make_fn,
                    block_sizes_cls,
                    qkv_layout_cls,
                    causal_mask_cls,
                    mh_mask_cls,
                ) -> jax.Array:
                    valid_fields = {f.name for f in dataclasses.fields(block_sizes_cls)}
                    candidates = {
                        "block_q": blk,
                        "block_kv": blk,
                        "block_kv_compute": blk_c,
                        "block_q_dkv": blk,
                        "block_kv_dkv": blk_kv_dkv,
                        "block_kv_dkv_compute": blk_c,
                        "block_q_dq": blk_dq,
                        "block_kv_dq": blk_dq,
                        "block_kv_dq_compute": blk_c,
                        "use_fused_bwd_kernel": False,
                        "q_layout": qkv_layout_cls.HEAD_DIM_MINOR,
                        "k_layout": qkv_layout_cls.HEAD_DIM_MINOR,
                        "v_layout": qkv_layout_cls.HEAD_DIM_MINOR,
                        "use_base2_exp": True,
                        "fuse_reciprocal": True,
                    }
                    bs = block_sizes_cls(**{k: v for k, v in candidates.items() if k in valid_fields})
                    num_heads = q_pad.shape[2]
                    mask = mh_mask_cls([causal_mask_cls(shape=(seq_len, seq_len)) for _ in range(num_heads)])
                    mk_sig = inspect.signature(make_fn)
                    mk_kwargs: dict[str, object] = {}
                    if "config" in mk_sig.parameters:
                        mk_kwargs["config"] = bs
                    elif "block_sizes" in mk_sig.parameters:
                        mk_kwargs["block_sizes"] = bs
                    for k, v in (
                        ("head_shards", 1),
                        ("q_seq_shards", 1),
                        ("downcast_smem_data", True),
                    ):
                        if k in mk_sig.parameters:
                            mk_kwargs[k] = v
                    splash_fn = make_fn(mask, **mk_kwargs)
                    q_hsd = jnp.swapaxes(q_pad * jnp.asarray(softmax_scale, dtype=q_pad.dtype), 1, 2)
                    k_hsd = jnp.swapaxes(k_pad, 1, 2)
                    v_hsd = jnp.swapaxes(v_pad, 1, 2)
                    out_hsd = jax.vmap(splash_fn)(q_hsd, k_hsd, v_hsd)
                    return jnp.swapaxes(out_hsd, 1, 2)[..., :d_v]

                try:
                    from tokamax._src.ops.splash_attention import splash_attention_kernel as _tkx_splash  # type: ignore
                    from jax.experimental.pallas.ops.tpu.splash_attention import (  # type: ignore
                        CausalMask as _JaxCausalMask,
                        MultiHeadMask as _JaxMultiHeadMask,
                        QKVLayout as _JaxQKVLayout,
                    )

                    try:
                        from tokamax._src.ops.splash_attention import splash_attention_mask as _tkx_mask  # type: ignore
                    except Exception:
                        _tkx_mask = None

                    _cfg_cls = getattr(_tkx_splash, "SplashConfig", None) or getattr(
                        _tkx_splash, "BlockSizes", None
                    )
                    if _cfg_cls is None:
                        raise AttributeError("tokamax splash_attention_kernel lacks SplashConfig/BlockSizes")
                    _cm_cls = (
                        getattr(_tkx_mask, "CausalMask", None)
                        or getattr(_tkx_splash, "CausalMask", _JaxCausalMask)
                    )
                    _mh_cls = (
                        getattr(_tkx_mask, "MultiHeadMask", None)
                        or getattr(_tkx_splash, "MultiHeadMask", _JaxMultiHeadMask)
                    )
                    _ql_cls = getattr(_tkx_splash, "QKVLayout", _JaxQKVLayout)
                    return _invoke_unfused_splash(
                        _tkx_splash.make_splash_mha, _cfg_cls, _ql_cls, _cm_cls, _mh_cls
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[MLA] tokamax unfused splash_attention failed at seq_len={seq_len}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                try:
                    from jax.experimental.pallas.ops.tpu.splash_attention import (
                        BlockSizes as _JaxBlockSizes,
                        CausalMask as _JaxCausalMask,
                        MultiHeadMask as _JaxMultiHeadMask,
                        QKVLayout as _JaxQKVLayout,
                        make_splash_mha as _jax_make_splash_mha,
                    )

                    return _invoke_unfused_splash(
                        _jax_make_splash_mha,
                        _JaxBlockSizes,
                        _JaxQKVLayout,
                        _JaxCausalMask,
                        _JaxMultiHeadMask,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[MLA] jax unfused splash_attention failed at seq_len={seq_len}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if seq_len <= 4096:
            try:
                import tokamax  # type: ignore

                fn = getattr(tokamax, "dot_product_attention", None)
                if fn is not None:
                    sig = inspect.signature(fn)
                    kwargs: dict[str, object] = {"is_causal": True}
                    if "logits_scale" in sig.parameters:
                        kwargs["logits_scale"] = softmax_scale
                    elif "scale" in sig.parameters:
                        kwargs["scale"] = softmax_scale
                    if "implementation" in sig.parameters:
                        kwargs["implementation"] = "mosaic"
                    out = fn(q_pad, k_pad, v_pad, **kwargs)
                    return out[..., :d_v]
            except Exception as exc:
                warnings.warn(
                    f"[MLA] tokamax.dot_product_attention failed at seq_len={seq_len}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if seq_len >= 128 and seq_len % 128 == 0:
            try:
                from jax.experimental.pallas.ops.tpu.splash_attention import (
                    CausalMask,
                    MultiHeadMask,
                    make_splash_mha,
                )

                num_heads = q_pad.shape[2]
                mask = MultiHeadMask([CausalMask(shape=(seq_len, seq_len)) for _ in range(num_heads)])
                splash_fn = make_splash_mha(mask=mask, head_shards=1, q_seq_shards=1)
                # Splash attention takes [H, S, D] per batch item and does not scale Q internally
                q_hsd = jnp.swapaxes(q_pad * jnp.asarray(softmax_scale, dtype=q_pad.dtype), 1, 2)
                k_hsd = jnp.swapaxes(k_pad, 1, 2)
                v_hsd = jnp.swapaxes(v_pad, 1, 2)
                out_hsd = jax.vmap(splash_fn)(q_hsd, k_hsd, v_hsd)
                return jnp.swapaxes(out_hsd, 1, 2)[..., :d_v]
            except Exception as exc:
                warnings.warn(
                    f"[MLA] splash_attention failed at seq_len={seq_len}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    return _sdpa_matched(q, k, v)


def apply_mla(
    params: dict[str, jax.Array],
    x: jax.Array,
    cfg: LoopMoEConfig,
    loop_idx: jax.Array | int = 0,
    cached_kv: tuple[jax.Array, jax.Array, jax.Array | None] | None = None,
    return_kv: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]] | tuple[jax.Array, dict[str, jax.Array], tuple[jax.Array, jax.Array, jax.Array | None]]:
    """Run Multi-head Latent Attention with optional GLM-5.4 Loop-0 KV & CSA2 mask reuse."""
    bsz, seq_len, _ = x.shape
    h = cfg.n_heads
    p = {
        k: (v[loop_idx % v.shape[0]] if is_per_loop_mla_leaf(k, v) else v)
        for k, v in params.items()
    }

    with jax.named_scope("mla_qkv_proj"):
        c_q = rms_norm(x @ p["w_dq"], p["q_norm_w"], eps=cfg.rms_norm_eps)
        half = cfg.d_qk_rope // 2
        w_uq_4d = p["w_uq"].reshape(cfg.d_q_lora, h, cfg.d_qk_head)
        w_uq_nope = w_uq_4d[:, :, : cfg.d_qk_nope].reshape(cfg.d_q_lora, h * cfg.d_qk_nope)
        w_uq_r1 = w_uq_4d[:, :, cfg.d_qk_nope : cfg.d_qk_nope + half].reshape(cfg.d_q_lora, h * half)
        w_uq_r2 = w_uq_4d[:, :, cfg.d_qk_nope + half :].reshape(cfg.d_q_lora, h * half)
        q_nope = (c_q @ w_uq_nope).reshape(bsz, seq_len, h, cfg.d_qk_nope)
        q_r1 = c_q @ w_uq_r1  # [B, S, H * half] -> width 256 (2 full 128-lanes, zero padding)
        q_r2 = c_q @ w_uq_r2

        if cached_kv is None:
            kv_raw = x @ p["w_dkv"]
            c_kv_raw, k_rope_shared = jnp.split(kv_raw, [cfg.d_kv_lora], axis=-1)
            c_kv = rms_norm(c_kv_raw, p["kv_norm_w"], eps=cfg.rms_norm_eps)
            k_nope = (c_kv @ p["w_uk"]).reshape(bsz, seq_len, h, cfg.d_qk_nope)
            v = (c_kv @ p["w_uv"]).reshape(bsz, seq_len, h, cfg.d_v)
        else:
            k_rope_shared = None
            k_nope = None
            v = cached_kv[1]

    with jax.named_scope("mla_rope"):
        inv_freq, mscale = compute_yarn_freqs(
            d_rope=cfg.d_qk_rope,
            theta=cfg.rope_theta,
            base_seq_len=cfg.base_seq_len,
            scale=cfg.yarn_scale,
            beta_fast=cfg.yarn_beta_fast,
            beta_slow=cfg.yarn_beta_slow,
        )
        positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
        angles = positions * inv_freq[None, :]  # [S, half]
        cos_2d = jnp.cos(angles).astype(x.dtype)
        sin_2d = jnp.sin(angles).astype(x.dtype)

        cos_h = jnp.tile(cos_2d, (1, h))[None, :, :]  # [1, S, H * half]
        sin_h = jnp.tile(sin_2d, (1, h))[None, :, :]
        q_out1 = (q_r1 * cos_h - q_r2 * sin_h).reshape(bsz, seq_len, h, half)
        q_out2 = (q_r1 * sin_h + q_r2 * cos_h).reshape(bsz, seq_len, h, half)
        q = jnp.concatenate([q_nope, q_out1, q_out2], axis=-1)

        if cached_kv is None:
            assert k_rope_shared is not None and k_nope is not None
            k_r1, k_r2 = jnp.split(k_rope_shared, 2, axis=-1)
            k_out1 = k_r1 * cos_2d[None, :, :] - k_r2 * sin_2d[None, :, :]
            k_out2 = k_r1 * sin_2d[None, :, :] + k_r2 * cos_2d[None, :, :]
            k_rope = jnp.broadcast_to(
                jnp.concatenate([k_out1, k_out2], axis=-1)[:, :, None, :],
                (bsz, seq_len, h, cfg.d_qk_rope),
            )
            k = jnp.concatenate([k_nope, k_rope], axis=-1)
        else:
            k = cached_kv[0]

    softmax_scale = (mscale * mscale) / math.sqrt(float(cfg.d_qk_head))

    aux_info: dict[str, jax.Array] = {}
    if cached_kv is None:
        sparse_mask = None
        if cfg.use_csa2_indexer and seq_len > cfg.csa2_topk:
            with jax.named_scope("mla_csa2_indexer"):
                sparse_mask, idx_scores = compute_csa2_lightning_mask(p, x, c_q, cfg)
                aux_info["csa2_score_mean"] = jnp.mean(idx_scores)
        kv_out = (k, v, sparse_mask)
    else:
        sparse_mask = cached_kv[2]
        kv_out = cached_kv

    with jax.named_scope("mla_attention"):
        attn_out = _run_causal_attention(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            use_tokamax=cfg.attn_use_tokamax,
            sparse_mask=sparse_mask,
        )
    with jax.named_scope("mla_out_proj"):
        out = attn_out.reshape(bsz, seq_len, h * cfg.d_v) @ p["w_o"]
    if return_kv:
        return out, aux_info, kv_out
    return out, aux_info
