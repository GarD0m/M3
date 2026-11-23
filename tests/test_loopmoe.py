"""Comprehensive verification tests for 1B 1-Layer DeepSeek-V4.1-Flash LoopMoE."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from loopmoe.checkpoint import load_latest_checkpoint, save_checkpoint
from loopmoe.config import (
    DEFAULT_CURRICULUM,
    SPECIAL_TOKEN_IDS,
    LoopMoEConfig,
    count_analytic_params,
    is_prime,
)
from loopmoe.mhc import (
    init_mhc_sublayer_params,
    mhc_compute_gates,
    mhc_post_merge,
    rms_norm,
    sinkhorn_knopp,
)
from loopmoe.mla import quantize_kv_fp4_e2m1
from loopmoe.model import (
    _saved_logits_ce,
    _saved_logits_ce_vd,
    chunked_cross_entropy_loss,
    count_tree_params,
    expand_checkpoint_1_to_8_loops,
    forward_logits,
    forward_model,
    init_model_params,
)
from loopmoe.moe import _moe_combine, _moe_dispatch, _permute_1d, update_router_biases
from loopmoe.optim import (
    OptimConfig,
    classify_param,
    create_muon_adamw_optimizer,
    make_warmup_stable_schedule,
    per_expert_newton_schulz,
)
from train_tokenizer import benchmark_tokenizer, build_bpe_tokenizer, save_hf_tokenizer_files


def test_full_1b_analytic_params_and_tiny_exact_match() -> None:
    cfg = LoopMoEConfig()
    stats = count_analytic_params(cfg)
    assert 1.10e9 <= stats["total"] <= 1.20e9, f"Expected ~1.15B params, got {stats['total']}"
    assert 0.20 <= stats["engram_sparse_ratio"] <= 0.25, stats["engram_sparse_ratio"]
    assert all(is_prime(p) for p in cfg.engram_table_sizes)
    assert len(set(cfg.engram_table_sizes)) == len(cfg.engram_table_sizes)

    # Verify curriculum token sum == 21B (20B R=1 + 1B R=8) and all stages use L3
    assert sum(s.target_tokens for s in DEFAULT_CURRICULUM if s.num_loops == 1) == 20_000_000_000
    assert sum(s.target_tokens for s in DEFAULT_CURRICULUM if s.num_loops == 8) == 1_000_000_000
    assert all(
        s.web_tier == "L3" and s.code_tier == "L3" and s.math_tier == "L3"
        for s in DEFAULT_CURRICULUM
    )
    assert all(
        s.seq_len == 4096
        and s.web_en_ratio == 0.40
        and s.web_zh_ratio == 0.40
        and s.code_ratio == 0.10
        and s.math_ratio == 0.10
        and abs((s.web_en_ratio + s.web_zh_ratio + s.code_ratio + s.math_ratio) - 1.0) < 1e-9
        for s in DEFAULT_CURRICULUM
    )
    for s in DEFAULT_CURRICULUM:
        en_s = s.with_english_only()
        assert en_s.seq_len == 4096
        assert en_s.web_en_ratio == 0.80
        assert en_s.web_zh_ratio == 0.0
        assert en_s.code_ratio == 0.10
        assert en_s.math_ratio == 0.10

    # Verify analytic count matches actual initialized pytree leaf count to the exact parameter
    tiny_cfg = LoopMoEConfig.tiny()
    tiny_params = init_model_params(jax.random.PRNGKey(0), tiny_cfg)
    assert count_analytic_params(tiny_cfg)["total"] == count_tree_params(tiny_params)


def test_forward_and_backward_r1_r8_and_csa2() -> None:
    key = jax.random.PRNGKey(42)
    x = jax.random.randint(key, (2, 32), 0, 512, dtype=jnp.int32)
    y = jax.random.randint(jax.random.PRNGKey(43), (2, 32), 0, 512, dtype=jnp.int32)

    for num_loops, use_csa2, remat, remat_sub in [
        (1, False, False, False),
        (8, False, True, True),
        (2, True, False, True),
    ]:
        cfg = LoopMoEConfig.tiny(num_loops=num_loops, use_csa2_indexer=use_csa2)
        cfg = LoopMoEConfig(
            **{**cfg.to_dict(), "remat_per_loop": remat, "remat_sublayers": remat_sub}
        )
        params = init_model_params(jax.random.PRNGKey(7), cfg)
        if num_loops > 1:
            params = expand_checkpoint_1_to_8_loops(params, jax.random.PRNGKey(8))
            # Open gates slightly so all loop steps contribute gradients
            idaln = dict(params["iter_adaln"])
            idaln["loop_alpha"] = jnp.full_like(idaln["loop_alpha"], 0.5)
            params["iter_adaln"] = idaln

        def loss_fn(p):
            loss, aux = forward_model(p, x, y, cfg)
            return loss, aux

        (loss, aux), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
        assert jnp.isfinite(loss)
        assert aux["expert_counts"].shape == (num_loops, cfg.n_routed_experts)
        assert float(jnp.sum(aux["expert_counts"][0])) == float(2 * 32 * cfg.top_k)

        # Every active gradient must be finite; router_bias gradient must be zero (Aux-Loss-Free)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))
        assert float(jnp.max(jnp.abs(grads["moe"]["router_bias"]))) == 0.0
        assert float(jnp.linalg.norm(grads["mla"]["w_dq"])) > 0.0
        assert float(jnp.linalg.norm(grads["moe"]["w_gate_up"])) > 0.0
        assert float(jnp.linalg.norm(grads["engram"]["tables"])) > 0.0


def test_mhc_sinkhorn_doubly_stochastic() -> None:
    logits_std1 = jax.random.normal(jax.random.PRNGKey(11), (3, 16, 4, 4), dtype=jnp.float32)
    mat_20 = sinkhorn_knopp(logits_std1, num_iters=20)
    assert jnp.all(mat_20 >= 0.0)
    np.testing.assert_allclose(np.asarray(jnp.sum(mat_20, axis=-1)), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(jnp.sum(mat_20, axis=-2)), 1.0, atol=1e-4)

    logits_wide = logits_std1 * 3.0
    mat_60 = sinkhorn_knopp(logits_wide, num_iters=60)
    assert jnp.all(mat_60 >= 0.0)
    np.testing.assert_allclose(np.asarray(jnp.sum(mat_60, axis=-1)), 1.0, atol=1e-2)
    np.testing.assert_allclose(np.asarray(jnp.sum(mat_60, axis=-2)), 1.0, atol=1e-4)


def test_r1_to_r8_smooth_expansion_preserves_logits_and_learns() -> None:
    cfg_r1 = LoopMoEConfig.tiny(num_loops=1)
    cfg_r8 = LoopMoEConfig.tiny(num_loops=8)
    params_r1 = init_model_params(jax.random.PRNGKey(99), cfg_r1)

    # Simulate trained non-trivial mHC/IterAdaLN/router_bias weights in Phase 1
    mhc_attn = dict(params_r1["mhc"]["attn"])
    mhc_attn["alpha_res"] = jnp.asarray(0.5, dtype=jnp.float32)
    mhc_attn["b_res"] = jax.random.normal(jax.random.PRNGKey(100), (4, 4), dtype=jnp.float32)
    params_r1["mhc"] = {**params_r1["mhc"], "attn": mhc_attn}
    params_r1["moe"] = {
        **params_r1["moe"],
        "router_bias": params_r1["moe"]["router_bias"].at[0].set(0.25),
    }

    x = jax.random.randint(jax.random.PRNGKey(101), (2, 16), 0, 512, dtype=jnp.int32)
    y = jax.random.randint(jax.random.PRNGKey(102), (2, 16), 0, 512, dtype=jnp.int32)

    logits_r1, _ = jax.jit(lambda p: forward_logits(p, x, cfg_r1))(params_r1)
    params_r8 = expand_checkpoint_1_to_8_loops(params_r1, jax.random.PRNGKey(103))
    logits_r8, _ = jax.jit(lambda p: forward_logits(p, x, cfg_r8))(params_r8)

    # In bfloat16, unrolled R=1 scan vs R=8 while-loop differs by at most 1 BF16 ULP (~0.0039)
    max_diff = float(jnp.max(jnp.abs(logits_r1 - logits_r8)))
    mean_diff = float(jnp.mean(jnp.abs(logits_r1 - logits_r8)))
    assert max_diff < 1e-2, f"BF16 step-0 logits drifted after R=1->8 expansion: {max_diff}"
    assert mean_diff < 1e-3, f"BF16 mean diff too large: {mean_diff}"

    # Verify exact mathematical preservation in float32
    cfg_r1_f32 = LoopMoEConfig(**{**cfg_r1.to_dict(), "dtype_name": "float32"})
    cfg_r8_f32 = LoopMoEConfig(**{**cfg_r8.to_dict(), "dtype_name": "float32"})
    p1_f32 = init_model_params(jax.random.PRNGKey(99), cfg_r1_f32)
    l1_f32, _ = jax.jit(lambda p: forward_logits(p, x, cfg_r1_f32))(p1_f32)
    p8_f32 = expand_checkpoint_1_to_8_loops(p1_f32, jax.random.PRNGKey(103))
    l8_f32, _ = jax.jit(lambda p: forward_logits(p, x, cfg_r8_f32))(p8_f32)
    assert float(jnp.max(jnp.abs(l1_f32 - l8_f32))) < 1e-5

    # Verify newly added loop gates t=1..7 immediately receive non-zero gradient
    _, grads = jax.jit(jax.value_and_grad(lambda p: forward_model(p, x, y, cfg_r8)[0]))(params_r8)
    alpha_grads_new = grads["iter_adaln"]["loop_alpha"][1:]
    assert float(jnp.max(jnp.abs(alpha_grads_new))) > 0.0


def test_muon_adamw_optimizer_per_expert_and_no_decay() -> None:
    # 1. Warmup-Stable NO-DECAY schedule check
    sched = make_warmup_stable_schedule(peak_lr=0.015, warmup_steps=100)
    np.testing.assert_allclose(float(sched(0)), 0.015 / 100.0, rtol=1e-5)
    assert float(sched(50)) > float(sched(0))
    np.testing.assert_allclose(float(sched(99)), 0.015, rtol=1e-6)
    np.testing.assert_allclose(float(sched(100)), 0.015, rtol=1e-6)
    np.testing.assert_allclose(float(sched(50_000)), 0.015, rtol=1e-6)
    np.testing.assert_allclose(float(sched(1_000_000)), 0.015, rtol=1e-6)

    # 2. Per-expert 3D vmap Newton-Schulz orthogonalization check
    g_3d = jax.random.normal(jax.random.PRNGKey(5), (4, 16, 32), dtype=jnp.float32)
    ortho_3d = per_expert_newton_schulz(g_3d, ns_steps=5)
    assert ortho_3d.shape == g_3d.shape
    for e in range(4):
        gram = ortho_3d[e] @ ortho_3d[e].T
        # Off-diagonal entries should be much smaller than diagonal after NS orthogonalization
        diag = jnp.diag(gram)
        off_diag = gram - jnp.diag(diag)
        assert float(jnp.mean(jnp.abs(off_diag))) < 0.25 * float(jnp.mean(diag))

    # 3. Full optimizer init + update + router_bias protection
    cfg = LoopMoEConfig.tiny(num_loops=8)
    params = init_model_params(jax.random.PRNGKey(6), cfg)
    params = expand_checkpoint_1_to_8_loops(params, jax.random.PRNGKey(7))
    opt_cfg = OptimConfig(warmup_steps=10)
    optimizer = create_muon_adamw_optimizer(opt_cfg)
    opt_state = optimizer.init(params)

    assert classify_param("moe/w_gate_up", params["moe"]["w_gate_up"]) == "muon"
    assert classify_param("mla/w_dq", params["mla"]["w_dq"]) == "muon"
    assert classify_param("embed_tokens", params["embed_tokens"]) == "adam_decay"
    assert classify_param("iter_adaln/loop_alpha", params["iter_adaln"]["loop_alpha"]) == "adam_no_decay"
    assert classify_param("moe/router_bias", params["moe"]["router_bias"]) == "frozen"

    x = jax.random.randint(jax.random.PRNGKey(8), (2, 16), 0, 512, dtype=jnp.int32)
    y = jax.random.randint(jax.random.PRNGKey(9), (2, 16), 0, 512, dtype=jnp.int32)

    @jax.jit
    def one_step(p, s):
        (_, aux), grads = jax.value_and_grad(lambda pp: forward_model(pp, x, y, cfg), has_aux=True)(p)
        updates, next_s = optimizer.update(grads, s, p)
        next_p = optax.apply_updates(p, updates)
        return updates, next_p, next_s, aux

    updates, new_params, new_opt_state, aux = one_step(params, opt_state)

    assert float(jnp.max(jnp.abs(updates["moe"]["router_bias"]))) == 0.0
    assert float(jnp.linalg.norm(updates["moe"]["w_gate_up"])) > 0.0
    assert float(jnp.linalg.norm(updates["mla"]["w_dq"])) > 0.0
    assert float(jnp.linalg.norm(updates["embed_tokens"])) > 0.0
    assert new_opt_state is not None

    # 4. Aux-Loss-Free bias update direction
    counts = jnp.zeros((8, cfg.n_routed_experts), dtype=jnp.float32).at[0, 0].set(100.0)
    updated_bias = update_router_biases(new_params["moe"]["router_bias"], counts, num_loops=8, update_rate=0.01)
    assert float(updated_bias[0, 0]) < 0.0
    assert float(updated_bias[0, 1]) > 0.0


def test_chunked_cross_entropy_equivalence() -> None:
    key = jax.random.PRNGKey(77)
    k_h, k_w, k_t, k_m = jax.random.split(key, 4)
    hidden = jax.random.normal(k_h, (2, 16, 64), dtype=jnp.bfloat16)
    lm_w = jax.random.normal(k_w, (64, 512), dtype=jnp.bfloat16) * 0.1
    targets = jax.random.randint(k_t, (2, 16), 0, 512, dtype=jnp.int32)
    mask = jax.random.bernoulli(k_m, p=0.8, shape=(2, 16)).astype(jnp.float32)

    def _dense_loss_fn(h, w):
        logits = (h @ w).astype(jnp.float32)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.sum(nll * mask) / jnp.sum(mask)

    dense_loss, (dense_dh, dense_dw) = jax.value_and_grad(_dense_loss_fn, argnums=(0, 1))(hidden, lm_w)
    lm_w_vd = lm_w.T

    for chunk_sz in (8, 16, 32):
        def _chunked_fn(h, w, csz=chunk_sz):
            return chunked_cross_entropy_loss(h, w, targets, chunk_size=csz, loss_mask=mask)

        chunked, (ch_dh, ch_dw) = jax.value_and_grad(_chunked_fn, argnums=(0, 1))(hidden, lm_w)
        np.testing.assert_allclose(float(chunked), float(dense_loss), atol=1e-4)
        np.testing.assert_allclose(np.asarray(ch_dh, dtype=np.float32), np.asarray(dense_dh, dtype=np.float32), atol=5e-3)
        np.testing.assert_allclose(np.asarray(ch_dw, dtype=np.float32), np.asarray(dense_dw, dtype=np.float32), atol=5e-3)

        def _chunked_vd_fn(h, w_vd, csz=chunk_sz):
            return chunked_cross_entropy_loss(
                h, w_vd, targets, chunk_size=csz, loss_mask=mask, weight_is_vd=True
            )

        chunked_vd, (ch_vd_dh, ch_vd_dw) = jax.value_and_grad(_chunked_vd_fn, argnums=(0, 1))(hidden, lm_w_vd)
        np.testing.assert_allclose(float(chunked_vd), float(dense_loss), atol=1e-4)
        np.testing.assert_allclose(np.asarray(ch_vd_dh, dtype=np.float32), np.asarray(dense_dh, dtype=np.float32), atol=5e-3)
        np.testing.assert_allclose(np.asarray(ch_vd_dw.T, dtype=np.float32), np.asarray(dense_dw, dtype=np.float32), atol=5e-3)

    # Also verify multi-chunk BF16-saved CE (_saved_logits_ce and _saved_logits_ce_vd) with num_chunks=2
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    sv_loss, (sv_dh, sv_dw) = jax.value_and_grad(
        lambda h, w: _saved_logits_ce(h, w, targets, mask, denom), argnums=(0, 1)
    )(hidden, lm_w)
    np.testing.assert_allclose(float(sv_loss), float(dense_loss), atol=1e-4)
    np.testing.assert_allclose(np.asarray(sv_dh, dtype=np.float32), np.asarray(dense_dh, dtype=np.float32), atol=5e-3)
    np.testing.assert_allclose(np.asarray(sv_dw, dtype=np.float32), np.asarray(dense_dw, dtype=np.float32), atol=5e-3)

    sv_vd_loss, (sv_vd_dh, sv_vd_dw) = jax.value_and_grad(
        lambda h, w_vd: _saved_logits_ce_vd(h, w_vd, targets, mask, denom), argnums=(0, 1)
    )(hidden, lm_w_vd)
    np.testing.assert_allclose(float(sv_vd_loss), float(dense_loss), atol=1e-4)
    np.testing.assert_allclose(np.asarray(sv_vd_dh, dtype=np.float32), np.asarray(dense_dh, dtype=np.float32), atol=5e-3)
    np.testing.assert_allclose(np.asarray(sv_vd_dw.T, dtype=np.float32), np.asarray(dense_dw, dtype=np.float32), atol=5e-3)


def test_checkpoint_roundtrip_and_fp4_kv(tmp_path: Path) -> None:
    cfg = LoopMoEConfig.tiny(num_loops=2)
    params = init_model_params(jax.random.PRNGKey(21), cfg)
    opt_cfg = OptimConfig(warmup_steps=10)
    opt = create_muon_adamw_optimizer(opt_cfg)
    opt_state = opt.init(params)

    save_checkpoint(
        ckpt_root=tmp_path,
        stage_id=3,
        step=25,
        tokens_seen=123_456,
        params=params,
        opt_state=opt_state,
        cfg=cfg,
        optim_cfg=opt_cfg,
        hf_repo_id=None,
    )

    loaded = load_latest_checkpoint(tmp_path, params_template=params, opt_state_template=opt_state)
    assert loaded is not None
    restored_params, restored_opt_state, meta = loaded
    assert meta["stage_id"] == 3
    assert meta["step"] == 25
    assert meta["tokens_seen"] == 123_456

    for a, b in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(restored_params),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
    assert restored_opt_state is not None

    kv = jax.random.normal(jax.random.PRNGKey(22), (2, 16, 4, 64), dtype=jnp.bfloat16)
    dequant, scales = quantize_kv_fp4_e2m1(kv)
    assert dequant.shape == kv.shape
    assert scales.shape == (2, 16, 4, 4)
    assert jnp.all(jnp.isfinite(dequant))


def test_local_bpe_tokenizer_bilingual_code_math_smoke(tmp_path: Path) -> None:
    tokenizer, trainer = build_bpe_tokenizer(vocab_size=512)
    corpus = [
        "The LoopMoE transformer uses 416 experts and 1234567890 digits cleanly.\n",
        "這是一個雙語預訓練模型，支援繁體中文、簡體中文與英文混合推論。\n",
        "def forward(x):\n    for i in range(8):\n        x = x + 1\n    return x\n",
        r"Solve $x^2 + 3x - 10 = 0$ for integers $x \in \{-5, 2\}$." + "\n",
    ] * 25
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    save_hf_tokenizer_files(tokenizer, tmp_path)

    assert (tmp_path / "tokenizer.json").exists()
    assert (tmp_path / "tokenizer_config.json").exists()
    report = benchmark_tokenizer(tokenizer)
    for domain, metrics in report.items():
        assert metrics["roundtrip_exact"] is True, f"Roundtrip failed on {domain}"

    # Verify special tokens & single-digit splitting
    assert tokenizer.token_to_id("<think>") == SPECIAL_TOKEN_IDS["<think>"]
    assert tokenizer.token_to_id("</think>") == SPECIAL_TOKEN_IDS["</think>"]
    digit_enc = tokenizer.encode("2026")
    assert len(digit_enc.tokens) == 4, f"Digits should split individually, got {digit_enc.tokens}"


def test_rms_norm_custom_vjp_and_mhc_merge_equivalence() -> None:
    def _ref_rms_norm(x_in: jax.Array, w_in: jax.Array | None = None, eps: float = 1e-6) -> jax.Array:
        x_f32 = x_in.astype(jnp.float32)
        inv = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
        x_hat = x_f32 * inv
        if w_in is not None:
            return (x_hat * w_in.astype(jnp.float32)).astype(x_in.dtype)
        return x_hat.astype(x_in.dtype)

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(303), 4)
    for dtype, tol in ((jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)):
        x = jax.random.normal(k1, (2, 8, 64), dtype=dtype)
        w = (1.0 + 0.1 * jax.random.normal(k2, (64,), dtype=dtype)).astype(dtype)
        cot = jax.random.normal(k3, (2, 8, 64), dtype=dtype)

        # Weighted RMSNorm forward + gradient match
        y_custom = rms_norm(x, w, eps=1e-6)
        y_ref = _ref_rms_norm(x, w, eps=1e-6)
        np.testing.assert_allclose(np.asarray(y_custom, dtype=np.float32), np.asarray(y_ref, dtype=np.float32), atol=tol, rtol=tol)

        dx_c, dw_c = jax.grad(lambda x_, w_: jnp.mean(rms_norm(x_, w_).astype(jnp.float32) * cot.astype(jnp.float32)), argnums=(0, 1))(x, w)
        dx_r, dw_r = jax.grad(lambda x_, w_: jnp.mean(_ref_rms_norm(x_, w_).astype(jnp.float32) * cot.astype(jnp.float32)), argnums=(0, 1))(x, w)
        np.testing.assert_allclose(np.asarray(dx_c, dtype=np.float32), np.asarray(dx_r, dtype=np.float32), atol=tol, rtol=tol)
        np.testing.assert_allclose(np.asarray(dw_c, dtype=np.float32), np.asarray(dw_r, dtype=np.float32), atol=tol, rtol=tol)

        # Unweighted RMSNorm forward + gradient match
        yu_custom = rms_norm(x, None, eps=1e-6)
        yu_ref = _ref_rms_norm(x, None, eps=1e-6)
        np.testing.assert_allclose(np.asarray(yu_custom, dtype=np.float32), np.asarray(yu_ref, dtype=np.float32), atol=tol, rtol=tol)

        dxu_c = jax.grad(lambda x_: jnp.mean(rms_norm(x_, None).astype(jnp.float32) * cot.astype(jnp.float32)))(x)
        dxu_r = jax.grad(lambda x_: jnp.mean(_ref_rms_norm(x_, None).astype(jnp.float32) * cot.astype(jnp.float32)))(x)
        np.testing.assert_allclose(np.asarray(dxu_c, dtype=np.float32), np.asarray(dxu_r, dtype=np.float32), atol=tol, rtol=tol)

    # mhc_post_merge equivalence (loop_gate=0.0 identity and loop_gate=0.37 blended)
    def _ref_mhc_merge(xs: jax.Array, sub_out: jax.Array, hp: jax.Array, hr: jax.Array, lg: jax.Array) -> jax.Array:
        dt = xs.dtype
        mixed = jnp.einsum("bsij,bsjd->bsid", hr.astype(dt), xs) + hp.astype(dt)[..., None] * sub_out.astype(dt)[:, :, None, :]
        g = jnp.asarray(lg, dtype=dt)
        return (jnp.asarray(1.0, dtype=dt) - g) * xs + g * mixed

    xs = jax.random.normal(k1, (2, 8, 4, 64), dtype=jnp.float32)
    sub = jax.random.normal(k2, (2, 8, 64), dtype=jnp.float32)
    hp = jax.random.uniform(k3, (2, 8, 4), minval=0.0, maxval=2.0, dtype=jnp.float32)
    hr_raw = jax.random.normal(k4, (2, 8, 4, 4), dtype=jnp.float32)
    hr = sinkhorn_knopp(hr_raw, num_iters=10)
    cot_m = jax.random.normal(k4, (2, 8, 4, 64), dtype=jnp.float32)

    # Exact identity when loop_gate == 0.0
    out_zero = mhc_post_merge(xs, sub, hp, hr, loop_gate=0.0)
    np.testing.assert_allclose(np.asarray(out_zero), np.asarray(xs), atol=1e-6)

    for dtype, tol in ((jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)):
        xs_d = xs.astype(dtype)
        sub_d = sub.astype(dtype)
        hp_d = hp.astype(dtype)
        hr_d = hr.astype(dtype)
        cot_d = cot_m.astype(jnp.float32)
        for gate_val in (0.0, 0.37):
            lg = jnp.asarray(gate_val, dtype=dtype)
            out_c = mhc_post_merge(xs_d, sub_d, hp_d, hr_d, loop_gate=lg)
            out_r = _ref_mhc_merge(xs_d, sub_d, hp_d, hr_d, lg)
            np.testing.assert_allclose(np.asarray(out_c, dtype=np.float32), np.asarray(out_r, dtype=np.float32), atol=tol, rtol=tol)

            grads_c = jax.grad(
                lambda a, b, c, d, g: jnp.mean(mhc_post_merge(a, b, c, d, loop_gate=g).astype(jnp.float32) * cot_d),
                argnums=(0, 1, 2, 3, 4),
            )(xs_d, sub_d, hp_d, hr_d, lg)
            grads_r = jax.grad(
                lambda a, b, c, d, g: jnp.mean(_ref_mhc_merge(a, b, c, d, g).astype(jnp.float32) * cot_d),
                argnums=(0, 1, 2, 3, 4),
            )(xs_d, sub_d, hp_d, hr_d, lg)
            for gc, gr in zip(grads_c, grads_r, strict=True):
                np.testing.assert_allclose(np.asarray(gc, dtype=np.float32), np.asarray(gr, dtype=np.float32), atol=tol, rtol=tol)


def test_mhc_compute_gates_and_moe_dispatch_combine_equivalence() -> None:
    def _ref_mhc_compute_gates(p, x_streams, sinkhorn_iters=10, eps=1e-6):
        bsz, seq_len, n_streams, d_model = x_streams.shape
        dt = x_streams.dtype
        x_flat = x_streams.reshape(bsz, seq_len, n_streams * d_model)
        x_norm = rms_norm(x_flat, p["norm_w"], eps=eps)
        all_dyn = (x_norm @ p["phi_all"]).astype(jnp.float32)
        pre_dyn = all_dyn[..., :n_streams]
        post_dyn = all_dyn[..., n_streams : 2 * n_streams]
        res_dyn = all_dyn[..., 2 * n_streams :].reshape(bsz, seq_len, n_streams, n_streams)
        h_pre = jax.nn.sigmoid(p["alpha_pre"] * pre_dyn + p["b_pre"])
        h_pre = (h_pre / (jnp.sum(h_pre, axis=-1, keepdims=True) + eps)).astype(dt)
        h_post = (2.0 * jax.nn.sigmoid(p["alpha_post"] * post_dyn + p["b_post"])).astype(dt)
        h_res = sinkhorn_knopp(p["alpha_res"] * res_dyn + p["b_res"], num_iters=sinkhorn_iters, eps=eps).astype(dt)
        h_in = jnp.sum(h_pre[:, :, :, None] * x_streams, axis=2)
        return h_in, h_pre, h_post, h_res

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(404), 4)
    for dtype, tol in ((jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)):
        p_gates = init_mhc_sublayer_params(k1, d_model=32, n_streams=4, dtype=dtype)
        # Non-trivial norm weights to ensure norm_w folding is exercised
        p_gates["norm_w"] = (1.0 + 0.1 * jax.random.normal(k2, p_gates["norm_w"].shape, dtype=dtype)).astype(dtype)
        xs = jax.random.normal(k3, (2, 6, 4, 32), dtype=dtype)
        c_in = jax.random.normal(k4, (2, 6, 32), dtype=jnp.float32)
        c_pre = jax.random.normal(k1, (2, 6, 4), dtype=jnp.float32)
        c_post = jax.random.normal(k2, (2, 6, 4), dtype=jnp.float32)
        c_res = jax.random.normal(k3, (2, 6, 4, 4), dtype=jnp.float32)

        outs_fast = mhc_compute_gates(p_gates, xs, sinkhorn_iters=10, eps=1e-6)
        outs_ref = _ref_mhc_compute_gates(p_gates, xs, sinkhorn_iters=10, eps=1e-6)
        for a, b in zip(outs_fast, outs_ref, strict=True):
            np.testing.assert_allclose(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), atol=tol, rtol=tol)

        def _loss_gates(fn, p, x_in):
            hi, hp, hpo, hr = fn(p, x_in, sinkhorn_iters=10, eps=1e-6)
            return (
                jnp.mean(hi.astype(jnp.float32) * c_in)
                + jnp.mean(hp.astype(jnp.float32) * c_pre)
                + jnp.mean(hpo.astype(jnp.float32) * c_post)
                + jnp.mean(hr.astype(jnp.float32) * c_res)
            )

        g_fast = jax.grad(lambda p, x_in: _loss_gates(mhc_compute_gates, p, x_in), argnums=(0, 1))(p_gates, xs)
        g_ref = jax.grad(lambda p, x_in: _loss_gates(_ref_mhc_compute_gates, p, x_in), argnums=(0, 1))(p_gates, xs)
        for lf, lr in zip(jax.tree_util.tree_leaves(g_fast), jax.tree_util.tree_leaves(g_ref), strict=True):
            np.testing.assert_allclose(np.asarray(lf, dtype=np.float32), np.asarray(lr, dtype=np.float32), atol=tol, rtol=tol)

    # Verify _moe_dispatch, _moe_combine, and searchsorted expert counting equivalence
    num_tokens, d_model, top_k, num_experts = 16, 32, 5, 12
    # Construct router selections leaving some experts empty and hitting boundary E-1
    logits_r = jax.random.normal(k1, (num_tokens, num_experts), dtype=jnp.float32)
    _, topk_idx = jax.lax.top_k(logits_r, top_k)
    topk_idx = topk_idx.at[0, 0].set(num_experts - 1)
    flat_experts = topk_idx.reshape(-1)
    sort_order = jnp.argsort(flat_experts)
    sorted_experts = flat_experts[sort_order]
    token_indices = sort_order // top_k
    inv_order = jnp.empty_like(sort_order).at[sort_order].set(
        jnp.arange(sort_order.shape[0], dtype=sort_order.dtype)
    )
    edges = jnp.arange(num_experts + 1, dtype=sorted_experts.dtype)
    counts_ss = jnp.diff(jnp.searchsorted(sorted_experts, edges, side="left"))
    counts_bc = jnp.bincount(flat_experts, length=num_experts)
    np.testing.assert_array_equal(np.asarray(counts_ss), np.asarray(counts_bc))

    cot_disp = jax.random.normal(k2, (num_tokens * top_k, d_model), dtype=jnp.float32)
    cot_comb = jax.random.normal(k3, (num_tokens, d_model), dtype=jnp.float32)
    for dtype, tol in ((jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)):
        x_flat = jax.random.normal(k4, (num_tokens, d_model), dtype=dtype)
        down_out = jax.random.normal(k1, (num_tokens * top_k, d_model), dtype=dtype)

        disp_fast = _moe_dispatch(x_flat, token_indices, inv_order, top_k)
        disp_ref = x_flat[token_indices]
        np.testing.assert_allclose(np.asarray(disp_fast, dtype=np.float32), np.asarray(disp_ref, dtype=np.float32), atol=tol)
        gx_fast = jax.grad(lambda x_: jnp.mean(_moe_dispatch(x_, token_indices, inv_order, top_k).astype(jnp.float32) * cot_disp))(x_flat)
        gx_ref = jax.grad(lambda x_: jnp.mean(x_[token_indices].astype(jnp.float32) * cot_disp))(x_flat)
        np.testing.assert_allclose(np.asarray(gx_fast, dtype=np.float32), np.asarray(gx_ref, dtype=np.float32), atol=tol, rtol=tol)

        comb_fast = _moe_combine(down_out, token_indices, inv_order, top_k)
        comb_ref = jnp.zeros((num_tokens, d_model), dtype=dtype).at[token_indices].add(down_out)
        np.testing.assert_allclose(np.asarray(comb_fast, dtype=np.float32), np.asarray(comb_ref, dtype=np.float32), atol=tol, rtol=tol)
        gd_fast = jax.grad(lambda d_: jnp.mean(_moe_combine(d_, token_indices, inv_order, top_k).astype(jnp.float32) * cot_comb))(down_out)
        gd_ref = jax.grad(lambda d_: jnp.mean(jnp.zeros((num_tokens, d_model), dtype=dtype).at[token_indices].add(d_).astype(jnp.float32) * cot_comb))(down_out)
        np.testing.assert_allclose(np.asarray(gd_fast, dtype=np.float32), np.asarray(gd_ref, dtype=np.float32), atol=tol, rtol=tol)

        # Verify _permute_1d forward & gradient equivalence
        w_1d = jax.random.normal(k2, (num_tokens * top_k,), dtype=dtype)
        cot_1d = jax.random.normal(k3, (num_tokens * top_k,), dtype=jnp.float32)
        p_fast = _permute_1d(w_1d, sort_order, inv_order)
        p_ref = w_1d[sort_order]
        np.testing.assert_allclose(np.asarray(p_fast, dtype=np.float32), np.asarray(p_ref, dtype=np.float32), atol=tol)
        gp_fast = jax.grad(lambda w_: jnp.mean(_permute_1d(w_, sort_order, inv_order).astype(jnp.float32) * cot_1d))(w_1d)
        gp_ref = jax.grad(lambda w_: jnp.mean(w_[sort_order].astype(jnp.float32) * cot_1d))(w_1d)
        np.testing.assert_allclose(np.asarray(gp_fast, dtype=np.float32), np.asarray(gp_ref, dtype=np.float32), atol=tol, rtol=tol)
