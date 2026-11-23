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
from loopmoe.mhc import sinkhorn_knopp
from loopmoe.mla import quantize_kv_fp4_e2m1
from loopmoe.model import (
    chunked_cross_entropy_loss,
    count_tree_params,
    expand_checkpoint_1_to_8_loops,
    forward_logits,
    forward_model,
    init_model_params,
)
from loopmoe.moe import update_router_biases
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
    assert 0.95e9 <= stats["total"] <= 1.05e9, f"Expected ~1B params, got {stats['total']}"
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
        assert float(jnp.linalg.norm(grads["moe"]["w_gate"])) > 0.0
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

    assert classify_param("moe/w_gate", params["moe"]["w_gate"]) == "muon"
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
    assert float(jnp.linalg.norm(updates["moe"]["w_gate"])) > 0.0
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

    dense_logits = (hidden @ lm_w).astype(jnp.float32)
    dense_nll = optax.softmax_cross_entropy_with_integer_labels(dense_logits, targets)
    dense_loss = jnp.sum(dense_nll * mask) / jnp.sum(mask)

    for chunk_sz in (8, 16, 32):
        chunked = chunked_cross_entropy_loss(
            hidden, lm_w, targets, chunk_size=chunk_sz, loss_mask=mask
        )
        np.testing.assert_allclose(float(chunked), float(dense_loss), atol=1e-4)


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
