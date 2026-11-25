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
    estimate_step_flops,
    generate_prime_table_sizes,
    is_prime,
)
from loopmoe.engram import _causal_depthwise_conv1d
from loopmoe.mhc import (
    collapse_streams,
    init_mhc_sublayer_params,
    mhc_compute_gates,
    mhc_post_merge,
    rms_norm,
    sinkhorn_knopp,
)
from loopmoe.mla import (
    apply_mla,
    apply_rope,
    compute_yarn_freqs,
    init_mla_params,
    quantize_kv_fp4_e2m1,
)
from loopmoe.model import (
    _fused_chunked_ce,
    _fused_chunked_ce_vd,
    _saved_logits_ce,
    _saved_logits_ce_vd,
    chunked_cross_entropy_loss,
    chunked_cross_entropy_loss_and_nll,
    count_tree_params,
    expand_checkpoint_1_to_8_loops,
    forward_dynamic_exit,
    forward_logits,
    forward_model,
    init_model_params,
)
from loopmoe.moe import (
    _extract_topk_scores,
    _moe_combine,
    _moe_dispatch,
    _permute_1d,
    _sorted_weights_from_sort,
    update_router_biases,
)
from loopmoe.optim import (
    OptimConfig,
    classify_param,
    create_muon_adamw_optimizer,
    make_warmup_stable_schedule,
    per_expert_newton_schulz,
)
from loopmoe.pallas_moe import pallas_moe_combine, pallas_moe_gather
from train_tokenizer import benchmark_tokenizer, build_bpe_tokenizer, save_hf_tokenizer_files


def test_full_1b_analytic_params_and_tiny_exact_match() -> None:
    cfg = LoopMoEConfig()
    stats = count_analytic_params(cfg)
    assert 1.45e9 <= stats["total"] <= 1.55e9, f"Expected ~1.49B params, got {stats['total']}"
    assert 0.38 <= stats["engram_sparse_ratio"] <= 0.42, stats["engram_sparse_ratio"]
    assert all(is_prime(p) for p in cfg.engram_table_sizes)
    assert len(set(cfg.engram_table_sizes)) == len(cfg.engram_table_sizes)
    assert cfg.engram_total_slots == 2_097_152
    assert cfg.engram_slot_alignment == 1024
    assert cfg.d_engram == 1024
    assert sum(cfg.engram_table_sizes) == 2_097_152
    assert sum(cfg.engram_table_sizes) % 1024 == 0
    aligned_2m = generate_prime_table_sizes(2_097_152, 4, align_to=1024)
    assert sum(aligned_2m) == 2_097_152 and sum(aligned_2m) % 1024 == 0
    assert all(is_prime(p) for p in aligned_2m) and len(set(aligned_2m)) == 4
    assert cfg.vocab_size == 32_768
    assert cfg.mhc_streams == 2
    assert cfg.n_routed_experts == 128
    assert cfg.d_ff_expert == 1024
    assert cfg.d_ff_shared == 1024
    assert cfg.top_k == 4
    assert cfg.seq_aux_loss_weight == 0.0
    assert cfg.engram_pre_loop_only is True
    assert cfg.untie_mla_per_loop is True
    assert cfg.mla_weight_banks == 8
    assert cfg.max_loops == 32
    assert cfg.reuse_kv_across_loops is True
    assert cfg.use_dynamic_loop is True
    assert cfg.halt_balance_weight == 0.01
    assert cfg.dynamic_loop_ce_sample_depths == 2
    assert cfg.train_loop_capacity_factor == 0.5
    assert cfg.bptt_window == 8
    assert cfg.untie_router_per_loop is True
    assert cfg.router_weight_banks == 8
    assert cfg.use_post_moe_rmsnorm is True
    assert cfg.embed_reinject_lambda == 0.5

    tied_cfg = LoopMoEConfig(untie_mla_per_loop=False)
    assert estimate_step_flops(cfg, 2, 4096) == estimate_step_flops(tied_cfg, 2, 4096)
    assert estimate_step_flops(LoopMoEConfig(num_loops=8, reuse_kv_across_loops=True), 2, 4096) < (
        estimate_step_flops(LoopMoEConfig(num_loops=8, reuse_kv_across_loops=False), 2, 4096)
    )
    assert estimate_step_flops(LoopMoEConfig(num_loops=32, train_loop_capacity_factor=0.5), 2, 4096) < (
        estimate_step_flops(LoopMoEConfig(num_loops=32, train_loop_capacity_factor=1.0), 2, 4096)
    )

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

    # Verify analytic count matches actual initialized pytree leaf count across cyclic banks & variants
    tiny_cfg = LoopMoEConfig.tiny()
    tiny_params = init_model_params(jax.random.PRNGKey(0), tiny_cfg)
    assert count_analytic_params(tiny_cfg)["total"] == count_tree_params(tiny_params)

    tiny_4s = LoopMoEConfig(**{**tiny_cfg.to_dict(), "mhc_streams": 4, "engram_pre_loop_only": False})
    assert count_analytic_params(tiny_4s)["total"] == count_tree_params(
        init_model_params(jax.random.PRNGKey(1), tiny_4s)
    )

    tiny_tied = LoopMoEConfig(
        **{
            **tiny_cfg.to_dict(),
            "untie_mla_per_loop": False,
            "untie_router_per_loop": False,
            "use_post_moe_rmsnorm": False,
        }
    )
    assert count_analytic_params(tiny_tied)["total"] == count_tree_params(
        init_model_params(jax.random.PRNGKey(2), tiny_tied)
    )

    for extra in (
        {"mla_weight_banks": 4, "router_weight_banks": 4},
        {"reuse_kv_across_loops": False},
        {"use_dynamic_loop": False},
        {"use_csa2_indexer": True},
        {"use_csa2_indexer": True, "reuse_kv_across_loops": False},
    ):
        cfg_var = LoopMoEConfig(**{**tiny_cfg.to_dict(), **extra})
        assert count_analytic_params(cfg_var)["total"] == count_tree_params(
            init_model_params(jax.random.PRNGKey(3), cfg_var)
        ), f"Param mismatch for {extra}"


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
        assert jnp.isfinite(aux["router_mmr"]) and float(aux["router_mmr"]) >= 1.0
        assert aux["expert_counts"].shape == (num_loops, cfg.n_routed_experts)
        assert float(jnp.sum(aux["expert_counts"][0])) == float(2 * 32 * cfg.top_k)

        # Every active gradient must be finite; router_bias gradient must be zero (Aux-Loss-Free)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))
        assert float(jnp.max(jnp.abs(grads["moe"]["router_bias"]))) == 0.0
        assert float(jnp.linalg.norm(grads["mla"]["w_dq"])) > 0.0
        assert float(jnp.linalg.norm(grads["moe"]["w_gate_up"])) > 0.0
        assert float(jnp.linalg.norm(grads["moe"]["post_norm_w"])) > 0.0
        assert float(jnp.linalg.norm(grads["moe"]["router_w"])) > 0.0
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
    mhc_attn["b_res"] = jax.random.normal(jax.random.PRNGKey(100), mhc_attn["b_res"].shape, dtype=jnp.float32)
    params_r1["mhc"] = {**params_r1["mhc"], "attn": mhc_attn}
    params_r1["moe"] = {
        **params_r1["moe"],
        "router_bias": params_r1["moe"]["router_bias"].at[0].set(0.25),
    }

    x = jax.random.randint(jax.random.PRNGKey(101), (2, 16), 0, 512, dtype=jnp.int32)
    y = jax.random.randint(jax.random.PRNGKey(102), (2, 16), 0, 512, dtype=jnp.int32)

    logits_r1, _ = jax.jit(lambda p: forward_logits(p, x, cfg_r1))(params_r1)
    params_r8 = expand_checkpoint_1_to_8_loops(params_r1, jax.random.PRNGKey(103))
    assert params_r8["mla"]["w_dq"].shape[0] == 8
    assert params_r8["mla"]["w_dkv"].ndim == 2
    np.testing.assert_array_equal(
        np.asarray(params_r8["mla"]["w_dkv"], dtype=np.float32),
        np.asarray(params_r1["mla"]["w_dkv"], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(params_r8["mla"]["w_dq"][0], dtype=np.float32),
        np.asarray(params_r8["mla"]["w_dq"][7], dtype=np.float32),
    )
    logits_r8, _ = jax.jit(lambda p: forward_logits(p, x, cfg_r8))(params_r8)
    logits_r8_dyn, dyn_info = jax.jit(lambda p: forward_dynamic_exit(p, x, cfg_r8))(params_r8)

    # In bfloat16, unrolled R=1 scan vs R=8 while-loop differs by at most 1 BF16 ULP (~0.0039)
    max_diff = float(jnp.max(jnp.abs(logits_r1 - logits_r8)))
    mean_diff = float(jnp.mean(jnp.abs(logits_r1 - logits_r8)))
    assert max_diff < 1e-2, f"BF16 step-0 logits drifted after R=1->8 expansion: {max_diff}"
    assert mean_diff < 1e-3, f"BF16 mean diff too large: {mean_diff}"
    assert float(jnp.max(jnp.abs(logits_r1 - logits_r8_dyn))) < 1e-2
    assert 1.0 <= float(dyn_info["mean_loops"]) <= 8.0

    # Verify exact mathematical preservation in float32
    cfg_r1_f32 = LoopMoEConfig(**{**cfg_r1.to_dict(), "dtype_name": "float32"})
    cfg_r8_f32 = LoopMoEConfig(**{**cfg_r8.to_dict(), "dtype_name": "float32"})
    p1_f32 = init_model_params(jax.random.PRNGKey(99), cfg_r1_f32)
    l1_f32, _ = jax.jit(lambda p: forward_logits(p, x, cfg_r1_f32))(p1_f32)
    p8_f32 = expand_checkpoint_1_to_8_loops(p1_f32, jax.random.PRNGKey(103))
    l8_f32, _ = jax.jit(lambda p: forward_logits(p, x, cfg_r8_f32))(p8_f32)
    l8_dyn_f32, _ = jax.jit(lambda p: forward_dynamic_exit(p, x, cfg_r8_f32))(p8_f32)
    assert float(jnp.max(jnp.abs(l1_f32 - l8_f32))) < 1e-5
    assert float(jnp.max(jnp.abs(l1_f32 - l8_dyn_f32))) < 1e-5

    # Verify newly added loop gates t=1..7 & halt_gate immediately receive non-zero gradient
    _, grads = jax.jit(jax.value_and_grad(lambda p: forward_model(p, x, y, cfg_r8)[0]))(params_r8)
    alpha_grads_new = grads["iter_adaln"]["loop_alpha"][1:]
    assert float(jnp.max(jnp.abs(alpha_grads_new))) > 0.0
    assert float(jnp.max(jnp.abs(alpha_grads_new[:, 0]))) > 0.0
    assert float(jnp.linalg.norm(grads["halt_gate"]["b"][:-1])) > 0.0


def test_muon_adamw_optimizer_per_expert_and_no_decay() -> None:
    # 1. Warmup-Stable NO-DECAY schedule check
    sched = make_warmup_stable_schedule(peak_lr=0.015, warmup_steps=100)
    np.testing.assert_allclose(float(sched(0)), 0.015 / 100.0, rtol=1e-5)
    assert float(sched(50)) > float(sched(0))
    np.testing.assert_allclose(float(sched(99)), 0.015, rtol=1e-6)
    np.testing.assert_allclose(float(sched(100)), 0.015, rtol=1e-6)
    np.testing.assert_allclose(float(sched(50_000)), 0.015, rtol=1e-6)
    np.testing.assert_allclose(float(sched(1_000_000)), 0.015, rtol=1e-6)

    # 2. Per-expert 3D vmap Newton-Schulz orthogonalization check (Cubic5 default + Quintic)
    g_3d = jax.random.normal(jax.random.PRNGKey(5), (4, 16, 32), dtype=jnp.float32)
    for mode in ("cubic5", "quintic"):
        ortho_3d = per_expert_newton_schulz(g_3d, ns_steps=5, mode=mode)
        assert ortho_3d.shape == g_3d.shape
        for e in range(4):
            gram = ortho_3d[e] @ ortho_3d[e].T
            # Off-diagonal entries should be much smaller than diagonal after NS orthogonalization
            diag = jnp.diag(gram)
            off_diag = gram - jnp.diag(diag)
            assert float(jnp.mean(jnp.abs(off_diag))) < 0.25 * float(jnp.mean(diag))
            if mode == "cubic5":
                sv = np.linalg.svd(np.asarray(ortho_3d[e]), compute_uv=False)
                assert float(sv.min()) >= 0.70 and float(sv.max()) <= 1.35, f"Cubic5 SV out of band: {sv}"

    g_tall = jax.random.normal(jax.random.PRNGKey(15), (4, 32, 16), dtype=jnp.float32)
    ortho_tall = per_expert_newton_schulz(g_tall, ns_steps=5, mode="cubic5")
    assert ortho_tall.shape == g_tall.shape
    for e in range(4):
        gram_t = ortho_tall[e].T @ ortho_tall[e]
        diag_t = jnp.diag(gram_t)
        off_t = gram_t - jnp.diag(diag_t)
        assert float(jnp.mean(jnp.abs(off_t))) < 0.25 * float(jnp.mean(diag_t))
        sv_t = np.linalg.svd(np.asarray(ortho_tall[e]), compute_uv=False)
        assert float(sv_t.min()) >= 0.70 and float(sv_t.max()) <= 1.35, f"Cubic5 tall SV out of band: {sv_t}"

    # 3. Full optimizer init + update + router_bias protection (Cubic5 default + Quintic fallback)
    cfg = LoopMoEConfig.tiny(num_loops=8)
    params = init_model_params(jax.random.PRNGKey(6), cfg)
    params = expand_checkpoint_1_to_8_loops(params, jax.random.PRNGKey(7))
    x = jax.random.randint(jax.random.PRNGKey(8), (2, 16), 0, 512, dtype=jnp.int32)
    y = jax.random.randint(jax.random.PRNGKey(9), (2, 16), 0, 512, dtype=jnp.int32)

    assert classify_param("moe/w_gate_up", params["moe"]["w_gate_up"]) == "muon"
    assert classify_param("mla/w_dq", params["mla"]["w_dq"]) == "muon"
    assert classify_param("mla/w_dkv", params["mla"]["w_dkv"]) == "muon"
    assert classify_param("engram/tables", params["engram"]["tables"]) == "engram_row_adam"
    assert classify_param("mla/q_norm_w", params["mla"]["q_norm_w"]) == "adam_no_decay"
    assert classify_param("mla/kv_norm_w", params["mla"]["kv_norm_w"]) == "adam_no_decay"
    assert classify_param("halt_gate/w", params["halt_gate"]["w"]) == "adam_decay"
    assert classify_param("halt_gate/b", params["halt_gate"]["b"]) == "adam_no_decay"
    assert classify_param("moe/router_w", params["moe"]["router_w"]) == "adam_decay"
    assert classify_param("moe/post_norm_w", params["moe"]["post_norm_w"]) == "adam_no_decay"
    assert classify_param("embed_tokens", params["embed_tokens"]) == "adam_decay"
    assert classify_param("iter_adaln/loop_alpha", params["iter_adaln"]["loop_alpha"]) == "adam_no_decay"
    assert classify_param("moe/router_bias", params["moe"]["router_bias"]) == "frozen"

    for mode, steps in (("cubic5", 5), ("quintic", 3)):
        opt_cfg = OptimConfig(warmup_steps=10, muon_ns_mode=mode, muon_ns_steps=steps)
        optimizer = create_muon_adamw_optimizer(opt_cfg)
        opt_state = optimizer.init(params)

        # Verify Row-Wise Factored Adam state on engram/tables (BF16 mu + 1 FP32 scalar per slot)
        row_adam_st = opt_state[1].inner_states["engram_row_adam"].inner_state[0]
        assert row_adam_st.mu["engram"]["tables"].shape == params["engram"]["tables"].shape
        assert row_adam_st.mu["engram"]["tables"].dtype == jnp.bfloat16
        assert row_adam_st.v_row["engram"]["tables"].shape == (
            params["engram"]["tables"].shape[0],
            1,
        )
        assert row_adam_st.v_row["engram"]["tables"].dtype == jnp.float32

        @jax.jit
        def one_step(p, s, opt=optimizer):
            (_, aux), grads = jax.value_and_grad(lambda pp: forward_model(pp, x, y, cfg), has_aux=True)(p)
            updates, next_s = opt.update(grads, s, p)
            next_p = optax.apply_updates(p, updates)
            return updates, next_p, next_s, aux

        updates, new_params, new_opt_state, aux = one_step(params, opt_state)

        assert float(jnp.max(jnp.abs(updates["moe"]["router_bias"]))) == 0.0
        assert float(jnp.linalg.norm(updates["moe"]["w_gate_up"])) > 0.0
        assert float(jnp.linalg.norm(updates["moe"]["w_down"])) > 0.0
        assert float(jnp.linalg.norm(updates["mla"]["w_dq"])) > 0.0
        assert float(jnp.linalg.norm(updates["embed_tokens"])) > 0.0
        assert float(jnp.linalg.norm(updates["engram"]["tables"])) > 0.0
        if mode == "cubic5":
            assert updates["moe"]["w_gate_up"].dtype == jnp.bfloat16
            assert updates["mla"]["w_dq"].dtype == jnp.bfloat16
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

    # Verify single-pass fused CE (_fused_chunked_ce and _fused_chunked_ce_vd) including non-unit cotangent scaling
    scale = 2.5
    h_chunks = hidden.reshape(2, 16, 64)
    t_chunks = targets.reshape(2, 16)
    m_chunks = mask.reshape(2, 16)
    fu_loss, (fu_dh, fu_dw) = jax.value_and_grad(
        lambda h, w: scale * _fused_chunked_ce(h, w, t_chunks, m_chunks, denom), argnums=(0, 1)
    )(h_chunks, lm_w)
    np.testing.assert_allclose(float(fu_loss), scale * float(dense_loss), atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(fu_dh.reshape(2, 16, 64), dtype=np.float32),
        scale * np.asarray(dense_dh, dtype=np.float32),
        atol=1.5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(fu_dw, dtype=np.float32),
        scale * np.asarray(dense_dw, dtype=np.float32),
        atol=1.5e-2,
    )

    fu_vd_loss, (fu_vd_dh, fu_vd_dw) = jax.value_and_grad(
        lambda h, w_vd: scale * _fused_chunked_ce_vd(h, w_vd, t_chunks, m_chunks, denom), argnums=(0, 1)
    )(h_chunks, lm_w_vd)
    np.testing.assert_allclose(float(fu_vd_loss), scale * float(dense_loss), atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(fu_vd_dh.reshape(2, 16, 64), dtype=np.float32),
        scale * np.asarray(dense_dh, dtype=np.float32),
        atol=1.5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(fu_vd_dw.T, dtype=np.float32),
        scale * np.asarray(dense_dw, dtype=np.float32),
        atol=1.5e-2,
    )


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

        # Verify _permute_1d and _sorted_weights_from_sort forward & gradient equivalence
        w_1d = jax.random.normal(k2, (num_tokens * top_k,), dtype=dtype)
        cot_1d = jax.random.normal(k3, (num_tokens * top_k,), dtype=jnp.float32)
        p_fast = _permute_1d(w_1d, sort_order, inv_order)
        p_ref = w_1d[sort_order]
        np.testing.assert_allclose(np.asarray(p_fast, dtype=np.float32), np.asarray(p_ref, dtype=np.float32), atol=tol)
        gp_fast = jax.grad(lambda w_: jnp.mean(_permute_1d(w_, sort_order, inv_order).astype(jnp.float32) * cot_1d))(w_1d)
        gp_ref = jax.grad(lambda w_: jnp.mean(w_[sort_order].astype(jnp.float32) * cot_1d))(w_1d)
        np.testing.assert_allclose(np.asarray(gp_fast, dtype=np.float32), np.asarray(gp_ref, dtype=np.float32), atol=tol, rtol=tol)

        _, inv_order_sort = jax.lax.sort(
            (sort_order, jnp.arange(sort_order.shape[0], dtype=sort_order.dtype)), num_keys=1
        )
        np.testing.assert_array_equal(np.asarray(inv_order_sort), np.asarray(inv_order))

        _, _, sw_f32 = jax.lax.sort(
            (flat_experts, jnp.arange(flat_experts.shape[0], dtype=jnp.int32), w_1d.astype(jnp.float32)),
            num_keys=1,
        )
        sw_fast = _sorted_weights_from_sort(w_1d, sw_f32, sort_order)
        np.testing.assert_allclose(np.asarray(sw_fast, dtype=np.float32), np.asarray(p_ref, dtype=np.float32), atol=tol)
        gsw_fast = jax.grad(lambda w_: jnp.mean(_sorted_weights_from_sort(w_, sw_f32, sort_order).astype(jnp.float32) * cot_1d))(w_1d)
        np.testing.assert_allclose(np.asarray(gsw_fast, dtype=np.float32), np.asarray(gp_ref, dtype=np.float32), atol=tol, rtol=tol)

        scores_r = jax.nn.sigmoid(logits_r.astype(dtype))
        cot_tk = jax.random.normal(k4, (num_tokens, top_k), dtype=jnp.float32)
        tk_fast = _extract_topk_scores(scores_r, topk_idx, num_experts)
        tk_ref = jnp.take_along_axis(scores_r, topk_idx, axis=-1)
        np.testing.assert_allclose(np.asarray(tk_fast, dtype=np.float32), np.asarray(tk_ref, dtype=np.float32), atol=tol)
        tk_fast_grad = jax.grad(
            lambda sc: jnp.mean(_extract_topk_scores(sc, topk_idx, num_experts).astype(jnp.float32) * cot_tk)
        )(scores_r)
        tk_ref_grad = jax.grad(
            lambda sc: jnp.mean(jnp.take_along_axis(sc, topk_idx, axis=-1).astype(jnp.float32) * cot_tk)
        )(scores_r)
        np.testing.assert_allclose(
            np.asarray(tk_fast_grad, dtype=np.float32), np.asarray(tk_ref_grad, dtype=np.float32), atol=tol, rtol=tol
        )

    # Verify apply_mla flat-head RoPE equivalence against reference apply_rope in float32
    cfg_mla = LoopMoEConfig(**{**LoopMoEConfig.tiny(num_loops=1).to_dict(), "dtype_name": "float32", "untie_mla_per_loop": False})
    mla_p = init_mla_params(k1, cfg_mla, dtype=jnp.float32)
    x_mla = jax.random.normal(k2, (2, 16, cfg_mla.d_model), dtype=jnp.float32)
    out_fast, _ = apply_mla(mla_p, x_mla, cfg_mla)

    # Reference path using apply_rope directly
    import math as _math
    c_q_ref = rms_norm(x_mla @ mla_p["w_dq"], mla_p["q_norm_w"], eps=cfg_mla.rms_norm_eps)
    q_all_ref = (c_q_ref @ mla_p["w_uq"]).reshape(2, 16, cfg_mla.n_heads, cfg_mla.d_qk_head)
    q_nope_ref, q_rope_ref = jnp.split(q_all_ref, [cfg_mla.d_qk_nope], axis=-1)
    c_kv_raw_ref, k_rope_shared_ref = jnp.split(x_mla @ mla_p["w_dkv"], [cfg_mla.d_kv_lora], axis=-1)
    c_kv_ref = rms_norm(c_kv_raw_ref, mla_p["kv_norm_w"], eps=cfg_mla.rms_norm_eps)
    k_nope_ref = (c_kv_ref @ mla_p["w_uk"]).reshape(2, 16, cfg_mla.n_heads, cfg_mla.d_qk_nope)
    v_ref = (c_kv_ref @ mla_p["w_uv"]).reshape(2, 16, cfg_mla.n_heads, cfg_mla.d_v)
    inv_f, ms = compute_yarn_freqs(
        d_rope=cfg_mla.d_qk_rope,
        theta=cfg_mla.rope_theta,
        base_seq_len=cfg_mla.base_seq_len,
        scale=cfg_mla.yarn_scale,
        beta_fast=cfg_mla.yarn_beta_fast,
        beta_slow=cfg_mla.yarn_beta_slow,
    )
    q_ref = jnp.concatenate([q_nope_ref, apply_rope(q_rope_ref, inv_f)], axis=-1)
    k_r_ref = jnp.broadcast_to(apply_rope(k_rope_shared_ref[:, :, None, :], inv_f), (2, 16, cfg_mla.n_heads, cfg_mla.d_qk_rope))
    k_ref = jnp.concatenate([k_nope_ref, k_r_ref], axis=-1)
    v_pad = jnp.pad(v_ref, [(0, 0), (0, 0), (0, 0), (0, max(0, cfg_mla.d_qk_head - cfg_mla.d_v))])
    attn_ref = jax.nn.dot_product_attention(
        q_ref, k_ref, v_pad, is_causal=True, scale=(ms * ms) / _math.sqrt(float(cfg_mla.d_qk_head))
    )[..., : cfg_mla.d_v]
    out_ref = attn_ref.reshape(2, 16, cfg_mla.n_heads * cfg_mla.d_v) @ mla_p["w_o"]
    np.testing.assert_allclose(np.asarray(out_fast), np.asarray(out_ref), atol=1e-5, rtol=1e-5)


def test_2stream_closed_form_mhc_and_engram_conv() -> None:
    def _ref_mhc2_gates(p, x_streams, eps=1e-6):
        dt = x_streams.dtype
        bsz, seq_len, _, d_model = x_streams.shape
        x_flat = x_streams.reshape(bsz, seq_len, 2 * d_model)
        x_norm = rms_norm(x_flat, p["norm_w"], eps=eps)
        all_dyn = (x_norm @ p["phi_all"]).astype(jnp.float32)
        h_pre = jax.nn.sigmoid(p["alpha_pre"] * all_dyn[..., :2] + p["b_pre"])
        h_pre = (h_pre / (jnp.sum(h_pre, axis=-1, keepdims=True) + eps)).astype(dt)
        h_post = (2.0 * jax.nn.sigmoid(p["alpha_post"] * all_dyn[..., 2:4] + p["b_post"])).astype(dt)
        pv = jax.nn.sigmoid(p["alpha_res"] * all_dyn[..., 4:5] + p["b_res"]).astype(dt)
        qv = jnp.asarray(1.0, dtype=dt) - pv
        h_res = jnp.stack(
            [jnp.concatenate([pv, qv], axis=-1), jnp.concatenate([qv, pv], axis=-1)],
            axis=-2,
        )
        h_in = jnp.sum(h_pre[:, :, :, None] * x_streams, axis=2)
        return h_in, h_pre, h_post, h_res

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(505), 4)
    for dtype, tol in ((jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)):
        p2 = init_mhc_sublayer_params(k1, d_model=32, n_streams=2, dtype=dtype)
        p2["norm_w"] = (1.0 + 0.1 * jax.random.normal(k2, p2["norm_w"].shape, dtype=dtype)).astype(dtype)
        p2["alpha_res"] = jnp.asarray(0.4, dtype=jnp.float32)
        p2["b_res"] = jnp.asarray([0.3], dtype=jnp.float32)
        xs = jax.random.normal(k3, (2, 6, 2, 32), dtype=dtype)
        assert collapse_streams(xs).dtype == dtype

        hi, hp, hpo, hr = mhc_compute_gates(p2, xs, sinkhorn_iters=10, eps=1e-6)
        assert hr.shape == (2, 6, 2, 2)
        np.testing.assert_allclose(np.asarray(jnp.sum(hr, axis=-1), dtype=np.float32), 1.0, atol=tol)
        np.testing.assert_allclose(np.asarray(jnp.sum(hr, axis=-2), dtype=np.float32), 1.0, atol=tol)

        outs_ref = _ref_mhc2_gates(p2, xs, eps=1e-6)
        for a, b in zip((hi, hp, hpo, hr), outs_ref, strict=True):
            np.testing.assert_allclose(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), atol=tol, rtol=tol)

        sub = jax.random.normal(k4, (2, 6, 32), dtype=dtype)
        out_zero = mhc_post_merge(xs, sub, hpo, hr, loop_gate=0.0)
        np.testing.assert_allclose(np.asarray(out_zero, dtype=np.float32), np.asarray(xs, dtype=np.float32), atol=1e-6)

        cot = jax.random.normal(k2, (2, 6, 2, 32), dtype=jnp.float32)
        lg = jnp.asarray(0.42, dtype=dtype)
        g_fast = jax.grad(
            lambda x_in, s_in, hp_in, hr_in: jnp.mean(
                mhc_post_merge(x_in, s_in, hp_in, hr_in, loop_gate=lg).astype(jnp.float32) * cot
            ),
            argnums=(0, 1, 2, 3),
        )(xs, sub, hpo, hr)
        g_ref = jax.grad(
            lambda x_in, s_in, hp_in, hr_in: jnp.mean(
                (
                    (1.0 - lg) * x_in
                    + lg * (jnp.einsum("bsij,bsjd->bsid", hr_in, x_in) + hp_in[..., None] * s_in[:, :, None, :])
                ).astype(jnp.float32)
                * cot
            ),
            argnums=(0, 1, 2, 3),
        )(xs, sub, hpo, hr)
        for gf, gr in zip(g_fast, g_ref, strict=True):
            np.testing.assert_allclose(np.asarray(gf, dtype=np.float32), np.asarray(gr, dtype=np.float32), atol=tol, rtol=tol)

    # Verify causal depthwise conv1d equivalence vs explicit dilated loop (including seq_len < pad_len)
    def _ref_causal_conv(x_in, w_in, b_in, dilation):
        k_sz = w_in.shape[0]
        pad_len = (k_sz - 1) * dilation
        s_len = x_in.shape[1]
        padded = jnp.pad(x_in, ((0, 0), (pad_len, 0), (0, 0)))
        acc = jnp.zeros_like(x_in)
        for k in range(k_sz):
            start = pad_len - k * dilation
            acc = acc + padded[:, start : start + s_len, :] * w_in[k][None, None, :].astype(x_in.dtype)
        return jax.nn.silu(acc + b_in[None, None, :].astype(x_in.dtype))

    for seq_len, dilation in ((16, 3), (5, 3), (8, 1)):
        xc = jax.random.normal(k1, (2, seq_len, 16), dtype=jnp.bfloat16)
        wc = jax.random.normal(k2, (4, 16), dtype=jnp.bfloat16)
        bc = jax.random.normal(k3, (16,), dtype=jnp.bfloat16)
        yc = _causal_depthwise_conv1d(xc, wc, bc, dilation=dilation)
        yr = _ref_causal_conv(xc, wc, bc, dilation=dilation)
        np.testing.assert_allclose(np.asarray(yc, dtype=np.float32), np.asarray(yr, dtype=np.float32), atol=2e-2, rtol=2e-2)


def test_async_checkpoint_write_and_inloop_engram(tmp_path: Path) -> None:
    cfg = LoopMoEConfig(**{**LoopMoEConfig.tiny(num_loops=2).to_dict(), "engram_pre_loop_only": False})
    params = init_model_params(jax.random.PRNGKey(31), cfg)
    opt_cfg = OptimConfig(warmup_steps=10)
    opt = create_muon_adamw_optimizer(opt_cfg)
    opt_state = opt.init(params)

    x = jax.random.randint(jax.random.PRNGKey(32), (2, 16), 0, 512, dtype=jnp.int32)
    y = jax.random.randint(jax.random.PRNGKey(33), (2, 16), 0, 512, dtype=jnp.int32)
    (loss, _), grads = jax.jit(jax.value_and_grad(lambda p: forward_model(p, x, y, cfg), has_aux=True))(params)
    assert jnp.isfinite(loss)
    assert float(jnp.linalg.norm(grads["engram"]["w_kv"])) > 0.0

    ckpt_dir, write_future = save_checkpoint(
        ckpt_root=tmp_path,
        stage_id=1,
        step=5,
        tokens_seen=500,
        params=params,
        opt_state=opt_state,
        cfg=cfg,
        optim_cfg=opt_cfg,
        async_write=True,
    )
    assert write_future is not None
    write_future.result()
    loaded = load_latest_checkpoint(tmp_path, params_template=params, opt_state_template=opt_state)
    assert loaded is not None and loaded[2]["step"] == 5 and ckpt_dir.exists()


def test_pallas_moe_dispatch_combine_kernels() -> None:
    import jax.experimental.pallas as pl
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    # 1. Exact bit-for-bit numerical equivalence via Pallas CPU interpret mode
    n_tok, top_k, d_model = 128, 4, 1024
    m_rows = n_tok * top_k
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(909), 3)
    idx = jax.random.randint(k1, (m_rows,), 0, n_tok, dtype=jnp.int32)
    perm = jax.random.permutation(k2, m_rows).astype(jnp.int32)

    for dtype in (jnp.bfloat16, jnp.float32):
        x = jax.random.normal(k3, (n_tok, d_model), dtype=dtype)
        g_out = pallas_moe_gather(x, idx, interpret=True)
        np.testing.assert_array_equal(np.asarray(g_out, dtype=np.float32), np.asarray(x[idx], dtype=np.float32))

        down = jax.random.normal(k1, (m_rows, d_model), dtype=dtype)
        c_out = pallas_moe_combine(down, perm, top_k=top_k, interpret=True)
        inv_2d = perm.reshape(n_tok, top_k)
        c_ref = down[inv_2d[:, 0]]
        for k in range(1, top_k):
            c_ref = c_ref + down[inv_2d[:, k]]
        np.testing.assert_array_equal(np.asarray(c_out, dtype=np.float32), np.asarray(c_ref, dtype=np.float32))

    # 2. Mosaic TPU v5e & v6e MLIR lowering at full 1B shapes (32768 x 2048, K=4) with zero shape_cast
    x_s = jax.ShapeDtypeStruct((32_768, 2048), jnp.bfloat16)
    idx_s = jax.ShapeDtypeStruct((131_072,), jnp.int32)
    down_s = jax.ShapeDtypeStruct((131_072, 2048), jnp.bfloat16)
    for dev in ("TPU v5 lite", "TPU v6 lite"):
        mesh = AbstractMesh((1,), ("x",), abstract_device=AbstractDevice(dev, num_cores=1, platform="tpu"))
        with use_abstract_mesh(mesh):
            mlir_g = pl.lower_as_mlir(pallas_moe_gather, x_s, idx_s, platforms=("tpu",))
            mlir_c = pl.lower_as_mlir(lambda a, b: pallas_moe_combine(a, b, top_k=4), down_s, idx_s, platforms=("tpu",))
            assert "shape_cast" not in mlir_g and "shape_cast" not in mlir_c
            assert "pallas_moe_gather" in mlir_g and "pallas_moe_combine" in mlir_c


def test_glm54_kv_reuse_and_dynamic_loop_early_exit() -> None:
    import math as _math

    # 1. Verify GLM-5.4 Loop-0 KV & CSA2 sparse-mask reuse skips KV & indexer projections cleanly
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1201), 3)
    for use_csa2 in (False, True):
        cfg_mla = LoopMoEConfig(
            **{
                **LoopMoEConfig.tiny(num_loops=2, use_csa2_indexer=use_csa2).to_dict(),
                "dtype_name": "float32",
                "csa2_topk": 4 if use_csa2 else 16,
            }
        )
        mla_p = init_mla_params(k1, cfg_mla, dtype=jnp.float32)
        x0 = jax.random.normal(k2, (2, 16, cfg_mla.d_model), dtype=jnp.float32)
        x1 = jax.random.normal(k3, (2, 16, cfg_mla.d_model), dtype=jnp.float32)

        _, _, kv0 = apply_mla(mla_p, x0, cfg_mla, loop_idx=0, return_kv=True)
        # Zero out ALL KV & CSA2 weights to prove cached_kv bypasses KV projections & indexer completely
        mla_p_nokv = dict(mla_p)
        for kv_k in ("w_dkv", "w_uk", "w_uv", "idx_w_q", "idx_w_k", "idx_w_score"):
            if kv_k in mla_p_nokv:
                mla_p_nokv[kv_k] = jnp.zeros_like(mla_p_nokv[kv_k])

        out_cached, _ = apply_mla(mla_p_nokv, x1, cfg_mla, loop_idx=1, cached_kv=kv0, return_kv=False)

        # Reference: loop-1 Q/O weights attending directly to cached kv0
        cfg_noshare = LoopMoEConfig(**{**cfg_mla.to_dict(), "reuse_kv_across_loops": False})
        assert cfg_noshare.reuse_kv_across_loops is False
        out_full, _ = apply_mla(mla_p, x1, cfg_mla, loop_idx=1, cached_kv=kv0, return_kv=False)
        np.testing.assert_allclose(np.asarray(out_cached), np.asarray(out_full), atol=1e-6, rtol=1e-6)

    # 2. Verify chunked_cross_entropy_loss_and_nll matches dense per-token NLL and weighted gradients
    kh, kw, kt, km = jax.random.split(jax.random.PRNGKey(1202), 4)
    hidden = jax.random.normal(kh, (2, 16, 64), dtype=jnp.float32)
    lm_w = jax.random.normal(kw, (64, 512), dtype=jnp.float32) * 0.1
    targets = jax.random.randint(kt, (2, 16), 0, 512, dtype=jnp.int32)
    weights = jax.random.uniform(km, (2, 16), minval=0.0, maxval=1.0, dtype=jnp.float32)
    denom_custom = jnp.asarray(12.5, dtype=jnp.float32)

    def _dense_weighted(h, w):
        logits = h @ w
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.sum(nll * weights) / denom_custom, jax.lax.stop_gradient(nll)

    (d_loss, d_nll), (d_dh, d_dw) = jax.value_and_grad(
        lambda h, w: _dense_weighted(h, w), argnums=(0, 1), has_aux=True
    )(hidden, lm_w)

    for weight_is_vd, w_mat in ((False, lm_w), (True, lm_w.T)):
        (c_loss, c_nll), (c_dh, c_dw) = jax.value_and_grad(
            lambda h, w, vd=weight_is_vd: chunked_cross_entropy_loss_and_nll(
                h, w, targets, chunk_size=8, loss_mask=weights, weight_is_vd=vd, denom_override=denom_custom
            ),
            argnums=(0, 1),
            has_aux=True,
        )(hidden, w_mat)
        np.testing.assert_allclose(float(c_loss), float(d_loss), atol=1e-5)
        np.testing.assert_allclose(np.asarray(c_nll), np.asarray(d_nll), atol=1e-5)
        np.testing.assert_allclose(np.asarray(c_dh), np.asarray(d_dh), atol=1e-4)
        dw_cmp = c_dw.T if weight_is_vd else c_dw
        np.testing.assert_allclose(np.asarray(dw_cmp), np.asarray(d_dw), atol=1e-4)

    # 3. Verify Eq. 6 dynamic-loop forward+backward gradients (subsample=1 & subsample=2, token-drop & rng_key)
    x_tok = jax.random.randint(jax.random.PRNGKey(1203), (2, 32), 0, 512, dtype=jnp.int32)
    y_tok = jax.random.randint(jax.random.PRNGKey(1204), (2, 32), 0, 512, dtype=jnp.int32)
    for sub in (1, 2):
        cfg_dyn = LoopMoEConfig(
            **{
                **LoopMoEConfig.tiny(num_loops=4).to_dict(),
                "dynamic_loop_ce_subsample": sub,
                "remat_per_loop": True,
            }
        )
        p_dyn = expand_checkpoint_1_to_8_loops(init_model_params(jax.random.PRNGKey(1205), cfg_dyn), jax.random.PRNGKey(1206))
        p_dyn["iter_adaln"] = {
            **p_dyn["iter_adaln"],
            "loop_alpha": jnp.full_like(p_dyn["iter_adaln"]["loop_alpha"], 0.5),
        }
        (loss, aux), grads = jax.jit(
            jax.value_and_grad(
                lambda p: forward_model(p, x_tok, y_tok, cfg_dyn, rng_key=jax.random.PRNGKey(1299)),
                has_aux=True,
            )
        )(p_dyn)
        assert jnp.isfinite(loss)
        assert 1.0 <= float(aux["expected_loops"]) <= 4.0
        assert 0.0 <= float(aux["halt_entropy"]) <= _math.log(4.0) + 1e-4
        # Loop 0 routes 100% (64 tokens), Loops >= 1 route Top-C (32 tokens) when capacity_factor=0.5
        assert float(jnp.sum(aux["expert_counts"][0])) == float(2 * 32 * cfg_dyn.top_k)
        assert float(jnp.sum(aux["expert_counts"][1])) == float(32 * cfg_dyn.top_k)
        assert float(jnp.linalg.norm(grads["halt_gate"]["w"][:4])) > 0.0
        assert float(jnp.linalg.norm(grads["halt_gate"]["b"][:3])) > 0.0
        assert float(jnp.linalg.norm(grads["mla"]["w_dq"][:4])) > 0.0
        assert float(jnp.linalg.norm(grads["mla"]["w_dkv"])) > 0.0
        assert float(jnp.linalg.norm(grads["moe"]["w_gate_up"])) > 0.0

    # 3b. Verify R=32 with cyclic 8-bank MLA/Router, sampled CE (ce_k=2), token-drop, & Truncated BPTT (window=4)
    cfg_r1_base = LoopMoEConfig.tiny(num_loops=1)
    p_r1_base = init_model_params(jax.random.PRNGKey(1210), cfg_r1_base)
    cfg_r32 = LoopMoEConfig(
        **{
            **cfg_r1_base.to_dict(),
            "num_loops": 32,
            "bptt_window": 4,
            "dynamic_loop_ce_sample_depths": 2,
            "train_loop_capacity_factor": 0.5,
            "remat_per_loop": True,
        }
    )
    p_r32 = expand_checkpoint_1_to_8_loops(p_r1_base, jax.random.PRNGKey(1211))
    l1_ref, _ = jax.jit(lambda p: forward_logits(p, x_tok, cfg_r1_base))(p_r1_base)
    l32_ref, _ = jax.jit(lambda p: forward_logits(p, x_tok, cfg_r32))(p_r32)
    assert float(jnp.max(jnp.abs(l1_ref - l32_ref))) < 1e-2

    p_r32["iter_adaln"] = {
        **p_r32["iter_adaln"],
        "loop_alpha": jnp.full_like(p_r32["iter_adaln"]["loop_alpha"], 0.25),
    }
    (loss_32, aux_32), grads_32 = jax.jit(
        jax.value_and_grad(
            lambda p: forward_model(p, x_tok, y_tok, cfg_r32, rng_key=jax.random.PRNGKey(1212)),
            has_aux=True,
        )
    )(p_r32)
    assert jnp.isfinite(loss_32)
    assert aux_32["expert_counts"].shape == (32, cfg_r32.n_routed_experts)
    assert float(jnp.sum(aux_32["expert_counts"][0])) == float(64 * cfg_r32.top_k)
    assert float(jnp.sum(aux_32["expert_counts"][31])) == float(32 * cfg_r32.top_k)
    assert float(jnp.linalg.norm(grads_32["mla"]["w_dq"])) > 0.0
    assert float(jnp.linalg.norm(grads_32["mla"]["w_dkv"])) > 0.0
    assert float(jnp.linalg.norm(grads_32["moe"]["w_gate_up"])) > 0.0
    assert float(jnp.linalg.norm(grads_32["halt_gate"]["b"][:5])) > 0.0
    assert float(jnp.linalg.norm(grads_32["halt_gate"]["b"][24:31])) > 0.0

    # 4. Verify forward_dynamic_exit truly performs early exit based on halt_gate
    cfg_exit = LoopMoEConfig.tiny(num_loops=4)
    p_exit = expand_checkpoint_1_to_8_loops(init_model_params(jax.random.PRNGKey(1207), cfg_exit), jax.random.PRNGKey(1208))
    p_exit["iter_adaln"] = {
        **p_exit["iter_adaln"],
        "loop_alpha": jnp.full_like(p_exit["iter_adaln"]["loop_alpha"], 0.5),
    }
    dyn_jit = jax.jit(lambda p: forward_dynamic_exit(p, x_tok, cfg_exit))

    p_early0 = {
        **p_exit,
        "halt_gate": {**p_exit["halt_gate"], "b": jnp.full((cfg_exit.max_loops,), 10.0, dtype=jnp.float32)},
    }
    _, info0 = dyn_jit(p_early0)
    assert int(info0["loops_executed"]) == 1
    np.testing.assert_allclose(float(info0["mean_loops"]), 1.0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(info0["exit_depth"]), np.zeros((2, 32), dtype=np.int32))

    p_early1 = {
        **p_exit,
        "halt_gate": {
            **p_exit["halt_gate"],
            "b": jnp.full((cfg_exit.max_loops,), -10.0, dtype=jnp.float32).at[1].set(10.0),
        },
    }
    _, info1 = dyn_jit(p_early1)
    assert int(info1["loops_executed"]) == 2
    np.testing.assert_allclose(float(info1["mean_loops"]), 2.0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(info1["exit_depth"]), np.ones((2, 32), dtype=np.int32))

    p_full = {
        **p_exit,
        "halt_gate": {**p_exit["halt_gate"], "b": jnp.full((cfg_exit.max_loops,), -10.0, dtype=jnp.float32)},
    }
    logits_dyn_full, info_full = dyn_jit(p_full)
    logits_static_full, _ = jax.jit(lambda p: forward_logits(p, x_tok, cfg_exit))(p_full)
    assert int(info_full["loops_executed"]) == 4
    np.testing.assert_allclose(float(info_full["mean_loops"]), 4.0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(logits_dyn_full, dtype=np.float32),
        np.asarray(logits_static_full, dtype=np.float32),
        atol=1e-2,
    )
