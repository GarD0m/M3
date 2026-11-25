"""1B 1-Layer DeepSeek-V4.1-Flash LoopMoE model, chunked cross-entropy, and R=1->N expansion."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import optax

from .config import LoopMoEConfig, count_analytic_params
from .engram import (
    apply_engram_step,
    init_engram_params,
    precompute_engram_kv,
    prefetch_engram_embeddings,
)
from .iter_adaln import (
    apply_adaln,
    compute_iter_adaln_modulations,
    init_iter_adaln_params,
)
from .mhc import (
    collapse_streams,
    expand_streams,
    init_mhc_params,
    mhc_compute_gates,
    mhc_post_merge,
    rms_norm,
)
from .mla import apply_mla, init_mla_params, is_per_loop_mla_leaf
from .moe import apply_moe, init_moe_params


def _get_dtype(dtype_name: str) -> jnp.dtype:
    return getattr(jnp, dtype_name)


def init_model_params(key: jax.Array, cfg: LoopMoEConfig) -> dict[str, Any]:
    """Initialize the complete 1B 1-Layer DeepSeek-V4.1-Flash LoopMoE parameter pytree."""
    dtype = _get_dtype(cfg.dtype_name)
    keys = jax.random.split(key, 8)

    params: dict[str, Any] = {
        "embed_tokens": jax.random.normal(keys[0], (cfg.vocab_size, cfg.d_model), dtype=dtype) * 0.02,
        "final_norm_w": jnp.ones((cfg.d_model,), dtype=dtype),
        "iter_adaln": init_iter_adaln_params(keys[1], cfg, dtype=dtype),
        "mla": init_mla_params(keys[2], cfg, dtype=dtype),
        "moe": init_moe_params(keys[3], cfg, dtype=dtype),
    }
    if not cfg.tie_word_embeddings:
        params["lm_head"] = jax.random.normal(keys[4], (cfg.d_model, cfg.vocab_size), dtype=dtype) * (
            1.0 / jnp.sqrt(float(cfg.d_model))
        )
    if cfg.use_mhc:
        params["mhc"] = init_mhc_params(keys[5], cfg, dtype=dtype)
    if cfg.use_engram:
        params["engram"] = init_engram_params(keys[6], cfg, dtype=dtype)
    if cfg.use_dynamic_loop:
        params["halt_gate"] = {
            "w": jax.random.normal(keys[7], (cfg.max_loops, cfg.d_model), dtype=dtype)
            * (0.01 / jnp.sqrt(float(cfg.d_model))),
            "b": jnp.zeros((cfg.max_loops,), dtype=jnp.float32),
        }
    return params


def count_tree_params(tree: Any) -> int:
    """Count total scalar elements in a JAX pytree."""
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(tree))


def _compute_halt_distribution(halt_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute arXiv:2611.05724 geometric halting distribution q_p(r) from logits [R, B, S]."""
    num_loops = halt_logits.shape[0]
    bsz, seq_len = halt_logits.shape[1], halt_logits.shape[2]
    survival = jnp.ones((bsz, seq_len), dtype=jnp.float32)
    q_list: list[jax.Array] = []
    lam_list: list[jax.Array] = []
    for r in range(num_loops - 1):
        lam_r = jax.nn.sigmoid(halt_logits[r].astype(jnp.float32))
        q_r = survival * lam_r
        survival = survival * (1.0 - lam_r)
        q_list.append(q_r)
        lam_list.append(lam_r)
    q_list.append(survival)
    lam_list.append(jnp.ones((bsz, seq_len), dtype=jnp.float32))
    return jnp.stack(q_list, axis=0), jnp.stack(lam_list, axis=0)


def _single_loop_step(
    params: dict[str, Any],
    x_streams: jax.Array,
    x_embed: jax.Array,
    engram_kv: tuple[jax.Array, jax.Array] | None,
    loop_idx: jax.Array | int,
    cfg: LoopMoEConfig,
    cached_kv: tuple[jax.Array, jax.Array, jax.Array | None] | None = None,
    return_kv: bool = False,
    keep_scores: jax.Array | None = None,
    moe_capacity_factor: float = 1.0,
) -> tuple[
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array | None] | None,
]:
    """Execute one recurrent iteration of LOOM re-injection + IterAdaLN + mHC(MLA) + mHC(MoE)."""
    with jax.named_scope("loop_step"):
        alphas = params["iter_adaln"]["loop_alpha"][loop_idx]  # [3]
        dt = x_streams.dtype
        alpha_engram = alphas[0].astype(dt)
        alpha_attn = alphas[1].astype(dt)
        alpha_moe = alphas[2].astype(dt)

        if cfg.embed_reinject_lambda > 0.0 and cfg.num_loops > 1:
            # LOOM embedding re-injection for t >= 1 (1-indexed t >= 2), gated by alpha_engram so
            # R=1 -> R=N expansion preserves exact step-0 logits when loop_alpha[1:, 0] == 0.
            g_t = jnp.asarray(cfg.embed_reinject_lambda, dtype=dt) / jnp.asarray(loop_idx + 1, dtype=dt)
            eff_g = jnp.where(
                jnp.asarray(loop_idx, dtype=jnp.int32) > 0,
                g_t * alpha_engram,
                jnp.asarray(0.0, dtype=dt),
            )
            x_streams = (jnp.asarray(1.0, dtype=dt) - eff_g) * x_streams + eff_g * x_embed[:, :, None, :]

        # 1. Context-aware Engram conditional memory refinement (skipped inside loop when pre-loop-only)
        if cfg.use_engram and not cfg.engram_pre_loop_only:
            def _engram_sublayer(p, xs, ekv, ae):
                with jax.named_scope("engram_step"):
                    h_summary = collapse_streams(xs)
                    engram_delta, gate_mean = apply_engram_step(p["engram"], h_summary, ekv, cfg)
                    return xs + ae * engram_delta[:, :, None, :], gate_mean

            engram_fn = jax.checkpoint(_engram_sublayer) if cfg.remat_moe else _engram_sublayer
            x_streams, engram_gate_mean = engram_fn(params, x_streams, engram_kv, alpha_engram)
        else:
            engram_gate_mean = jnp.asarray(0.0, dtype=jnp.float32)

        # 2. Pure Per-Loop IterAdaLN modulation conditioned on loop_idx (zero [B, S, 12288] HBM)
        with jax.named_scope("iter_adaln"):
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = compute_iter_adaln_modulations(
                params["iter_adaln"], loop_idx, dtype=dt
            )

        # 3. MLA sublayer inside Single-Pass mHC (optionally emitting/reusing Loop-0 KV & CSA2 mask)
        with jax.named_scope("mhc_attn_sublayer"):
            if cfg.use_mhc:
                h_in_a, _, h_post_a, h_res_a = mhc_compute_gates(
                    params["mhc"]["attn"], x_streams, cfg.mhc_sinkhorn_iters, cfg.rms_norm_eps
                )
            else:
                h_in_a = x_streams[:, :, 0, :]

            h_in_a_norm = apply_adaln(
                h_in_a, params["iter_adaln"]["norm_attn_w"], scale_a, shift_a, cfg.rms_norm_eps
            )
            if return_kv:
                attn_out, _, kv_out = apply_mla(
                    params["mla"], h_in_a_norm, cfg, loop_idx=loop_idx, cached_kv=cached_kv, return_kv=True
                )
            else:
                attn_out, _ = apply_mla(
                    params["mla"], h_in_a_norm, cfg, loop_idx=loop_idx, cached_kv=cached_kv, return_kv=False
                )
                kv_out = None
            attn_out = attn_out * gate_a.astype(dt)

            if cfg.use_mhc:
                x_streams = mhc_post_merge(x_streams, attn_out, h_post_a, h_res_a, loop_gate=alpha_attn)
            else:
                x_streams = x_streams + alpha_attn * attn_out[:, :, None, :]

        # 4. DeepSeek MoE sublayer inside Single-Pass mHC (optional Top-C token-drop & remat)
        def _moe_sublayer(p, xs, sm, scm, gm, am, lidx, ks):
            with jax.named_scope("mhc_moe_sublayer"):
                if cfg.use_mhc:
                    h_in_m, _, h_post_m, h_res_m = mhc_compute_gates(
                        p["mhc"]["moe"], xs, cfg.mhc_sinkhorn_iters, cfg.rms_norm_eps
                    )
                else:
                    h_in_m = xs[:, :, 0, :]

                h_in_m_norm = apply_adaln(
                    h_in_m, p["iter_adaln"]["norm_moe_w"], scm, sm, cfg.rms_norm_eps
                )
                moe_out, ec, aux_l, mmr = apply_moe(
                    p["moe"],
                    h_in_m_norm,
                    lidx,
                    cfg,
                    keep_scores=ks,
                    capacity_factor=moe_capacity_factor,
                )
                moe_out = moe_out * gm.astype(dt)

                if cfg.use_mhc:
                    next_xs = mhc_post_merge(xs, moe_out, h_post_m, h_res_m, loop_gate=am)
                else:
                    next_xs = xs + am * moe_out[:, :, None, :]
                return next_xs, ec, aux_l, mmr

        moe_fn = jax.checkpoint(_moe_sublayer) if cfg.remat_moe else _moe_sublayer
        x_streams, expert_counts, seq_aux_loss, router_mmr = moe_fn(
            params, x_streams, shift_m, scale_m, gate_m, alpha_moe, loop_idx, keep_scores
        )

        return x_streams, (expert_counts, seq_aux_loss, engram_gate_mean, router_mmr), kv_out


def _embed_and_engram(
    params: dict[str, Any],
    input_ids: jax.Array,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array] | None, jax.Array]:
    """Compute token embeddings, single pre-loop Engram conditional memory, and mHC streams."""
    x = params["embed_tokens"][input_ids]  # [B, S, D]
    pre_gate_mean = jnp.asarray(0.0, dtype=jnp.float32)
    engram_kv: tuple[jax.Array, jax.Array] | None = None
    if cfg.use_engram:
        with jax.named_scope("engram_prefetch"):
            engram_e = prefetch_engram_embeddings(params["engram"], input_ids, cfg)
            assert engram_e is not None
        if cfg.engram_pre_loop_only:
            def _pre_engram_apply(p_eng, x_in, eng_e, a0):
                with jax.named_scope("engram_step"):
                    ekv = precompute_engram_kv(p_eng, eng_e, cfg)
                    engram_delta, p_gm = apply_engram_step(p_eng, x_in, ekv, cfg)
                    return x_in + a0.astype(x_in.dtype) * engram_delta, p_gm

            pre_fn = jax.checkpoint(_pre_engram_apply) if cfg.remat_moe else _pre_engram_apply
            x, pre_gate_mean = pre_fn(
                params["engram"], x, engram_e, params["iter_adaln"]["loop_alpha"][0, 0]
            )
            engram_kv = None
        else:
            with jax.named_scope("engram_prefetch"):
                engram_kv = precompute_engram_kv(params["engram"], engram_e, cfg)

    x_embed = x
    stream_init = params["mhc"]["stream_init"] if cfg.use_mhc else None
    x_streams = expand_streams(x, stream_init)
    return x_streams, x_embed, engram_kv, pre_gate_mean


def _select_ce_sample_depths(
    num_loops: int,
    k_sample: int,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """Select `k_sample` distinct deep loop indices in `[1, num_loops - 1]` sorted ascending."""
    if rng_key is not None:
        perm = jax.random.permutation(rng_key, num_loops - 1)[:k_sample] + 1
        return jnp.sort(perm).astype(jnp.int32)
    if k_sample <= 1:
        return jnp.asarray([num_loops - 1], dtype=jnp.int32)
    step = float(num_loops - 2) / float(k_sample - 1)
    idxs = [1 + int(round(i * step)) for i in range(k_sample)]
    idxs[-1] = num_loops - 1
    return jnp.asarray(idxs, dtype=jnp.int32)


def forward_hidden(
    params: dict[str, Any],
    input_ids: jax.Array,
    cfg: LoopMoEConfig,
    rng_key: jax.Array | None = None,
    sample_depths: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    """Run embedding + Engram + dynamic LoopMoE (KV reuse, MoE token-drop, truncated BPTT, sampled CE)."""
    x_streams, x_embed, engram_kv, pre_gate_mean = _embed_and_engram(params, input_ids, cfg)

    # Always run Loop 0 statically (100% tokens) so Loop-0 KV & CSA2 sparse mask stay loop-invariant
    emit_kv = cfg.reuse_kv_across_loops and (cfg.num_loops > 1)
    x_streams_0, (ec0, aux0, gm0, mmr0), cached_kv_0 = _single_loop_step(
        params,
        x_streams,
        x_embed,
        engram_kv,
        0,
        cfg,
        cached_kv=None,
        return_kv=emit_kv,
        keep_scores=None,
        moe_capacity_factor=1.0,
    )

    if cfg.num_loops == 1:
        hidden = rms_norm(collapse_streams(x_streams_0), params["final_norm_w"], eps=cfg.rms_norm_eps)
        engram_gate_mean = (
            pre_gate_mean if (cfg.use_engram and cfg.engram_pre_loop_only) else gm0
        )
        aux: dict[str, Any] = {
            "expert_counts": ec0[None, :],
            "seq_aux_loss": aux0,
            "engram_gate_mean": engram_gate_mean,
            "router_mmr": mmr0,
            "expected_loops": jnp.asarray(1.0, dtype=jnp.float32),
            "halt_entropy": jnp.asarray(0.0, dtype=jnp.float32),
        }
        return hidden, aux

    loop_kv = cached_kv_0 if cfg.reuse_kv_across_loops else None
    bsz, seq_len = input_ids.shape
    h0 = rms_norm(collapse_streams(x_streams_0), params["final_norm_w"], eps=cfg.rms_norm_eps)
    use_token_drop = (
        cfg.use_dynamic_loop
        and (0.0 < float(cfg.train_loop_capacity_factor) < 1.0)
        and ((bsz * seq_len) > 1)
    )
    use_sampled = cfg.use_dynamic_loop and (sample_depths is not None)

    if cfg.use_dynamic_loop:
        w_halt = params["halt_gate"]["w"][: cfg.num_loops]
        b_halt = params["halt_gate"]["b"][: cfg.num_loops]
        hl_0 = (
            jnp.einsum("bsd,d->bs", h0, w_halt[0].astype(h0.dtype)).astype(jnp.float32) + b_halt[0]
        )
        surv_0 = 1.0 - jax.nn.sigmoid(hl_0)
    else:
        w_halt = None
        b_halt = None
        hl_0 = jnp.zeros((bsz, seq_len), dtype=jnp.float32)
        surv_0 = jnp.ones((bsz, seq_len), dtype=jnp.float32)

    cutoff = (
        (cfg.num_loops - cfg.bptt_window)
        if (cfg.bptt_window > 0 and cfg.num_loops > (cfg.bptt_window + 1))
        else 1
    )

    def _keep_scores_for_step(surv_c: jax.Array, loop_idx: jax.Array) -> tuple[jax.Array | None, float]:
        if not use_token_drop:
            return None, 1.0
        ks = jax.lax.stop_gradient(surv_c)
        if rng_key is not None and cfg.train_loop_explore_noise > 0.0:
            step_key = jax.random.fold_in(rng_key, loop_idx)
            ks = ks + jax.random.uniform(
                step_key,
                ks.shape,
                dtype=jnp.float32,
                minval=0.0,
                maxval=float(cfg.train_loop_explore_noise),
            )
        return ks, float(cfg.train_loop_capacity_factor)

    prefix_outputs = None
    prefix_buf = jnp.zeros((0,), dtype=jnp.float32)
    xs_carry = x_streams_0
    surv_carry = surv_0

    if cutoff > 1:
        init_prefix_buf = (
            jnp.zeros((int(sample_depths.shape[0]), bsz, seq_len, h0.shape[-1]), dtype=h0.dtype)
            if use_sampled
            else jnp.zeros((0,), dtype=jnp.float32)
        )

        def _prefix_body(carry, loop_idx: jax.Array):
            xs_c, surv_c, buf_c = carry
            ks, moe_cap = _keep_scores_for_step(surv_c, loop_idx)
            next_xs, step_metrics, _ = _single_loop_step(
                params,
                xs_c,
                x_embed,
                engram_kv,
                loop_idx,
                cfg,
                cached_kv=loop_kv,
                return_kv=False,
                keep_scores=ks,
                moe_capacity_factor=moe_cap,
            )
            if cfg.use_dynamic_loop:
                assert w_halt is not None and b_halt is not None
                h_t = rms_norm(
                    collapse_streams(next_xs), params["final_norm_w"], eps=cfg.rms_norm_eps
                )
                dot_t = jnp.einsum(
                    "bsd,d->bs", h_t, w_halt[loop_idx].astype(h_t.dtype)
                ).astype(jnp.float32)
                hl_t = dot_t + b_halt[loop_idx]
                next_surv = surv_c * (1.0 - jax.nn.sigmoid(hl_t))
                if use_sampled:
                    assert sample_depths is not None
                    match = (sample_depths == loop_idx)[:, None, None, None]
                    next_buf = jnp.where(match, h_t[None, ...], buf_c)
                    h_emit = jnp.zeros((0,), dtype=jnp.float32)
                else:
                    next_buf = buf_c
                    h_emit = h_t
                return (next_xs, next_surv, next_buf), (step_metrics, dot_t, h_emit)
            h_empty = jnp.zeros((0,), dtype=jnp.float32)
            return (next_xs, surv_c, buf_c), (step_metrics, h_empty, h_empty)

        # Truncated BPTT boundary: stop_gradient over the ENTIRE prefix scan so XLA discards
        # 100% of prefix backward compute & VJP residuals (fixes R=32 21.4G HLO temporaries OOM).
        (xs_carry, surv_carry, prefix_buf), prefix_raw = jax.lax.stop_gradient(
            jax.lax.scan(
                _prefix_body,
                (xs_carry, surv_carry, init_prefix_buf),
                jnp.arange(1, cutoff, dtype=jnp.int32),
            )
        )
        ec_p, aux_p, gm_p, mmr_p = prefix_raw[0]
        dot_p, h_p = prefix_raw[1], prefix_raw[2]
        if cfg.use_dynamic_loop:
            assert w_halt is not None and b_halt is not None
            hl_p = dot_p + b_halt[1:cutoff, None, None]
            if use_sampled:
                assert sample_depths is not None
                for j in range(int(sample_depths.shape[0])):
                    d_j = sample_depths[j]
                    is_pref = d_j < cutoff
                    p_idx = jnp.clip(d_j - 1, 0, cutoff - 2)
                    live_dot = jnp.einsum(
                        "bsd,d->bs", prefix_buf[j], w_halt[d_j].astype(prefix_buf.dtype)
                    ).astype(jnp.float32)
                    hl_p = jnp.where(
                        is_pref, hl_p.at[p_idx].set(live_dot + b_halt[d_j]), hl_p
                    )
        else:
            hl_p = dot_p
        prefix_outputs = ((ec_p, aux_p, gm_p, mmr_p), hl_p, h_p)

    def _suffix_body(carry, loop_idx: jax.Array):
        xs_c, surv_c = carry
        ks, moe_cap = _keep_scores_for_step(surv_c, loop_idx)
        next_xs, step_metrics, _ = _single_loop_step(
            params,
            xs_c,
            x_embed,
            engram_kv,
            loop_idx,
            cfg,
            cached_kv=loop_kv,
            return_kv=False,
            keep_scores=ks,
            moe_capacity_factor=moe_cap,
        )
        if cfg.use_dynamic_loop:
            assert w_halt is not None and b_halt is not None
            h_t = rms_norm(
                collapse_streams(next_xs), params["final_norm_w"], eps=cfg.rms_norm_eps
            )
            hl_t = (
                jnp.einsum(
                    "bsd,d->bs", h_t, w_halt[loop_idx].astype(h_t.dtype)
                ).astype(jnp.float32)
                + b_halt[loop_idx]
            )
            next_surv = surv_c * (1.0 - jax.nn.sigmoid(hl_t))
            return (next_xs, next_surv), (step_metrics, hl_t, h_t)
        h_empty = jnp.zeros((0,), dtype=jnp.float32)
        return (next_xs, surv_c), (step_metrics, h_empty, h_empty)

    suffix_fn = jax.checkpoint(_suffix_body) if cfg.remat_per_loop else _suffix_body
    (x_streams_last, _), suffix_outputs = jax.lax.scan(
        suffix_fn, (xs_carry, surv_carry), jnp.arange(cutoff, cfg.num_loops, dtype=jnp.int32)
    )

    if prefix_outputs is not None:
        (ec_p, aux_p, gm_p, mmr_p), hl_p, h_p = prefix_outputs
        (ec_s, aux_s, gm_s, mmr_s), hl_s, h_s = suffix_outputs
        ec_rest = jnp.concatenate([ec_p, ec_s], axis=0)
        aux_rest = jnp.concatenate([aux_p, aux_s], axis=0)
        gm_rest = jnp.concatenate([gm_p, gm_s], axis=0)
        mmr_rest = jnp.concatenate([mmr_p, mmr_s], axis=0)
        hl_rest = jnp.concatenate([hl_p, hl_s], axis=0)
        h_rest = None if (use_sampled or not cfg.use_dynamic_loop) else jnp.concatenate([h_p, h_s], axis=0)
    else:
        (ec_rest, aux_rest, gm_rest, mmr_rest), hl_rest, h_s = suffix_outputs
        h_rest = None if (use_sampled or not cfg.use_dynamic_loop) else h_s

    if use_sampled:
        assert sample_depths is not None
        suffix_len = cfg.num_loops - cutoff
        sampled_list = []
        for j in range(int(sample_depths.shape[0])):
            d_j = sample_depths[j]
            s_idx = jnp.clip(d_j - cutoff, 0, suffix_len - 1)
            h_from_s = h_s[s_idx]
            if cutoff > 1:
                sampled_list.append(jnp.where(d_j >= cutoff, h_from_s, prefix_buf[j]))
            else:
                sampled_list.append(h_from_s)
        final_sample_buf = jnp.stack(sampled_list, axis=0)
    else:
        final_sample_buf = prefix_buf

    expert_counts = jnp.concatenate([ec0[None, :], ec_rest], axis=0)
    seq_aux_loss = jnp.mean(jnp.concatenate([aux0[None], aux_rest], axis=0))
    engram_gate_mean = (
        pre_gate_mean
        if (cfg.use_engram and cfg.engram_pre_loop_only)
        else jnp.mean(jnp.concatenate([gm0[None], gm_rest], axis=0))
    )
    router_mmr = jnp.mean(jnp.concatenate([mmr0[None], mmr_rest], axis=0))

    aux = {
        "expert_counts": expert_counts,  # [num_loops, E]
        "seq_aux_loss": seq_aux_loss,
        "engram_gate_mean": engram_gate_mean,
        "router_mmr": router_mmr,
    }

    if cfg.use_dynamic_loop:
        halt_logits = jnp.concatenate([hl_0[None, ...], hl_rest], axis=0)  # [R, B, S]
        halt_probs, _ = _compute_halt_distribution(halt_logits)
        depths = jnp.arange(1, cfg.num_loops + 1, dtype=jnp.float32)[:, None, None]
        expected_loops = jnp.mean(jnp.sum(depths * halt_probs, axis=0))
        q_bar = jnp.mean(halt_probs, axis=(1, 2))
        halt_entropy = -jnp.sum(q_bar * jnp.log(jnp.maximum(q_bar, 1e-8)))

        hidden_last = rms_norm(
            collapse_streams(x_streams_last), params["final_norm_w"], eps=cfg.rms_norm_eps
        )
        if use_sampled:
            aux["h0"] = h0
            aux["sampled_hiddens"] = final_sample_buf
            aux["sample_depths"] = sample_depths
        else:
            assert h_rest is not None
            aux["all_hiddens"] = jnp.concatenate([h0[None, ...], h_rest], axis=0)
        aux["halt_probs"] = halt_probs
        aux["expected_loops"] = expected_loops
        aux["halt_entropy"] = halt_entropy
        return hidden_last, aux

    hidden = rms_norm(collapse_streams(x_streams_last), params["final_norm_w"], eps=cfg.rms_norm_eps)
    aux["expected_loops"] = jnp.asarray(float(cfg.num_loops), dtype=jnp.float32)
    aux["halt_entropy"] = jnp.asarray(0.0, dtype=jnp.float32)
    return hidden, aux


def get_lm_head_weight(params: dict[str, Any], cfg: LoopMoEConfig) -> jax.Array:
    """Return [D, V] LM head projection matrix (tied or untied)."""
    if cfg.tie_word_embeddings:
        return params["embed_tokens"].T
    return params["lm_head"]


def forward_logits(
    params: dict[str, Any],
    input_ids: jax.Array,
    cfg: LoopMoEConfig,
) -> tuple[jax.Array, dict[str, Any]]:
    """Compute full logits [B, S, V] at max active loop depth (intended for eval / unit tests)."""
    prev_cap = cfg.train_loop_capacity_factor
    eval_cfg = (
        LoopMoEConfig(**{**cfg.to_dict(), "train_loop_capacity_factor": 1.0})
        if prev_cap < 1.0
        else cfg
    )
    hidden, aux = forward_hidden(params, input_ids, eval_cfg)
    aux.pop("all_hiddens", None)
    aux.pop("sampled_hiddens", None)
    aux.pop("h0", None)
    logits = (hidden @ get_lm_head_weight(params, eval_cfg)).astype(jnp.float32)
    return logits, aux


def forward_dynamic_exit(
    params: dict[str, Any],
    input_ids: jax.Array,
    cfg: LoopMoEConfig,
    halt_threshold: float | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Inference forward pass with true arXiv:2611.05724 per-token early exit and Loop-0 KV reuse."""
    bsz, seq_len = input_ids.shape
    if cfg.num_loops == 1:
        logits, _ = forward_logits(params, input_ids, cfg)
        return logits, {
            "exit_depth": jnp.zeros((bsz, seq_len), dtype=jnp.int32),
            "mean_loops": jnp.asarray(1.0, dtype=jnp.float32),
            "loops_executed": jnp.asarray(1, dtype=jnp.int32),
        }

    if not cfg.use_dynamic_loop:
        logits, _ = forward_logits(params, input_ids, cfg)
        return logits, {
            "exit_depth": jnp.full((bsz, seq_len), cfg.num_loops - 1, dtype=jnp.int32),
            "mean_loops": jnp.asarray(float(cfg.num_loops), dtype=jnp.float32),
            "loops_executed": jnp.asarray(cfg.num_loops, dtype=jnp.int32),
        }

    tau = float(cfg.halt_threshold if halt_threshold is None else halt_threshold)
    w_halt = params["halt_gate"]["w"][: cfg.num_loops]
    b_halt = params["halt_gate"]["b"][: cfg.num_loops]

    x_streams, x_embed, engram_kv, _ = _embed_and_engram(params, input_ids, cfg)
    x_streams_0, _, cached_kv_0 = _single_loop_step(
        params,
        x_streams,
        x_embed,
        engram_kv,
        0,
        cfg,
        cached_kv=None,
        return_kv=cfg.reuse_kv_across_loops,
    )
    h_0 = rms_norm(collapse_streams(x_streams_0), params["final_norm_w"], eps=cfg.rms_norm_eps)
    lam_0 = jax.nn.sigmoid(
        jnp.einsum("bsd,d->bs", h_0, w_halt[0].astype(h_0.dtype)).astype(jnp.float32) + b_halt[0]
    )
    halted_0 = lam_0 >= tau
    loop_kv = cached_kv_0 if cfg.reuse_kv_across_loops else None

    init_state = (
        jnp.asarray(1, dtype=jnp.int32),
        x_streams_0,
        h_0,
        jnp.zeros((bsz, seq_len), dtype=jnp.int32),
        halted_0,
    )

    def _cond(state):
        r, _, _, _, halted = state
        return (r < cfg.num_loops) & jnp.logical_not(jnp.all(halted))

    def _body(state):
        r, xs, out_h, exit_d, halted = state
        next_xs, _, _ = _single_loop_step(
            params,
            xs,
            x_embed,
            engram_kv,
            r,
            cfg,
            cached_kv=loop_kv,
            return_kv=False,
        )
        h_r = rms_norm(collapse_streams(next_xs), params["final_norm_w"], eps=cfg.rms_norm_eps)
        lam_r = jax.nn.sigmoid(
            jnp.einsum("bsd,d->bs", h_r, w_halt[r].astype(h_r.dtype)).astype(jnp.float32)
            + b_halt[r]
        )
        out_h = jnp.where(halted[:, :, None], out_h, h_r)
        exit_d = jnp.where(halted, exit_d, r)
        halted = halted | (lam_r >= tau) | (r == (cfg.num_loops - 1))
        return (r + 1, next_xs, out_h, exit_d, halted)

    loops_executed, _, out_hidden, exit_depth, _ = jax.lax.while_loop(_cond, _body, init_state)
    logits = (out_hidden @ get_lm_head_weight(params, cfg)).astype(jnp.float32)
    return logits, {
        "exit_depth": exit_depth,
        "mean_loops": jnp.mean(exit_depth.astype(jnp.float32) + 1.0),
        "loops_executed": loops_executed,
    }


def _ce_chunk_loss_and_lse(
    logits_bf16: jax.Array,
    tc: jax.Array,
    mc: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    m = jnp.max(logits_bf16, axis=-1, keepdims=True)
    exp_s = jnp.exp(logits_bf16 - m)
    sum_exp = jnp.sum(exp_s, axis=-1, dtype=jnp.float32)
    lse = m.squeeze(-1).astype(jnp.float32) + jnp.log(sum_exp)
    z_t = (
        jnp.take_along_axis(logits_bf16, tc[:, None], axis=-1)
        .squeeze(-1)
        .astype(jnp.float32)
    )
    return jnp.sum((lse - z_t) * mc), lse


def _ce_chunk_dlogits(
    logits_bf16: jax.Array,
    lse_c: jax.Array,
    tc: jax.Array,
    mc: jax.Array,
    vocab_idx: jax.Array,
    scale_factor: jax.Array,
    out_dtype: jnp.dtype,
) -> jax.Array:
    # Compute probs and diff strictly in native BF16 to halve [chunk, V] HBM & VPU bandwidth
    dt = logits_bf16.dtype
    probs = jnp.exp(logits_bf16 - lse_c[:, None].astype(dt))
    diff = jnp.where(vocab_idx == tc[:, None], probs - jnp.asarray(1.0, dtype=dt), probs)
    return (diff * (mc * scale_factor).astype(dt)[:, None]).astype(out_dtype)


def _ce_chunk_nll_and_dlogits(
    logits_bf16: jax.Array,
    tc: jax.Array,
    mc: jax.Array,
    vocab_idx: jax.Array,
    scale_factor: jax.Array,
    out_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Single-exp native-BF16 CE forward+backward chunk kernel (only [chunk] reductions in FP32)."""
    dt = logits_bf16.dtype
    m = jnp.max(logits_bf16, axis=-1, keepdims=True)
    exp_s = jnp.exp(logits_bf16 - m)
    sum_exp = jnp.sum(exp_s, axis=-1, dtype=jnp.float32)
    lse = m.squeeze(-1).astype(jnp.float32) + jnp.log(sum_exp)
    z_t = (
        jnp.take_along_axis(logits_bf16, tc[:, None], axis=-1)
        .squeeze(-1)
        .astype(jnp.float32)
    )
    nll_c = lse - z_t
    inv_sum = (jnp.asarray(1.0, dtype=jnp.float32) / sum_exp).astype(dt)
    probs = exp_s * inv_sum[:, None]
    diff = jnp.where(vocab_idx == tc[:, None], probs - jnp.asarray(1.0, dtype=dt), probs)
    d_logits = (diff * (mc * scale_factor).astype(dt)[:, None]).astype(out_dtype)
    return nll_c, d_logits


def _run_scan_ce_fwd(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate scalar CE loss over chunks while retaining only [num_chunks, chunk] LSE."""

    def _fwd_step(loss_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array]):
        hc, tc, mc = chunk_in
        logits_bf16 = hc @ lm_head_weight
        chunk_loss, lse = _ce_chunk_loss_and_lse(logits_bf16, tc, mc)
        return loss_acc + chunk_loss, lse

    total_loss, lse_chunks = jax.lax.scan(
        _fwd_step,
        jnp.asarray(0.0, dtype=jnp.float32),
        (h_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, lse_chunks


@jax.custom_vjp
def _scan_chunked_ce(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _ = _run_scan_ce_fwd(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)
    return out


def _scan_chunked_ce_fwd(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, lse_chunks = _run_scan_ce_fwd(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)
    return out, (h_chunks, lm_head_weight, t_chunks, m_chunks, lse_chunks, denom)


def _scan_chunked_ce_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    h_chunks, lm_head_weight, t_chunks, m_chunks, lse_chunks, denom = res
    vocab_size = lm_head_weight.shape[1]
    vocab_idx = jnp.arange(vocab_size, dtype=t_chunks.dtype)[None, :]
    scale_factor = g_out / denom

    def _bwd_step(
        d_lm_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        hc, tc, mc, lse_c = chunk_in
        logits_bf16 = hc @ lm_head_weight
        d_logits = _ce_chunk_dlogits(logits_bf16, lse_c, tc, mc, vocab_idx, scale_factor, hc.dtype)
        d_hc = (d_logits @ lm_head_weight.T).astype(hc.dtype)
        d_lm_acc = d_lm_acc + (hc.T @ d_logits).astype(d_lm_acc.dtype)
        return d_lm_acc, d_hc

    d_lm_w, d_h_chunks = jax.lax.scan(
        _bwd_step,
        jnp.zeros_like(lm_head_weight),
        (h_chunks, t_chunks, m_chunks, lse_chunks),
        unroll=1,
    )
    return d_h_chunks, d_lm_w, None, None, None


_scan_chunked_ce.defvjp(_scan_chunked_ce_fwd, _scan_chunked_ce_bwd)


def _run_scan_ce_fwd_vd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Layout-native [V, D] chunked CE forward avoiding [V, D] -> [D, V] transpose copy."""

    def _fwd_step(loss_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array]):
        hc, tc, mc = chunk_in
        logits_bf16 = jax.lax.dot_general(hc, lm_weight_vd, (((1,), (1,)), ((), ())))
        chunk_loss, lse = _ce_chunk_loss_and_lse(logits_bf16, tc, mc)
        return loss_acc + chunk_loss, lse

    total_loss, lse_chunks = jax.lax.scan(
        _fwd_step,
        jnp.asarray(0.0, dtype=jnp.float32),
        (h_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, lse_chunks


@jax.custom_vjp
def _scan_chunked_ce_vd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _ = _run_scan_ce_fwd_vd(h_chunks, lm_weight_vd, t_chunks, m_chunks, denom)
    return out


def _scan_chunked_ce_vd_fwd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, lse_chunks = _run_scan_ce_fwd_vd(h_chunks, lm_weight_vd, t_chunks, m_chunks, denom)
    return out, (h_chunks, lm_weight_vd, t_chunks, m_chunks, lse_chunks, denom)


def _scan_chunked_ce_vd_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    h_chunks, lm_weight_vd, t_chunks, m_chunks, lse_chunks, denom = res
    vocab_size = lm_weight_vd.shape[0]
    vocab_idx = jnp.arange(vocab_size, dtype=t_chunks.dtype)[None, :]
    scale_factor = g_out / denom

    def _bwd_step(
        d_w_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        hc, tc, mc, lse_c = chunk_in
        logits_bf16 = jax.lax.dot_general(hc, lm_weight_vd, (((1,), (1,)), ((), ())))
        d_logits = _ce_chunk_dlogits(logits_bf16, lse_c, tc, mc, vocab_idx, scale_factor, hc.dtype)
        d_hc = (d_logits @ lm_weight_vd).astype(hc.dtype)
        d_w_acc = d_w_acc + jax.lax.dot_general(
            d_logits, hc, (((0,), (0,)), ((), ()))
        ).astype(d_w_acc.dtype)
        return d_w_acc, d_hc

    d_lm_w_vd, d_h_chunks = jax.lax.scan(
        _bwd_step,
        jnp.zeros_like(lm_weight_vd),
        (h_chunks, t_chunks, m_chunks, lse_chunks),
        unroll=1,
    )
    return d_h_chunks, d_lm_w_vd, None, None, None


_scan_chunked_ce_vd.defvjp(_scan_chunked_ce_vd_fwd, _scan_chunked_ce_vd_bwd)


def _run_saved_logits_ce_fwd(
    h_chunks: jax.Array,
    lm_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
    weight_is_vd: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute single-GEMM BF16 logits and scan LSE over sub-chunks (zero dynamic_update_slice)."""
    num_chunks, sub_chunk, d_model = h_chunks.shape
    h_flat = h_chunks.reshape(num_chunks * sub_chunk, d_model)
    logits_all = (
        jax.lax.dot_general(h_flat, lm_weight, (((1,), (1,)), ((), ())))
        if weight_is_vd
        else (h_flat @ lm_weight)
    )
    logits_chunks = logits_all.reshape(num_chunks, sub_chunk, logits_all.shape[-1])

    def _lse_step(loss_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array]):
        logits_c, tc, mc = chunk_in
        chunk_loss, lse = _ce_chunk_loss_and_lse(logits_c, tc, mc)
        return loss_acc + chunk_loss, lse

    total_loss, lse_chunks = jax.lax.scan(
        _lse_step,
        jnp.asarray(0.0, dtype=jnp.float32),
        (logits_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, logits_chunks, lse_chunks


def _run_saved_logits_ce_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
    weight_is_vd: bool,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    h_chunks, lm_weight, logits_chunks, t_chunks, m_chunks, lse_chunks, denom = res
    vocab_size = lm_weight.shape[0] if weight_is_vd else lm_weight.shape[1]
    vocab_idx = jnp.arange(vocab_size, dtype=t_chunks.dtype)[None, :]
    scale_factor = g_out / denom

    def _bwd_step(
        d_w_acc: jax.Array,
        chunk_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        hc, logits_c, tc, mc, lse_c = chunk_in
        d_logits = _ce_chunk_dlogits(logits_c, lse_c, tc, mc, vocab_idx, scale_factor, hc.dtype)
        if weight_is_vd:
            d_hc = (d_logits @ lm_weight).astype(hc.dtype)
            d_w_acc = d_w_acc + jax.lax.dot_general(
                d_logits, hc, (((0,), (0,)), ((), ()))
            ).astype(d_w_acc.dtype)
        else:
            d_hc = (d_logits @ lm_weight.T).astype(hc.dtype)
            d_w_acc = d_w_acc + (hc.T @ d_logits).astype(d_w_acc.dtype)
        return d_w_acc, d_hc

    d_lm_w, d_h_chunks = jax.lax.scan(
        _bwd_step,
        jnp.zeros_like(lm_weight),
        (h_chunks, logits_chunks, t_chunks, m_chunks, lse_chunks),
        unroll=1,
    )
    return d_h_chunks, d_lm_w, None, None, None


@jax.custom_vjp
def _saved_logits_ce(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _, _ = _run_saved_logits_ce_fwd(h_chunks, lm_head_weight, t_chunks, m_chunks, denom, False)
    return out


def _saved_logits_ce_fwd(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, logits_chunks, lse_chunks = _run_saved_logits_ce_fwd(
        h_chunks, lm_head_weight, t_chunks, m_chunks, denom, False
    )
    return out, (h_chunks, lm_head_weight, logits_chunks, t_chunks, m_chunks, lse_chunks, denom)


def _saved_logits_ce_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    return _run_saved_logits_ce_bwd(res, g_out, False)


_saved_logits_ce.defvjp(_saved_logits_ce_fwd, _saved_logits_ce_bwd)


@jax.custom_vjp
def _saved_logits_ce_vd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _, _ = _run_saved_logits_ce_fwd(h_chunks, lm_weight_vd, t_chunks, m_chunks, denom, True)
    return out


def _saved_logits_ce_vd_fwd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, logits_chunks, lse_chunks = _run_saved_logits_ce_fwd(
        h_chunks, lm_weight_vd, t_chunks, m_chunks, denom, True
    )
    return out, (h_chunks, lm_weight_vd, logits_chunks, t_chunks, m_chunks, lse_chunks, denom)


def _saved_logits_ce_vd_bwd(
    res: tuple[jax.Array, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    return _run_saved_logits_ce_bwd(res, g_out, True)


_saved_logits_ce_vd.defvjp(_saved_logits_ce_vd_fwd, _saved_logits_ce_vd_bwd)


def _run_fused_ce_fwd_bwd(
    h_chunks: jax.Array,
    lm_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
    weight_is_vd: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-pass chunked CE computing loss and unit-scale gradients without retaining logits."""
    vocab_size = lm_weight.shape[0] if weight_is_vd else lm_weight.shape[1]
    vocab_idx = jnp.arange(vocab_size, dtype=t_chunks.dtype)[None, :]
    inv_denom = jnp.asarray(1.0, dtype=jnp.float32) / denom

    def _step(
        carry: tuple[jax.Array, jax.Array],
        chunk_in: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        loss_acc, d_w_acc = carry
        hc, tc, mc = chunk_in
        logits_bf16 = (
            jax.lax.dot_general(hc, lm_weight, (((1,), (1,)), ((), ())))
            if weight_is_vd
            else (hc @ lm_weight)
        )
        nll_c, d_logits = _ce_chunk_nll_and_dlogits(
            logits_bf16, tc, mc, vocab_idx, inv_denom, hc.dtype
        )
        chunk_loss = jnp.sum(nll_c * mc)
        if weight_is_vd:
            d_hc = (d_logits @ lm_weight).astype(hc.dtype)
            d_w_acc = d_w_acc + jax.lax.dot_general(
                d_logits, hc, (((0,), (0,)), ((), ()))
            ).astype(d_w_acc.dtype)
        else:
            d_hc = (d_logits @ lm_weight.T).astype(hc.dtype)
            d_w_acc = d_w_acc + (hc.T @ d_logits).astype(d_w_acc.dtype)
        return (loss_acc + chunk_loss, d_w_acc), d_hc

    (total_loss, d_lm_w), d_h_chunks = jax.lax.scan(
        _step,
        (jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(lm_weight)),
        (h_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, d_h_chunks, d_lm_w


@jax.custom_vjp
def _fused_chunked_ce(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _ = _run_scan_ce_fwd(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)
    return out


def _fused_chunked_ce_fwd(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    out, d_h_chunks, d_lm_w = _run_fused_ce_fwd_bwd(
        h_chunks, lm_head_weight, t_chunks, m_chunks, denom, False
    )
    return out, (d_h_chunks, d_lm_w)


def _fused_chunked_ce_bwd(
    res: tuple[jax.Array, jax.Array],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
    d_h_chunks, d_lm_w = res
    return (
        d_h_chunks * g_out.astype(d_h_chunks.dtype),
        d_lm_w * g_out.astype(d_lm_w.dtype),
        None,
        None,
        None,
    )


_fused_chunked_ce.defvjp(_fused_chunked_ce_fwd, _fused_chunked_ce_bwd)


@jax.custom_vjp
def _fused_chunked_ce_vd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> jax.Array:
    out, _ = _run_scan_ce_fwd_vd(h_chunks, lm_weight_vd, t_chunks, m_chunks, denom)
    return out


def _fused_chunked_ce_vd_fwd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    out, d_h_chunks, d_lm_w = _run_fused_ce_fwd_bwd(
        h_chunks, lm_weight_vd, t_chunks, m_chunks, denom, True
    )
    return out, (d_h_chunks, d_lm_w)


_fused_chunked_ce_vd.defvjp(_fused_chunked_ce_vd_fwd, _fused_chunked_ce_bwd)


def _ce_chunk_nll_and_lse(
    logits_bf16: jax.Array,
    tc: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    m = jnp.max(logits_bf16, axis=-1, keepdims=True)
    exp_s = jnp.exp(logits_bf16 - m)
    sum_exp = jnp.sum(exp_s, axis=-1, dtype=jnp.float32)
    lse = m.squeeze(-1).astype(jnp.float32) + jnp.log(sum_exp)
    z_t = (
        jnp.take_along_axis(logits_bf16, tc[:, None], axis=-1)
        .squeeze(-1)
        .astype(jnp.float32)
    )
    return lse - z_t, lse


def _run_scan_ce_fwd_nll(
    h_chunks: jax.Array,
    lm_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
    weight_is_vd: bool,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate scalar CE loss and per-token NLL over chunks."""

    def _fwd_step(loss_acc: jax.Array, chunk_in: tuple[jax.Array, jax.Array, jax.Array]):
        hc, tc, mc = chunk_in
        logits_bf16 = (
            jax.lax.dot_general(hc, lm_weight, (((1,), (1,)), ((), ())))
            if weight_is_vd
            else (hc @ lm_weight)
        )
        nll_c, _ = _ce_chunk_nll_and_lse(logits_bf16, tc)
        return loss_acc + jnp.sum(nll_c * mc), nll_c

    total_loss, nll_chunks = jax.lax.scan(
        _fwd_step,
        jnp.asarray(0.0, dtype=jnp.float32),
        (h_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, nll_chunks


def _run_fused_ce_fwd_bwd_nll(
    h_chunks: jax.Array,
    lm_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
    weight_is_vd: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Single-pass chunked CE computing loss, per-token NLL, and unit-scale gradients."""
    vocab_size = lm_weight.shape[0] if weight_is_vd else lm_weight.shape[1]
    vocab_idx = jnp.arange(vocab_size, dtype=t_chunks.dtype)[None, :]
    inv_denom = jnp.asarray(1.0, dtype=jnp.float32) / denom

    def _step(
        carry: tuple[jax.Array, jax.Array],
        chunk_in: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        loss_acc, d_w_acc = carry
        hc, tc, mc = chunk_in
        logits_bf16 = (
            jax.lax.dot_general(hc, lm_weight, (((1,), (1,)), ((), ())))
            if weight_is_vd
            else (hc @ lm_weight)
        )
        nll_c, d_logits = _ce_chunk_nll_and_dlogits(
            logits_bf16, tc, mc, vocab_idx, inv_denom, hc.dtype
        )
        chunk_loss = jnp.sum(nll_c * mc)
        if weight_is_vd:
            d_hc = (d_logits @ lm_weight).astype(hc.dtype)
            d_w_acc = d_w_acc + jax.lax.dot_general(
                d_logits, hc, (((0,), (0,)), ((), ()))
            ).astype(d_w_acc.dtype)
        else:
            d_hc = (d_logits @ lm_weight.T).astype(hc.dtype)
            d_w_acc = d_w_acc + (hc.T @ d_logits).astype(d_w_acc.dtype)
        return (loss_acc + chunk_loss, d_w_acc), (nll_c, d_hc)

    (total_loss, d_lm_w), (nll_chunks, d_h_chunks) = jax.lax.scan(
        _step,
        (jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(lm_weight)),
        (h_chunks, t_chunks, m_chunks),
        unroll=1,
    )
    return total_loss / denom, nll_chunks, d_h_chunks, d_lm_w


@jax.custom_vjp
def _fused_chunked_ce_nll(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _run_scan_ce_fwd_nll(h_chunks, lm_head_weight, t_chunks, m_chunks, denom, False)


def _fused_chunked_ce_nll_fwd(
    h_chunks: jax.Array,
    lm_head_weight: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    out, nll_chunks, d_h_chunks, d_lm_w = _run_fused_ce_fwd_bwd_nll(
        h_chunks, lm_head_weight, t_chunks, m_chunks, denom, False
    )
    return (out, nll_chunks), (d_h_chunks, d_lm_w)


def _fused_chunked_ce_nll_bwd(
    res: tuple[jax.Array, jax.Array],
    g_out: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, None, None, None]:
    d_h_chunks, d_lm_w = res
    g_loss, _ = g_out
    return (
        d_h_chunks * g_loss.astype(d_h_chunks.dtype),
        d_lm_w * g_loss.astype(d_lm_w.dtype),
        None,
        None,
        None,
    )


_fused_chunked_ce_nll.defvjp(_fused_chunked_ce_nll_fwd, _fused_chunked_ce_nll_bwd)


@jax.custom_vjp
def _fused_chunked_ce_nll_vd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _run_scan_ce_fwd_nll(h_chunks, lm_weight_vd, t_chunks, m_chunks, denom, True)


def _fused_chunked_ce_nll_vd_fwd(
    h_chunks: jax.Array,
    lm_weight_vd: jax.Array,
    t_chunks: jax.Array,
    m_chunks: jax.Array,
    denom: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    out, nll_chunks, d_h_chunks, d_lm_w = _run_fused_ce_fwd_bwd_nll(
        h_chunks, lm_weight_vd, t_chunks, m_chunks, denom, True
    )
    return (out, nll_chunks), (d_h_chunks, d_lm_w)


_fused_chunked_ce_nll_vd.defvjp(_fused_chunked_ce_nll_vd_fwd, _fused_chunked_ce_nll_bwd)


def _prepare_ce_chunks(
    hidden: jax.Array,
    targets: jax.Array,
    chunk_size: int,
    loss_mask: jax.Array | None,
    denom_override: jax.Array | float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    bsz, seq_len, d_model = hidden.shape
    total_tokens = bsz * seq_len
    h_flat = hidden.reshape(total_tokens, d_model)
    t_flat = targets.reshape(total_tokens)
    m_flat = (
        jnp.ones((total_tokens,), dtype=jnp.float32)
        if loss_mask is None
        else loss_mask.reshape(total_tokens).astype(jnp.float32)
    )
    denom = (
        jnp.maximum(jnp.sum(m_flat), 1.0)
        if denom_override is None
        else jnp.maximum(jnp.asarray(denom_override, dtype=jnp.float32), 1.0)
    )
    effective_chunk = min(chunk_size, total_tokens)
    if total_tokens % effective_chunk != 0:
        effective_chunk = math.gcd(total_tokens, effective_chunk)
        if effective_chunk == 1 and total_tokens <= 4096:
            effective_chunk = total_tokens
    num_chunks = total_tokens // effective_chunk
    return (
        h_flat.reshape(num_chunks, effective_chunk, d_model),
        t_flat.reshape(num_chunks, effective_chunk),
        m_flat.reshape(num_chunks, effective_chunk),
        denom,
    )


def chunked_cross_entropy_loss(
    hidden: jax.Array,
    lm_head_weight: jax.Array,
    targets: jax.Array,
    chunk_size: int = 16_384,
    loss_mask: jax.Array | None = None,
    weight_is_vd: bool = False,
) -> jax.Array:
    """Single-pass fused chunked linear projection + softmax cross-entropy."""
    h_chunks, t_chunks, m_chunks, denom = _prepare_ce_chunks(hidden, targets, chunk_size, loss_mask)
    fused_fn = _fused_chunked_ce_vd if weight_is_vd else _fused_chunked_ce
    return fused_fn(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)


def chunked_cross_entropy_loss_and_nll(
    hidden: jax.Array,
    lm_head_weight: jax.Array,
    targets: jax.Array,
    chunk_size: int = 16_384,
    loss_mask: jax.Array | None = None,
    weight_is_vd: bool = False,
    denom_override: jax.Array | float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single-pass fused chunked CE returning `(weighted_loss, stop_grad(per_token_nll))`."""
    bsz, seq_len, _ = hidden.shape
    h_chunks, t_chunks, m_chunks, denom = _prepare_ce_chunks(
        hidden, targets, chunk_size, loss_mask, denom_override=denom_override
    )
    fused_fn = _fused_chunked_ce_nll_vd if weight_is_vd else _fused_chunked_ce_nll
    loss, nll_chunks = fused_fn(h_chunks, lm_head_weight, t_chunks, m_chunks, denom)
    return loss, jax.lax.stop_gradient(nll_chunks.reshape(bsz, seq_len))


def _stride_slice(
    arr: jax.Array, seq_axis: int, stride: int, phase: jax.Array | int
) -> jax.Array:
    """Slice sequence axis every `stride` tokens starting at `phase` with static output shape."""
    if stride <= 1:
        return arr
    s_len = arr.shape[seq_axis]
    if isinstance(phase, int) or (s_len % stride != 0):
        slices: list[Any] = [slice(None)] * arr.ndim
        slices[seq_axis] = slice(None, None, stride)
        return arr[tuple(slices)]
    new_shape = arr.shape[:seq_axis] + (s_len // stride, stride) + arr.shape[seq_axis + 1 :]
    return jnp.take(arr.reshape(new_shape), phase, axis=seq_axis + 1)


def forward_model(
    params: dict[str, Any],
    input_ids: jax.Array,
    targets: jax.Array,
    cfg: LoopMoEConfig,
    loss_mask: jax.Array | None = None,
    rng_key: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Forward pass returning total loss (Eq. 6 dynamic loop + sampled CE + BPTT) and telemetry."""
    if rng_key is not None and cfg.num_loops > 1 and cfg.use_dynamic_loop:
        key_drop, key_depth, key_phase = jax.random.split(rng_key, 3)
    else:
        key_drop = key_depth = key_phase = None

    sample_depths: jax.Array | None = None
    if cfg.num_loops > 1 and cfg.use_dynamic_loop:
        k_ce_cfg = int(cfg.dynamic_loop_ce_sample_depths)
        deep_avail = cfg.num_loops - 1
        if 1 <= k_ce_cfg < deep_avail:
            sample_depths = _select_ce_sample_depths(cfg.num_loops, k_ce_cfg, key_depth)

    hidden, aux = forward_hidden(
        params,
        input_ids,
        cfg,
        rng_key=key_drop,
        sample_depths=sample_depths,
    )
    if cfg.tie_word_embeddings:
        lm_w = params["embed_tokens"]
        weight_is_vd = True
    else:
        lm_w = params["lm_head"]
        weight_is_vd = False

    if cfg.num_loops == 1 or not cfg.use_dynamic_loop:
        with jax.named_scope("chunked_ce_loss"):
            ce_loss = chunked_cross_entropy_loss(
                hidden,
                lm_w,
                targets,
                chunk_size=cfg.chunk_ce_size,
                loss_mask=loss_mask,
                weight_is_vd=weight_is_vd,
            )
        total_loss = ce_loss + aux["seq_aux_loss"]
        aux["ce_loss"] = ce_loss
        return total_loss, aux

    halt_probs = aux.pop("halt_probs")  # [R, B, S]
    r_steps, bsz, seq_len = halt_probs.shape
    base_mask = (
        jnp.ones((bsz, seq_len), dtype=jnp.float32)
        if loss_mask is None
        else loss_mask.astype(jnp.float32)
    )
    denom_full = jnp.maximum(jnp.sum(base_mask), 1.0)

    if "sampled_hiddens" in aux:
        h0 = aux.pop("h0")
        deep_hiddens = aux.pop("sampled_hiddens")  # [K, B, S, D]
        used_depths = aux.pop("sample_depths")     # [K]
        q_deep = halt_probs[used_depths]
        depth_scale = float(r_steps - 1) / float(deep_hiddens.shape[0])
    else:
        all_hiddens = aux.pop("all_hiddens")  # [R, B, S, D]
        h0 = all_hiddens[0]
        deep_hiddens = all_hiddens[1:]
        q_deep = halt_probs[1:]
        depth_scale = 1.0

    n_deep, _, _, d_model = deep_hiddens.shape
    stride = max(1, int(cfg.dynamic_loop_ce_subsample))
    phase: jax.Array | int = (
        jax.random.randint(key_phase, (), 0, stride, dtype=jnp.int32)
        if (key_phase is not None and stride > 1 and seq_len % stride == 0)
        else 0
    )

    with jax.named_scope("chunked_ce_loss"):
        if stride == 1:
            n_stack = 1 + n_deep
            h_stack = jnp.concatenate([h0[None, ...], deep_hiddens], axis=0)
            w_deep = jax.lax.stop_gradient(q_deep) * float(depth_scale) * base_mask[None, :, :]
            w_stack = jnp.concatenate([base_mask[None, :, :], w_deep], axis=0)
            lm_loss, nll_flat = chunked_cross_entropy_loss_and_nll(
                h_stack.reshape(n_stack * bsz, seq_len, d_model),
                lm_w,
                jnp.broadcast_to(targets[None, :, :], (n_stack, bsz, seq_len)).reshape(
                    n_stack * bsz, seq_len
                ),
                chunk_size=cfg.chunk_ce_size,
                loss_mask=w_stack.reshape(n_stack * bsz, seq_len),
                weight_is_vd=weight_is_vd,
                denom_override=denom_full,
            )
            nll_stack = nll_flat.reshape(n_stack, bsz, seq_len)
            ce_loss_r0 = jnp.sum(nll_stack[0] * base_mask) / denom_full
            gate_loss = (
                jnp.sum(halt_probs[0] * jax.lax.stop_gradient(nll_stack[0]) * base_mask)
                + float(depth_scale)
                * jnp.sum(q_deep * jax.lax.stop_gradient(nll_stack[1:]) * base_mask[None, :, :])
            ) / denom_full
        else:
            loss_0, nll_0 = chunked_cross_entropy_loss_and_nll(
                h0,
                lm_w,
                targets,
                chunk_size=cfg.chunk_ce_size,
                loss_mask=base_mask,
                weight_is_vd=weight_is_vd,
                denom_override=denom_full,
            )
            sub_mask = _stride_slice(base_mask, 1, stride, phase)
            denom_sub = jnp.maximum(jnp.sum(sub_mask), 1.0)
            sub_targets = _stride_slice(targets, 1, stride, phase)
            s_sub = sub_targets.shape[1]
            q0_sub = _stride_slice(halt_probs[0], 1, stride, phase)
            q_deep_sub = _stride_slice(q_deep, 2, stride, phase)
            deep_h_sub = _stride_slice(deep_hiddens, 2, stride, phase)
            nll_0_sub = _stride_slice(nll_0, 1, stride, phase)
            reconsider_loss, nll_deep_flat = chunked_cross_entropy_loss_and_nll(
                deep_h_sub.reshape(n_deep * bsz, s_sub, d_model),
                lm_w,
                jnp.broadcast_to(sub_targets[None, :, :], (n_deep, bsz, s_sub)).reshape(
                    n_deep * bsz, s_sub
                ),
                chunk_size=cfg.chunk_ce_size,
                loss_mask=(
                    jax.lax.stop_gradient(q_deep_sub) * float(depth_scale) * sub_mask[None, :, :]
                ).reshape(n_deep * bsz, s_sub),
                weight_is_vd=weight_is_vd,
                denom_override=denom_sub,
            )
            lm_loss = loss_0 + reconsider_loss
            ce_loss_r0 = loss_0
            nll_deep_sub = nll_deep_flat.reshape(n_deep, bsz, s_sub)
            gate_loss = (
                jnp.sum(q0_sub * jax.lax.stop_gradient(nll_0_sub) * sub_mask)
                + float(depth_scale)
                * jnp.sum(
                    q_deep_sub * jax.lax.stop_gradient(nll_deep_sub) * sub_mask[None, :, :]
                )
            ) / denom_sub

    with jax.named_scope("dynamic_halt_loss"):
        q_bar = jnp.sum(halt_probs * base_mask[None, :, :], axis=(1, 2)) / denom_full
        halt_entropy = -jnp.sum(q_bar * jnp.log(jnp.maximum(q_bar, 1e-8)))
        balance_loss = -jnp.asarray(cfg.halt_balance_weight, dtype=jnp.float32) * halt_entropy
        depths = jnp.arange(1, r_steps + 1, dtype=jnp.float32)
        expected_loops = jnp.sum(depths * q_bar)

    total_loss = lm_loss + gate_loss + balance_loss + aux["seq_aux_loss"]
    aux["ce_loss"] = lm_loss
    aux["ce_loss_r0"] = ce_loss_r0
    aux["gate_loss"] = gate_loss
    aux["halt_entropy"] = halt_entropy
    aux["expected_loops"] = expected_loops
    return total_loss, aux


def expand_checkpoint_1_to_n_loops(
    params: dict[str, Any],
    key: jax.Array,
    noise_std: float = 1e-3,
) -> dict[str, Any]:
    """Expand a Phase-1 (R=1) trained checkpoint to Phase-2 (R=N) preserving step-0 logits."""
    new_params = jax.tree_util.tree_map(lambda x: x, params)
    adaln = dict(new_params["iter_adaln"])
    k_emb, k_router = jax.random.split(key, 2)

    loop_emb = adaln["loop_emb"]
    max_loops = loop_emb.shape[0]
    base_emb = loop_emb[0:1]
    noise = jax.random.normal(k_emb, (max_loops - 1, loop_emb.shape[1]), dtype=loop_emb.dtype) * noise_std
    expanded_emb = jnp.concatenate([base_emb, jnp.repeat(base_emb, max_loops - 1, axis=0) + noise], axis=0)
    adaln["loop_emb"] = expanded_emb

    # Force t=0 active (preserving Phase 1 value) and t=1..max_loops-1 exact zero for step-0 identity
    loop_alpha = jnp.zeros_like(adaln["loop_alpha"]).at[0].set(adaln["loop_alpha"][0])
    adaln["loop_alpha"] = loop_alpha
    new_params["iter_adaln"] = adaln

    # Warm-start ONLY per-loop MLA Q/O banks from trained loop 0; keep single-copy KV/CSA2 intact
    if new_params["mla"]["w_dq"].ndim == 3:
        new_params["mla"] = {
            k: (jnp.repeat(v[0:1], v.shape[0], axis=0) if is_per_loop_mla_leaf(k, v) else v)
            for k, v in new_params["mla"].items()
        }

    # Broadcast learned Phase-1 router bias and warm-start untied per-bank routers
    moe = dict(new_params["moe"])
    base_bias = moe["router_bias"][0:1]
    moe["router_bias"] = jnp.repeat(base_bias, max_loops, axis=0)
    if moe["router_w"].ndim == 3:
        base_rw = moe["router_w"][0:1]
        n_rw_banks = moe["router_w"].shape[0]
        rw_noise = (
            jax.random.normal(k_router, (n_rw_banks - 1, *base_rw.shape[1:]), dtype=base_rw.dtype)
            * noise_std
        )
        moe["router_w"] = jnp.concatenate(
            [base_rw, jnp.repeat(base_rw, n_rw_banks - 1, axis=0) + rw_noise], axis=0
        )
    new_params["moe"] = moe
    return new_params


expand_checkpoint_1_to_8_loops = expand_checkpoint_1_to_n_loops


def summarize_model(cfg: LoopMoEConfig) -> str:
    """Format human-readable parameter and capacity breakdown."""
    stats = count_analytic_params(cfg)
    loop_summary = (
        f"max_R={cfg.num_loops}, dyn={cfg.is_dynamic_loop_active}, kv_reuse={cfg.reuse_kv_across_loops}, "
        f"cap={cfg.train_loop_capacity_factor:.2f}, ce_k={cfg.dynamic_loop_ce_sample_depths}"
        if cfg.num_loops > 1
        else "R=1, dyn=False"
    )
    return (
        f"LoopMoE(d={cfg.d_model}, V={cfg.vocab_size}, {loop_summary}, "
        f"experts={cfg.n_routed_experts}x{cfg.d_ff_expert} top-{cfg.top_k}+shared) | "
        f"Total={stats['total']/1e6:.2f}M | Emb={stats['embedding']/1e6:.2f}M | "
        f"Engram={stats['engram_total']/1e6:.2f}M ({stats['engram_sparse_ratio']*100:.1f}% sparse) | "
        f"MLA={stats['mla']/1e6:.2f}M | MoE={stats['moe_total']/1e6:.2f}M | "
        f"Active/loop(non-emb)={stats['active_per_loop_non_emb']/1e6:.2f}M "
        f"(Attn:FFN = 1:{stats['attn_to_ffn_active_ratio']:.2f})"
    )
