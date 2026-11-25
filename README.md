# 1.49B 1-Layer DeepSeek-V4.1-Flash LoopMoE

A **1.49B parameter, single physical layer (`L=1`)** LLM combining **DeepSeek-V4.1-Flash** architectural primitives with **LoopMoE** & dynamic early-exit (`arXiv:2611.05724`), engineered specifically for explicit config-driven **TPU v6e-1** (`918 TFLOPS BF16`, `32 GB HBM`) training via **JAX + Optax + OpenXLA Tokamax**.

---

## Architecture Breakdown (`d_model = 2048`, `V = 32,768`, `L = 1`)

| Module | Design | Parameters |
| :--- | :--- | :---: |
| **Tied Token Embedding / Head** | `32,768 × 2048`, native BF16 chunked CE | **67.11M** |
| **Engram Conditional Memory** (`arXiv:2601.07372`) | 4 prime 1024-aligned tables (`2048 × 1024`, `d_head=256`), Row-Wise Factored AdamW | **541.08M** (39.9% sparse) |
| **Single-Pass mHC** (`arXiv:2512.24880`) | `n_streams = 2` closed-form Birkhoff doubly stochastic residual | **0.26M** |
| **LoopMoE IterAdaLN** | Per-loop bottleneck MLP (`b=128`) + `loop_alpha[t]` step-0 identity gating (`max_loops=32`) | **~2M** |
| **MLA + YaRN** | `16` heads, `d_q_lora=768`, `d_kv_lora=512`, `d_qk=96+32=128`, `d_v=128`; 8-bank cyclic Q/O + Loop-0 KV reuse; **Fused Splash Attention** | **61.68M** |
| **Fine-Grained DeepSeek MoE** | `128` routed (`d_ff=1024`, top-4) + `1` shared (`d_ff=1024`), **Pallas DMA gather/combine + Tokamax `ragged_dot`**, Aux-Loss-Free bias | **813.70M** |
| **Total** | Active non-emb/loop: `41.80M` (`Attn:FFN = 1:3.06`), `bptt_window=8` | **1,485.60M (~1.49B)** |

---

## Explicit TPU v6e-1 Config & Lookahead Curriculum (`configs/v6e_1_train.json`)

All hyperparameters, per-stage batch sizes, and rematerialization flags are explicitly declared in [`configs/v6e_1_train.json`](configs/v6e_1_train.json) (no hidden defaults):

- **Phase 1 (`R = 1`)**:
  - `S1 (4K)`: `microbatch=16, grad_accum=4, warmup_steps=1000, remat_moe=false, remat_sublayers=false` (no Lookahead during warmup)
  - `S2 (8K)`: `microbatch=8, grad_accum=4, warmup_steps=0, remat_moe=false, remat_sublayers=false`
  - `S3 (16K)`: `microbatch=2, grad_accum=8, warmup_steps=0, remat_moe=false, remat_sublayers=false`
  - `S4 (32K)`: `microbatch=1, grad_accum=8, warmup_steps=0, remat_moe=false, remat_sublayers=false`
- **Phase 2 (`R = 2, 3, ..., 32` at `32K`)**:
  - `S5..S35`: `microbatch=1, grad_accum=8, bptt_window=8, remat_moe=true, remat_sublayers=false, chunk_ce_size=8192`
- **Lookahead Time-Efficiency Switching**:
  - After stage warmup completes, alternates a large block on stage $A$ (`main_steps=200`) with a small probe block on next stage $B$ (`probe_steps=20`), using strictly **stage $B$'s training `ce_loss`** (`eff_A = (prev_tail - head) / t_A` vs `eff_B = (head - tail) / t_B_eff`) to switch only when stage $B$ yields faster loss reduction per second.

---

## Quickstart

```bash
pip install -r requirements.txt
pytest -v tests/test_loopmoe.py

# Train on TPU v6e-1 using explicit config (zero hidden defaults)
python scripts/train_tpu.py \
  --config configs/v6e_1_train.json \
  --data-mode stream \
  --tokenizer artifacts/tokenizer_32k_en/tokenizer.json \
  --ckpt-dir checkpoints \
  --cache-dir jax_cache
```
