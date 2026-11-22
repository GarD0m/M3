# 1B 1-Layer DeepSeek-V4.1-Flash LoopMoE

A ~1.00B parameter, **single physical layer (`L=1`)** LLM combining **DeepSeek-V4.1-Flash** architectural primitives with **LoopMoE** (`arXiv:2606.04438`), engineered for local 64K bilingual tokenizer construction and high-MFU single-chip Google Colab **TPU v6e-1** (`918 TFLOPS BF16`, `32 GB HBM`) training via **JAX + Optax + OpenXLA Tokamax**.

---

## Architecture Breakdown (`d_model = 2048`, `V = 65,536`, `L = 1`)

| Module | Design | Parameters |
| :--- | :--- | :---: |
| **Tied Token Embedding / Head** | `65,536 × 2048`, exact 256×256 MXU aligned | **134.22M** |
| **Engram Conditional Memory** (`arXiv:2601.07372`) | 8 prime tables (`~1.46M × 128`, 2-gram & 3-gram), pre-loop single gather, RMSNorm context gating + causal conv | **191.13M** (22.5% sparse) |
| **Single-Pass mHC** (`arXiv:2512.24880`) | `n_streams = 4`, Sinkhorn-Knopp doubly stochastic `H_res` (20 iters) + bounded `H_pre`, `H_post` | **0.40M** |
| **LoopMoE IterAdaLN** (`arXiv:2606.04438`) | Joint `(h_t, loop_emb[t])` bottleneck MLP + `loop_alpha[t]` step-0 identity gating for `R=1 -> R=8` | **2.10M** |
| **MLA + YaRN (+ optional CSA2)** | `16` heads, `d_q_lora=768`, `d_kv_lora=512`, `d_qk=128+64`, `d_v=128`; **Tokamax / Pallas Splash Attention** | **11.40M** |
| **Fine-Grained DeepSeek MoE** | `416` routed (`d_ff=256`, top-12) + `1` shared (`d_ff=512`), **Tokamax Megablox `ragged_dot`**, Aux-Loss-Free per-loop bias | **658.31M** |
| **Total** | Active non-emb/loop: `40.17M` (`Attn:FFN = 1:1.93`) | **997.61M (~1.00B)** |

---

## 7-Stage Progressive Curriculum (21B Tokens Total)

Global batch size: **262,144 tokens/step** (`32,768` microbatch tokens × `8` grad accum steps).  
Optimizer: **Muon** (`lr=0.015`, 3D `vmap` per-expert Newton-Schulz `ns_steps=5`, Moonshot `consistent_rms=0.2`) + BF16 **AdamW** (`lr=1.5e-3`, `b2=0.95`) with **Warmup-Stable schedule (NO decay)**.

| Stage | Seq Len | Tokens | Loops | Web EN | Web ZH | Code | Math | Tiers |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 1K | 4B | 1 | 32% | 28% | 22% | 18% | Ultra-FineWeb-L3 + Code-L3 + Math-L3 |
| 2 | 2K | 4B | 1 | 28% | 24% | 25% | 23% | Ultra-FineWeb-L3 + Code-L3 + Math-L3 |
| 3 | 4K | 4B | 1 | 24% | 20% | 28% | 28% | Ultra-FineWeb-L3 + Code-L3 + Math-L3 |
| 4 | 8K | 4B | 1 | 20% | 16% | 36% | 28% | Ultra-FineWeb-L3 + Code-L3 + Math-L3 |
| 5 | 16K | 2B | 1 | 18% | 14% | 40% | 28% | Ultra-FineWeb-L3 + Code-L3 + Math-L3 |
| 6 | 32K | 2B | 1 | 16% | 14% | 42% | 28% | Ultra-FineWeb-L3 + Code-L3 + Math-L3 |
| 7 | 4K | 1B | **8** | 12% | 10% | 35% | 43% | Ultra-FineWeb-L3 + Code-L3 + Math-L3 |

---

## Quickstart

### 1. Local Setup & Verification
```bash
pip install -r requirements-local.txt
pytest -v tests/test_loopmoe.py
```

### 2. Build 64K Bilingual Tokenizer Locally (L3 Corpus)
```bash
python scripts/train_tokenizer.py \
  --target-gb 2.0 \
  --output-dir artifacts/tokenizer_64k
```

### 3. (Optional) Pre-pack `uint16` L3 Shards Locally
```bash
python scripts/prepare_shards.py \
  --tokenizer artifacts/tokenizer_64k/tokenizer.json \
  --out-dir artifacts/shards
```

### 4. Train on Google Colab TPU v6e-1 (Local Checkpoints)
Open [`notebooks/colab_tpu_v6e_train.ipynb`](notebooks/colab_tpu_v6e_train.ipynb) on a Colab **TPU v6e-1** runtime, or run:
```bash
python scripts/train_tpu.py \
  --data-mode stream \
  --tokenizer artifacts/tokenizer_64k/tokenizer.json \
  --ckpt-dir checkpoints \
  --keep-last-k 3
```
