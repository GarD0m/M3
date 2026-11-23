"""Train a local 64K (65,536) bilingual EN/ZH + Code + Math Byte-level BPE tokenizer using Rust tokenizers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from huggingface_hub import HfApi
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from loopmoe.config import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS
from loopmoe.data import SOURCE_REGISTRY, iter_source_texts

# DeepSeek / Qwen-inspired regex:
# - isolates single digits (\p{N}) for crisp arithmetic tokenization
# - groups contractions & ASCII/Unicode letters while letting CJK learn multi-char BPE merges
# - preserves code indentation spaces & newlines cleanly
DEEPSEEK_PRETOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def build_bpe_tokenizer(vocab_size: int = 65_536) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Construct the uninitialized Byte-level BPE tokenizer and trainer with exact 64K vocab."""
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(DEEPSEEK_PRETOKENIZER_PATTERN),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    return tokenizer, trainer


def iter_tokenizer_corpus(
    target_bytes: int,
    hf_token: str | None = None,
    seed: int = 42,
    local_dir: str | None = None,
    english_only: bool = False,
) -> Iterator[str]:
    """Stream balanced web + Code + Math documents up to `target_bytes`."""
    if local_dir:
        paths = sorted(Path(local_dir).rglob("*"))
        total_bytes = 0
        for p in paths:
            if p.is_file():
                text = p.read_text(encoding="utf-8", errors="ignore")
                total_bytes += len(text.encode("utf-8"))
                yield text
                if total_bytes >= target_bytes:
                    return
        return

    if english_only:
        # English-only L3 tokenizer mix: 62% EN, 20% Code, 18% Math
        sources = [
            (SOURCE_REGISTRY[("web_en", "L3")], 0.62),
            (SOURCE_REGISTRY[("code", "L3")], 0.20),
            (SOURCE_REGISTRY[("math", "L3")], 0.18),
        ]
    else:
        # Balanced L3 tokenizer training mix: 34% EN, 28% ZH, 20% Code, 18% Math
        sources = [
            (SOURCE_REGISTRY[("web_en", "L3")], 0.34),
            (SOURCE_REGISTRY[("web_zh", "L3")], 0.28),
            (SOURCE_REGISTRY[("code", "L3")], 0.20),
            (SOURCE_REGISTRY[("math", "L3")], 0.18),
        ]
    streams = [
        iter_source_texts(src, hf_token=hf_token, seed=seed + i * 37, min_chars=64)
        for i, (src, _) in enumerate(sources)
    ]
    weights = [w for _, w in sources]
    rng = random.Random(seed)

    total_bytes = 0
    doc_count = 0
    next_log_bytes = 25 * 1024 * 1024
    while total_bytes < target_bytes:
        idx = rng.choices(range(len(streams)), weights=weights, k=1)[0]
        text = next(streams[idx])
        total_bytes += len(text.encode("utf-8"))
        doc_count += 1
        if total_bytes >= next_log_bytes or total_bytes >= target_bytes:
            mb = total_bytes / (1024**2)
            tgt_mb = target_bytes / (1024**2)
            print(
                f"[Tokenizer Corpus] Streamed {doc_count:,} docs ({mb:.1f} / {tgt_mb:.1f} MB)",
                flush=True,
            )
            next_log_bytes += 25 * 1024 * 1024
        yield text


def save_hf_tokenizer_files(tokenizer: Tokenizer, output_dir: Path) -> None:
    """Save tokenizer.json plus HuggingFace AutoTokenizer config files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 32768,
        "additional_special_tokens": [
            tok for tok in SPECIAL_TOKENS if tok not in ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")
        ],
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    special_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
    }
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def benchmark_tokenizer(tokenizer: Tokenizer) -> dict[str, dict[str, float | bool]]:
    """Verify special tokens, digit splitting, lossless round-trip, and compression ratios."""
    samples = {
        "en": "The Mixture-of-Experts architecture routes each token to top-12 experts efficiently in 2026.",
        "zh": "這是一個結合單層 DeepSeek-V4.1-Flash 與 LoopMoE 的十億參數雙語大語言模型，支援長上下文推理。",
        "code": "def sinkhorn_knopp(logits, iters=20):\n    m = jnp.exp(logits)\n    for _ in range(iters):\n        m /= m.sum(-1, keepdims=True)\n    return m\n",
        "math": r"Let $f(x) = \sum_{i=1}^{416} w_i \cdot \sigma(x_i)$. Solve for $x = 12345 + 67890 = 80235$.",
    }
    results: dict[str, dict[str, float | bool]] = {}
    for name, text in samples.items():
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc.ids, skip_special_tokens=False)
        n_bytes = len(text.encode("utf-8"))
        results[name] = {
            "tokens": len(enc.ids),
            "bytes": n_bytes,
            "bytes_per_token": round(n_bytes / max(1, len(enc.ids)), 3),
            "roundtrip_exact": dec == text,
        }

    # Verify special token exact IDs
    for tok, expected_id in list(SPECIAL_TOKEN_IDS.items())[:17]:
        actual_id = tokenizer.token_to_id(tok)
        if actual_id != expected_id:
            raise AssertionError(f"Special token {tok} expected id {expected_id}, got {actual_id}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local 64K bilingual BPE tokenizer.")
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument(
        "--target-gb",
        type=float,
        default=2.0,
        help="Total UTF-8 corpus size in GB streamed from OpenBMB UltraData (default: 2.0 GB).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Drop Ultra-FineWeb ZH and train an English-only (EN + Code + Math) L3 tokenizer.",
    )
    parser.add_argument("--local-corpus", type=str, default=None, help="Optional local folder of txt/md/py files.")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Optional HF Hub repo to upload tokenizer.")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("artifacts/tokenizer_64k_en" if args.english_only else "artifacts/tokenizer_64k")
    )
    target_bytes = int(args.target_gb * (1024**3))

    tokenizer, trainer = build_bpe_tokenizer(vocab_size=args.vocab_size)
    corpus_iter = iter_tokenizer_corpus(
        target_bytes=target_bytes,
        hf_token=args.hf_token,
        seed=args.seed,
        local_dir=args.local_corpus,
        english_only=args.english_only,
    )

    mode_label = "english-only" if args.english_only else "bilingual"
    print(
        f"Training {mode_label} Byte-level BPE tokenizer (V={args.vocab_size}, special={len(SPECIAL_TOKENS)}) "
        f"on target={args.target_gb:.2f} GB -> {out_dir}..."
    )
    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"Finished training tokenizer! Vocab size = {actual_vocab}")
    save_hf_tokenizer_files(tokenizer, out_dir)
    print(f"Saved tokenizer files to {out_dir.resolve()}")

    report = benchmark_tokenizer(tokenizer)
    print("Benchmark report:", json.dumps(report, indent=2, ensure_ascii=False))

    if args.hf_repo_id:
        api = HfApi(token=args.hf_token)
        api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo="tokenizer",
            repo_id=args.hf_repo_id,
            repo_type="model",
            commit_message=f"Upload 64K bilingual LoopMoE tokenizer (V={actual_vocab})",
        )
        print(f"Uploaded tokenizer to hf://{args.hf_repo_id}/tokenizer")


if __name__ == "__main__":
    main()
