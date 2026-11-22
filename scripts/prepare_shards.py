"""Tokenize and pack OpenBMB UltraData stages into uint16 shards and push to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tokenizers import Tokenizer

from loopmoe.config import DEFAULT_CURRICULUM
from loopmoe.data import iter_mixed_stage_texts, pack_documents_to_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack OpenBMB UltraData stages into uint16 HF shards.")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer_64k/tokenizer.json")
    parser.add_argument("--out-dir", type=str, default="artifacts/shards")
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[s.stage_id for s in DEFAULT_CURRICULUM],
        help="Stage IDs (1..7) to prepare.",
    )
    parser.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    parser.add_argument("--max-tokens-override", type=int, default=None, help="Optional token cap per stage for testing.")
    parser.add_argument("--hf-dataset-repo", type=str, default=None, help="HF dataset repo ID to upload shards.")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--delete-after-upload", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    stage_map = {s.stage_id: s for s in DEFAULT_CURRICULUM}

    for sid in args.stages:
        if sid not in stage_map:
            raise ValueError(f"Unknown stage id: {sid}")
        stage = stage_map[sid]
        target = args.max_tokens_override or stage.target_tokens
        print(
            f"\n=== Preparing {stage.name} (seq_len={stage.seq_len}, target={target:,} tokens) ===\n"
            f"Mix: EN={stage.web_en_ratio:.2f} ZH={stage.web_zh_ratio:.2f} "
            f"Code={stage.code_ratio:.2f} Math={stage.math_ratio:.2f}"
        )
        text_iter = iter_mixed_stage_texts(stage, seed=args.seed, hf_token=args.hf_token)
        shards = pack_documents_to_shards(
            text_iter=text_iter,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            stage=stage,
            tokens_per_shard=args.tokens_per_shard,
            max_tokens=target,
            hf_dataset_repo=args.hf_dataset_repo,
            hf_token=args.hf_token,
            delete_after_upload=args.delete_after_upload,
        )
        print(f"Completed {stage.name}: {len(shards)} shards.")


if __name__ == "__main__":
    main()
