"""OpenBMB UltraData bilingual streaming, uint16 shard packing, and background TPU prefetching."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import random
import threading
from typing import Iterator

import fsspec
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .config import SPECIAL_TOKEN_IDS, StageConfig


@dataclass(frozen=True)
class UltraDataSource:
    repo_id: str
    path_in_repo: str
    domain: str


SOURCE_REGISTRY: dict[tuple[str, str], UltraDataSource] = {
    ("web_en", "base"): UltraDataSource("openbmb/Ultra-FineWeb", "data/ultrafineweb_en", "web_en"),
    ("web_en", "L1"): UltraDataSource("openbmb/Ultra-FineWeb", "data/ultrafineweb_l1_en_hq", "web_en"),
    ("web_en", "L3"): UltraDataSource("openbmb/Ultra-FineWeb-L3", "data/ultrafineweb_en_l3", "web_en"),
    ("web_zh", "base"): UltraDataSource("openbmb/Ultra-FineWeb", "data/ultrafineweb_zh", "web_zh"),
    ("web_zh", "L1"): UltraDataSource("openbmb/Ultra-FineWeb", "data/ultrafineweb_zh", "web_zh"),
    ("web_zh", "L3"): UltraDataSource("openbmb/Ultra-FineWeb-L3", "data/ultrafineweb_zh_l3", "web_zh"),
    ("code", "L2"): UltraDataSource("openbmb/UltraData-Code", "data/UltraData-Code-L2", "code"),
    ("code", "L3"): UltraDataSource("openbmb/UltraData-Code", "data/UltraData-Code-L3", "code"),
    ("math", "L1"): UltraDataSource("openbmb/UltraData-Math", "data/UltraData-Math-L1", "math"),
    ("math", "L2"): UltraDataSource("openbmb/UltraData-Math", "data/UltraData-Math-L2-preview", "math"),
    ("math", "L3"): UltraDataSource("openbmb/UltraData-Math", "data/UltraData-Math-L3", "math"),
}


def list_parquet_files(
    source: UltraDataSource,
    hf_token: str | None = None,
    seed: int = 42,
) -> list[str]:
    """List and deterministically shuffle parquet files for an UltraData source."""
    api = HfApi(token=hf_token)
    entries = api.list_repo_tree(
        repo_id=source.repo_id,
        path_in_repo=source.path_in_repo,
        repo_type="dataset",
        recursive=True,
    )
    files = [e.path for e in entries if getattr(e, "path", "").endswith(".parquet")]
    rng = random.Random(seed)
    rng.shuffle(files)
    return files


def _extract_text_from_batch_row(row_dict: dict[str, object]) -> str | None:
    """Extract primary document string across UltraData schemas."""
    for key in ("full_content", "content", "text"):
        val = row_dict.get(key)
        if isinstance(val, str) and val.strip():
            # Prepend repo/path header for L2 code files when available for repo-level context
            repo = row_dict.get("repo_name")
            rel = row_dict.get("relative_path")
            if isinstance(rel, str) and rel:
                prefix = f"# {repo}/{rel}\n" if isinstance(repo, str) and repo else f"# {rel}\n"
                return prefix + val
            return val
    return None


def iter_source_texts(
    source: UltraDataSource,
    hf_token: str | None = None,
    seed: int = 42,
    min_chars: int = 64,
) -> Iterator[str]:
    """Stream text documents from OpenBMB UltraData parquet files endlessly (cyclic)."""
    files = list_parquet_files(source, hf_token=hf_token, seed=seed)
    if not files:
        raise RuntimeError(f"No parquet files found at {source.repo_id}:{source.path_in_repo}")

    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    while True:
        for rel_path in files:
            url = hf_hub_url(source.repo_id, rel_path, repo_type="dataset")
            try:
                with fsspec.open(url, "rb", headers=headers) as f:
                    pf = pq.ParquetFile(f)
                    available = set(pf.schema_arrow.names)
                    wanted = [
                        c
                        for c in ("full_content", "content", "text", "repo_name", "relative_path")
                        if c in available
                    ]
                    if not wanted:
                        continue
                    for batch in pf.iter_batches(batch_size=512, columns=wanted):
                        cols = {k: batch.column(k).to_pylist() for k in wanted}
                        text_keys = [k for k in ("full_content", "content", "text") if k in cols]
                        repos = cols.get("repo_name")
                        rels = cols.get("relative_path")
                        n_rows = len(cols[wanted[0]])
                        for i in range(n_rows):
                            val = None
                            for tk in text_keys:
                                cand = cols[tk][i]
                                if isinstance(cand, str) and cand.strip():
                                    val = cand
                                    break
                            if val is None:
                                continue
                            rel = rels[i] if rels is not None else None
                            if isinstance(rel, str) and rel:
                                repo = repos[i] if repos is not None else None
                                prefix = (
                                    f"# {repo}/{rel}\n"
                                    if isinstance(repo, str) and repo
                                    else f"# {rel}\n"
                                )
                                val = prefix + val
                            if len(val) >= min_chars:
                                yield val
            except Exception:
                # Skip transient network/corrupt parquet slice rather than crashing long runs
                continue


def iter_mixed_stage_texts(
    stage: StageConfig,
    seed: int = 42,
    hf_token: str | None = None,
) -> Iterator[str]:
    """Yield mixed bilingual web + code + math documents weighted by StageConfig ratios."""
    ratios = [
        ("web_en", stage.web_tier, stage.web_en_ratio),
        ("web_zh", stage.web_tier, stage.web_zh_ratio),
        ("code", stage.code_tier, stage.code_ratio),
        ("math", stage.math_tier, stage.math_ratio),
    ]
    active = [(dom, tier, w) for dom, tier, w in ratios if w > 0.0]
    weights = [w for _, _, w in active]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # Prefer longer documents for 16K/32K stages to expose real long-range dependencies
    min_chars = 512 if stage.seq_len >= 16384 else (256 if stage.seq_len >= 8192 else 64)

    streams = [
        iter_source_texts(
            SOURCE_REGISTRY[(dom, tier)],
            hf_token=hf_token,
            seed=seed + idx * 101 + stage.stage_id * 17,
            min_chars=min_chars,
        )
        for idx, (dom, tier, _) in enumerate(active)
    ]
    rng = random.Random(seed + stage.stage_id * 997)

    while True:
        choice = rng.choices(range(len(streams)), weights=probs, k=1)[0]
        yield next(streams[choice])


def pack_documents_to_shards(
    text_iter: Iterator[str],
    tokenizer: Tokenizer,
    out_dir: str | Path,
    stage: StageConfig,
    tokens_per_shard: int = 50_000_000,
    max_tokens: int | None = None,
    hf_dataset_repo: str | None = None,
    hf_token: str | None = None,
    delete_after_upload: bool = False,
) -> list[str]:
    """Tokenize, pack with <|eos|> delimiters into uint16 .bin shards, and optionally upload to HF Hub."""
    stage_dir = Path(out_dir) / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    eos_id = SPECIAL_TOKEN_IDS["<|eos|>"]
    target_total = max_tokens if max_tokens is not None else stage.target_tokens

    api = HfApi(token=hf_token) if hf_dataset_repo else None
    if api is not None and hf_dataset_repo:
        api.create_repo(repo_id=hf_dataset_repo, repo_type="dataset", exist_ok=True)

    shard_names: list[str] = []
    buffer: list[int] = []
    batch_texts: list[str] = []
    total_written = 0
    shard_idx = 0

    def _flush_shard(tokens_arr: np.ndarray, idx: int) -> str:
        name = f"shard_{idx:05d}.bin"
        path = stage_dir / name
        tokens_arr.astype(np.uint16).tofile(path)
        if api is not None and hf_dataset_repo:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{stage.name}/{name}",
                repo_id=hf_dataset_repo,
                repo_type="dataset",
                commit_message=f"Add {stage.name}/{name} ({len(tokens_arr):,} tokens)",
            )
            if delete_after_upload:
                path.unlink(missing_ok=True)
        return name

    while total_written < target_total:
        batch_texts.clear()
        for _ in range(256):
            batch_texts.append(next(text_iter))
        encodings = tokenizer.encode_batch(batch_texts)
        for enc in encodings:
            buffer.extend(enc.ids)
            buffer.append(eos_id)
            while len(buffer) >= tokens_per_shard and total_written < target_total:
                take = min(tokens_per_shard, target_total - total_written)
                arr = np.asarray(buffer[:take], dtype=np.uint16)
                del buffer[:take]
                shard_names.append(_flush_shard(arr, shard_idx))
                total_written += len(arr)
                shard_idx += 1
                if total_written >= target_total:
                    break
            if total_written >= target_total:
                break

    # Flush remainder if non-empty and under target
    if buffer and total_written < target_total:
        take = min(len(buffer), target_total - total_written)
        arr = np.asarray(buffer[:take], dtype=np.uint16)
        shard_names.append(_flush_shard(arr, shard_idx))
        total_written += len(arr)

    manifest = {
        "stage": stage.name,
        "seq_len": stage.seq_len,
        "total_tokens": total_written,
        "shards": shard_names,
    }
    manifest_path = stage_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if api is not None and hf_dataset_repo:
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"{stage.name}/manifest.json",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            commit_message=f"Update {stage.name} manifest ({total_written:,} tokens)",
        )
    return shard_names


class ShardBatchPrefetcher:
    """Background-threaded uint16 packed shard reader feeding TPU batches without CPU stalls."""

    def __init__(
        self,
        stage: StageConfig,
        shard_dir: str | Path,
        hf_dataset_repo: str | None = None,
        hf_token: str | None = None,
        seed: int = 42,
        queue_size: int = 4,
        microbatch_tokens: int | None = None,
    ):
        self.stage = stage
        self.shard_dir = Path(shard_dir) / stage.name
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.hf_dataset_repo = hf_dataset_repo
        self.hf_token = hf_token
        self.seed = seed
        self.micro_tok = microbatch_tokens or stage.microbatch_tokens
        self.micro_bs = self.micro_tok // stage.seq_len
        self.seq_len = stage.seq_len
        self._queue: queue.Queue[tuple[np.ndarray, np.ndarray] | BaseException] = queue.Queue(
            maxsize=queue_size
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _discover_shards(self) -> list[str]:
        local_bins = sorted(p.name for p in self.shard_dir.glob("*.bin"))
        if local_bins:
            return local_bins
        if not self.hf_dataset_repo:
            raise FileNotFoundError(f"No .bin shards found in {self.shard_dir}")
        manifest_path = hf_hub_download(
            repo_id=self.hf_dataset_repo,
            filename=f"{self.stage.name}/manifest.json",
            repo_type="dataset",
            local_dir=str(self.shard_dir.parent),
            token=self.hf_token,
        )
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return list(manifest["shards"])

    def _load_shard(self, shard_name: str) -> np.ndarray:
        local_path = self.shard_dir / shard_name
        if not local_path.exists() and self.hf_dataset_repo:
            downloaded = hf_hub_download(
                repo_id=self.hf_dataset_repo,
                filename=f"{self.stage.name}/{shard_name}",
                repo_type="dataset",
                local_dir=str(self.shard_dir.parent),
                token=self.hf_token,
            )
            local_path = Path(downloaded)
        return np.memmap(local_path, dtype=np.uint16, mode="r")

    def _worker(self) -> None:
        try:
            shards = self._discover_shards()
            rng = np.random.default_rng(self.seed + self.stage.stage_id * 31)
            carry = np.empty((0,), dtype=np.uint16)

            while not self._stop.is_set():
                order = rng.permutation(len(shards))
                for s_idx in order:
                    if self._stop.is_set():
                        return
                    data = np.asarray(self._load_shard(shards[int(s_idx)]), dtype=np.uint16)
                    if len(carry) > 0:
                        data = np.concatenate([carry, data])

                    usable = ((len(data) - 1) // (self.micro_bs * self.seq_len)) * (
                        self.micro_bs * self.seq_len
                    )
                    if usable < self.micro_bs * self.seq_len:
                        carry = data
                        continue

                    carry = data[usable:]
                    # Packed sequence view: [num_batches, micro_bs, seq_len]
                    n_batches = usable // (self.micro_bs * self.seq_len)
                    x_all = data[:usable].reshape(n_batches, self.micro_bs, self.seq_len)
                    y_all = data[1 : usable + 1].reshape(n_batches, self.micro_bs, self.seq_len)

                    batch_perm = rng.permutation(n_batches)
                    for b_i in batch_perm:
                        if self._stop.is_set():
                            return
                        item = (
                            np.array(x_all[b_i], dtype=np.uint16, copy=True),
                            np.array(y_all[b_i], dtype=np.uint16, copy=True),
                        )
                        while not self._stop.is_set():
                            try:
                                self._queue.put(item, timeout=0.5)
                                break
                            except queue.Full:
                                continue
        except BaseException as exc:  # pragma: no cover - surfaced to consumer
            self._queue.put(exc)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class StreamingTokenPrefetcher:
    """Direct on-the-fly streaming tokenizer + packer for Colab runs without pre-packed shards."""

    def __init__(
        self,
        stage: StageConfig,
        tokenizer: Tokenizer,
        hf_token: str | None = None,
        seed: int = 42,
        queue_size: int = 4,
        microbatch_tokens: int | None = None,
    ):
        self.stage = stage
        self.tokenizer = tokenizer
        self.hf_token = hf_token
        self.seed = seed
        self.micro_tok = microbatch_tokens or stage.microbatch_tokens
        self.micro_bs = self.micro_tok // stage.seq_len
        self.seq_len = stage.seq_len
        self._queue: queue.Queue[tuple[np.ndarray, np.ndarray] | BaseException] = queue.Queue(
            maxsize=queue_size
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            eos_id = SPECIAL_TOKEN_IDS["<|eos|>"]
            text_iter = iter_mixed_stage_texts(self.stage, seed=self.seed, hf_token=self.hf_token)
            buf: list[int] = []
            need = self.micro_bs * self.seq_len + 1

            while not self._stop.is_set():
                while len(buf) < need:
                    batch_texts = [next(text_iter) for _ in range(256)]
                    for enc in self.tokenizer.encode_batch(batch_texts):
                        buf.extend(enc.ids)
                        buf.append(eos_id)
                arr = np.asarray(buf[:need], dtype=np.uint16)
                del buf[: self.micro_bs * self.seq_len]
                x = arr[:-1].reshape(self.micro_bs, self.seq_len)
                y = arr[1:].reshape(self.micro_bs, self.seq_len)
                while not self._stop.is_set():
                    try:
                        self._queue.put((x, y), timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # pragma: no cover
            self._queue.put(exc)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
