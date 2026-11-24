"""Resumable checkpoint serialization and async Hugging Face Hub synchronization."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Any

from flax import serialization
from huggingface_hub import HfApi, snapshot_download
import jax

from .config import LoopMoEConfig
from .optim import OptimConfig

_CKPT_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _ensure_host_arrays(tree: Any) -> Any:
    """Transfer JAX arrays to host numpy/array values safely before serialization."""
    return jax.device_get(tree)


def save_checkpoint(
    ckpt_root: str | Path,
    stage_id: int,
    step: int,
    tokens_seen: int,
    params: Any,
    opt_state: Any,
    cfg: LoopMoEConfig,
    optim_cfg: OptimConfig,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    keep_last_k: int = 3,
    async_upload: bool = True,
    async_write: bool = False,
) -> tuple[Path, Future | None]:
    """Save model params, optimizer state, and stage metadata locally and optionally push to HF Hub."""
    root = Path(ckpt_root)
    ckpt_name = f"stage{stage_id}_step{step:08d}"
    ckpt_dir = root / ckpt_name

    # Synchronously copy device arrays to host before the training loop can donate/mutate them
    host_params = _ensure_host_arrays(params)
    host_opt_state = _ensure_host_arrays(opt_state)
    meta = {
        "stage_id": stage_id,
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": cfg.to_dict(),
        "optim_config": asdict(optim_cfg),
    }

    def _write_and_maybe_upload(run_upload_async: bool) -> Future | None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Save params & opt_state separately so R=1 -> R=8 expansion can reload params cleanly without opt_state
        params_bytes = serialization.to_bytes(host_params)
        (ckpt_dir / "params.msgpack").write_bytes(params_bytes)

        opt_bytes = serialization.to_bytes(host_opt_state)
        (ckpt_dir / "opt_state.msgpack").write_bytes(opt_bytes)

        # Atomic latest pointer
        latest_meta = {"latest": ckpt_name, **meta}
        latest_tmp = root / "latest.json.tmp"
        latest_tmp.write_text(json.dumps(latest_meta, indent=2), encoding="utf-8")
        os.replace(latest_tmp, root / "latest.json")

        # Prune older local checkpoints
        existing = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith("stage")],
            key=lambda p: p.stat().st_mtime,
        )
        if keep_last_k > 0 and len(existing) > keep_last_k:
            for old_dir in existing[:-keep_last_k]:
                shutil.rmtree(old_dir, ignore_errors=True)

        upload_future: Future | None = None
        if hf_repo_id:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
            # Upload latest checkpoint directory + latest.json pointer
            upload_future = api.upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"checkpoints/{ckpt_name}",
                repo_id=hf_repo_id,
                repo_type="model",
                commit_message=f"Checkpoint {ckpt_name} (tokens={tokens_seen:,})",
                run_as_future=run_upload_async,
            )
            api.upload_file(
                path_or_fileobj=str(root / "latest.json"),
                path_in_repo="checkpoints/latest.json",
                repo_id=hf_repo_id,
                repo_type="model",
                commit_message=f"Update latest pointer -> {ckpt_name}",
                run_as_future=False,
            )
        return upload_future

    if async_write:
        write_future = _CKPT_EXECUTOR.submit(_write_and_maybe_upload, False)
        return ckpt_dir, write_future

    upload_future = _write_and_maybe_upload(async_upload)
    return ckpt_dir, upload_future


def _find_local_latest(root: Path) -> Path | None:
    latest_file = root / "latest.json"
    if latest_file.exists():
        info = json.loads(latest_file.read_text(encoding="utf-8"))
        cand = root / info["latest"]
        if cand.exists():
            return cand
    dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("stage")],
        key=lambda p: p.name,
    )
    return dirs[-1] if dirs else None


def load_latest_checkpoint(
    ckpt_root: str | Path,
    params_template: Any,
    opt_state_template: Any | None = None,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
) -> tuple[Any, Any | None, dict[str, Any]] | None:
    """Load the latest local or HF Hub checkpoint. Returns (params, opt_state, metadata) or None."""
    root = Path(ckpt_root)
    root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _find_local_latest(root)

    if ckpt_dir is None and hf_repo_id:
        api = HfApi(token=hf_token)
        try:
            latest_ptr = api.hf_hub_download(
                repo_id=hf_repo_id,
                filename="checkpoints/latest.json",
                repo_type="model",
                local_dir=str(root / "_hf_tmp"),
            )
            info = json.loads(Path(latest_ptr).read_text(encoding="utf-8"))
            ckpt_name = info["latest"]
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="model",
                allow_patterns=[f"checkpoints/{ckpt_name}/*", "checkpoints/latest.json"],
                local_dir=str(root / "_hf_dl"),
                token=hf_token,
            )
            dl_ckpt = root / "_hf_dl" / "checkpoints" / ckpt_name
            if dl_ckpt.exists():
                target = root / ckpt_name
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(dl_ckpt), str(target))
                shutil.copy2(str(root / "_hf_dl" / "checkpoints" / "latest.json"), str(root / "latest.json"))
                ckpt_dir = target
        except Exception:
            return None

    if ckpt_dir is None or not (ckpt_dir / "params.msgpack").exists():
        return None

    meta = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
    params = serialization.from_bytes(params_template, (ckpt_dir / "params.msgpack").read_bytes())

    opt_state = None
    if opt_state_template is not None and (ckpt_dir / "opt_state.msgpack").exists():
        opt_state = serialization.from_bytes(
            opt_state_template, (ckpt_dir / "opt_state.msgpack").read_bytes()
        )
    return params, opt_state, meta
