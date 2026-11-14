"""Streaming SHA256 atomic checkpoint saving and deterministic FSDP resume for ResWAM."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)

from ..utils.dist import restore_rng, rng_state

SCHEMA = "reswam-checkpoint-v1"
CHECKPOINT_SYNC_TIMEOUT = datetime.timedelta(hours=2)
PIPE_BYTES = 1 << 20
ALIAS_ROOT = "/dev/shm" if os.path.isdir("/dev/shm") else None
_SYNC_GROUP: Any = None

_WRITER = r"""
import hashlib, json, os, sys
digest, size, error = hashlib.sha256(), 0, None
with os.fdopen(int(sys.argv[1]), "rb", buffering=0) as source:
    try:
        target = open(sys.argv[2], "wb")
    except OSError as caught:
        target, error = None, caught
    while chunk := source.read(1 << 22):
        digest.update(chunk)
        size += len(chunk)
        if error is None:
            try:
                target.write(chunk)
            except OSError as caught:
                error = caught
    if target is not None:
        try:
            target.close()
        except OSError as caught:
            error = error or caught
if error is not None:
    sys.exit(f"{type(error).__name__}: {error}")
print(json.dumps({"size": size, "sha256": digest.hexdigest()}))
"""


def init_sync_group(timeout: datetime.timedelta = CHECKPOINT_SYNC_TIMEOUT) -> Any:
    global _SYNC_GROUP
    if _SYNC_GROUP is None and dist.is_initialized():
        _SYNC_GROUP = dist.new_group(backend="gloo", timeout=timeout)
    return _SYNC_GROUP


def _sync_group() -> Any:
    return init_sync_group()


def save_hashed(obj: Any, target: Path, *, archive_name: str | None = None) -> dict[str, Any]:
    """Stream ``torch.save(obj, ...)`` to ``target`` while computing size and SHA256 in flight."""
    target = Path(target)
    read_end, write_end = os.pipe()
    try:
        import fcntl

        fcntl.fcntl(write_end, 1031, PIPE_BYTES)
    except OSError:
        pass
    writer = subprocess.Popen(
        [sys.executable, "-I", "-c", _WRITER, str(read_end), str(target)],
        pass_fds=(read_end,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(read_end)
    try:
        with tempfile.TemporaryDirectory(prefix="reswam-save-", dir=ALIAS_ROOT) as directory:
            alias = Path(directory) / (archive_name or target.name)
            alias.symlink_to(f"/proc/{os.getpid()}/fd/{write_end}")
            torch.save(obj, alias)
    except BaseException:
        os.close(write_end)
        writer.kill()
        writer.communicate()
        raise
    os.close(write_end)
    out, err = writer.communicate()
    if writer.returncode != 0:
        raise RuntimeError(f"checkpoint writer failed for {target}: {err.strip()[-800:]}")
    written = json.loads(out)
    if target.stat().st_size != written["size"]:
        raise RuntimeError(f"checkpoint file size differs from bytes written: {target}")
    return {"size": int(written["size"]), "sha256": str(written["sha256"])}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}


def _all_ok(stage: str, error: str | None, world_size: int) -> None:
    if not dist.is_initialized() or world_size <= 1:
        if error:
            raise RuntimeError(f"{stage} failed: {error}")
        return
    errors: list[str | None] = [None] * world_size
    dist.all_gather_object(errors, error, group=_sync_group())
    if any(errors):
        raise RuntimeError(f"{stage} failed: {[e for e in errors if e]}")


def save(
    wrapped: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    output: Path,
    step: int,
    data_position: dict[str, int],
    codec: dict[str, Any],
    config: dict[str, Any],
    rank: int,
    world_size: int,
    full: bool,
    keep_resume_states: int,
) -> None:
    root = output / "checkpoints"
    temporary = root / f".step-{step:08d}.incomplete"
    final = root / f"step-{step:08d}.sharded"
    setup_error = None
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            committed_step = latest_step(output)
            if final.exists():
                if committed_step == step:
                    raise RuntimeError(f"{final} already exists")
                shutil.rmtree(final, ignore_errors=True)
                for orphan in (
                    root / f"step-{step:08d}.full.pt",
                    root / f".step-{step:08d}.full.pt.tmp",
                    root / f"step-{step:08d}.manifest.json",
                    root / f".step-{step:08d}.manifest.tmp",
                ):
                    orphan.unlink(missing_ok=True)
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir()
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"
    _all_ok("checkpoint setup", setup_error, world_size)
    with FSDP.state_dict_type(
        wrapped,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        model = wrapped.state_dict()
        optim = FSDP.sharded_optim_state_dict(wrapped, optimizer)
    payload = {
        "schema_version": SCHEMA,
        "global_optimizer_step": step,
        "model": model,
        "optimizer": optim,
        "scheduler": scheduler.state_dict(),
        "scaler": None,
        "rng": rng_state(),
        "data_position": data_position,
        "codec": codec,
        "world_size": world_size,
        "complete": True,
    }
    error = None
    shard_identity = None
    try:
        shard_name = f"rank-{rank:05d}.pt"
        shard_meta = save_hashed(payload, temporary / shard_name, archive_name=shard_name)
        shard_identity = {"path": str(final / shard_name), **shard_meta}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    del payload, model, optim
    if dist.is_initialized() and world_size > 1:
        gathered_shards: list[Any] = [None] * world_size
        dist.all_gather_object(gathered_shards, (error, shard_identity), group=_sync_group())
    else:
        gathered_shards = [(error, shard_identity)]
    shard_errors = [err for err, _ in gathered_shards if err]
    if shard_errors:
        raise RuntimeError(f"resume-state save failed: {shard_errors}")
    shard_identities = [ident for _, ident in gathered_shards]
    full_model = None
    if full:
        extract_error = None
        try:
            with FSDP.state_dict_type(
                wrapped,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                full_model = wrapped.state_dict()
        except Exception as exc:
            extract_error = f"{type(exc).__name__}: {exc}"
        _all_ok("full-state extract", extract_error, world_size)
    error = None
    if rank == 0:
        try:
            os.replace(temporary, final)
            for ident in shard_identities:
                if Path(ident["path"]).stat().st_size != ident["size"]:
                    raise RuntimeError(f"shard size mismatch after rename: {ident['path']}")
            latest: dict[str, Any] = {
                "schema_version": SCHEMA,
                "step": step,
                "world_size": world_size,
                "shards": shard_identities,
                "data_position": data_position,
                "complete": True,
            }
            if full:
                path = root / f"step-{step:08d}.full.pt"
                tmp_full = root / f".step-{step:08d}.full.pt.tmp"
                tmp_full.unlink(missing_ok=True)
                full_meta = save_hashed(
                    {
                        "schema_version": SCHEMA,
                        "global_optimizer_step": step,
                        "model": full_model,
                        "codec": codec,
                        "config": config,
                        "complete": True,
                    },
                    tmp_full,
                    archive_name=path.name,
                )
                os.replace(tmp_full, path)
                full_identity = {"path": str(path), **full_meta}
                latest["full"] = full_identity
                manifest_path = root / f"step-{step:08d}.manifest.json"
                manifest_payload = {
                    "schema_version": SCHEMA,
                    "step": step,
                    "world_size": world_size,
                    "full": full_identity,
                    "shards": latest["shards"],
                    "data_position": data_position,
                    "codec_sha256": codec.get("contract_sha256"),
                    "complete": True,
                }
                manifest_tmp = root / f".step-{step:08d}.manifest.tmp"
                manifest_tmp.write_text(json.dumps(manifest_payload, indent=1, sort_keys=True) + "\n")
                os.replace(manifest_tmp, manifest_path)
                latest["manifest"] = _identity(manifest_path)
            tmp = root / ".latest.tmp"
            tmp.write_text(json.dumps(latest, indent=1) + "\n")
            os.replace(tmp, root / "latest.json")
            removed = prune_resume_states(root, keep_resume_states)
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "step": step,
                        "full": full,
                        "pruned_resume_states": removed,
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    del full_model
    _all_ok("checkpoint publish", error, world_size)


def prune_resume_states(root: Path, keep: int) -> list[int]:
    if keep <= 0:
        return []
    steps = sorted(
        int(p.name.split("-")[1].split(".")[0]) for p in root.glob("step-*.sharded") if p.is_dir()
    )
    latest = json.loads((root / "latest.json").read_text())["step"]
    removed = []
    for step in steps[:-keep]:
        if step == latest:
            continue
        shutil.rmtree(root / f"step-{step:08d}.sharded")
        removed.append(step)
    return removed


def latest_step(output: Path) -> int | None:
    path = output / "checkpoints/latest.json"
    return json.loads(path.read_text())["step"] if path.is_file() else None


def load(
    wrapped: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    output: Path,
    codec: dict[str, Any],
    rank: int,
    world_size: int,
) -> tuple[int, dict[str, int]]:
    error, payload = None, None
    try:
        latest = json.loads((output / "checkpoints/latest.json").read_text())
        if latest["world_size"] != world_size:
            raise RuntimeError(
                f"resume state has {latest['world_size']} ranks, run has {world_size}"
            )
        row = latest["shards"][rank]
        if _identity(Path(row["path"])) != row:
            raise RuntimeError("resume shard hash mismatch")
        payload = torch.load(row["path"], map_location="cpu", weights_only=False)
        if (
            payload.get("complete") is not True
            or payload["global_optimizer_step"] != latest["step"]
            or payload["scheduler"]["step_count"] != latest["step"]
        ):
            raise RuntimeError("incomplete or inconsistent resume payload")
        if payload["codec"]["contract_sha256"] != codec["contract_sha256"]:
            raise RuntimeError("codec of the resume state differs from this run's codec")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _all_ok("resume read", error, world_size)
    with FSDP.state_dict_type(
        wrapped,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        wrapped.load_state_dict(payload["model"], strict=True)
        optim = FSDP.optim_state_dict_to_load(wrapped, optimizer, payload["optimizer"])
    optimizer.load_state_dict(optim)
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return int(payload["global_optimizer_step"]), dict(payload["data_position"])
