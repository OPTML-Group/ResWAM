"""Memory-mapped FP32 cache of frozen DINOv3 camera features for ResWAM datasets."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

CACHE_SCHEMA = "reswam-dino-view-feature-cache-v1"
TOKENS, WIDTH = 64, 768
ROW_BYTES = TOKENS * WIDTH * 4
MANIFEST_NAME = "DINO_CACHE_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layout_sha256(episodes: Sequence[dict[str, Any]], rows: Sequence[int]) -> str:
    """SHA256 of [[episode_index, begin, length], ...] in dataset order."""
    payload = [
        [int(ep["episode_index"]), int(begin), int(ep["length"])]
        for ep, begin in zip(episodes, rows, strict=True)
    ]
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class ViewDinoCache:
    """Memory-mapped row reader for a single camera view; opened lazily per DataLoader worker."""

    def __init__(self, root: Path, manifest: dict[str, Any], manifest_sha256: str) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.view = str(manifest["view"])
        self.frames = int(manifest["frames"])
        self.bounds = [(int(row["begin"]), int(row["end"])) for row in manifest["shards"]]
        self.files = [self.root / row["file"] for row in manifest["shards"]]
        self._maps: list[np.memmap | None] = [None] * len(self.files)
        self._pid: int | None = None

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        view: str,
        episodes: Sequence[dict[str, Any]],
        rows: Sequence[int],
        expected_manifest_sha256: str | None = None,
        full_hash: bool = False,
    ) -> "ViewDinoCache":
        manifest_path = Path(root) / MANIFEST_NAME
        if not manifest_path.is_file():
            raise RuntimeError(f"DINO cache manifest missing: {manifest_path}")
        digest = sha256_file(manifest_path)
        if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
            raise RuntimeError(
                f"DINO cache manifest sha256 mismatch for {view}: {digest} != {expected_manifest_sha256}"
            )
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "schema_version": CACHE_SCHEMA,
            "complete": True,
            "view": str(view),
            "dtype": "float32",
            "row_shape": [TOKENS, WIDTH],
            "frames": sum(int(ep["length"]) for ep in episodes),
            "episode_layout_sha256": layout_sha256(episodes, rows),
        }
        if {key: manifest.get(key) for key in expected} != expected:
            raise RuntimeError(
                f"DINO cache manifest mismatch for {view}: got { {k: manifest.get(k) for k in expected} }, "
                f"expected {expected}"
            )
        cache = cls(root, manifest, digest)
        cache.verify(full_hash=full_hash)
        return cache

    def verify(self, *, full_hash: bool = False) -> None:
        begin = 0
        for row, path, (low, high) in zip(self.manifest["shards"], self.files, self.bounds, strict=True):
            if low != begin or high <= low:
                raise RuntimeError(f"DINO cache shards not contiguous in {self.root}")
            begin = high
            info = path.stat()
            if info.st_size != (high - low) * ROW_BYTES or info.st_size != int(row["size"]):
                raise RuntimeError(f"DINO cache shard size mismatch: {path}")
            if stat.S_IMODE(info.st_mode) & 0o222:
                raise RuntimeError(f"DINO cache shard is writable (expected read-only): {path}")
            if full_hash and sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"DINO cache shard sha256 mismatch: {path}")
        if begin != self.frames:
            raise RuntimeError(f"DINO cache does not cover all {self.frames} frames in {self.root}")

    def _map(self, shard: int) -> np.memmap:
        if self._pid != os.getpid():
            self._maps, self._pid = [None] * len(self.files), os.getpid()
        if self._maps[shard] is None:
            low, high = self.bounds[shard]
            self._maps[shard] = np.memmap(
                self.files[shard], dtype="<f4", mode="r", shape=(high - low, TOKENS, WIDTH)
            )
        return self._maps[shard]

    def row(self, index: int) -> torch.Tensor:
        index = int(index)
        if not 0 <= index < self.frames:
            raise IndexError(index)
        for shard, (low, high) in enumerate(self.bounds):
            if low <= index < high:
                return torch.from_numpy(np.array(self._map(shard)[index - low], dtype=np.float32))
        raise IndexError(index)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_maps"], state["_pid"] = [None] * len(self.files), None
        return state


def float_view(tensor: torch.Tensor) -> torch.Tensor:
    """Carry FP32 payload as int32 bits so FSDP bf16 input casting leaves it bit-exact."""
    if tensor.dtype != torch.float32:
        raise RuntimeError("only float32 tensors are carried bit-exactly")
    return tensor.contiguous().view(torch.int32)


def unview(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor
    if tensor.dtype != torch.int32:
        raise RuntimeError("bit-exact float payload must arrive as int32 or float32")
    return tensor.view(torch.float32)


def open_world_caches(
    cache_root: str | Path,
    *,
    world_views: Sequence[str],
    episodes: Sequence[dict[str, Any]],
    rows: Sequence[int],
    expected_manifests: dict[str, str] | None = None,
    full_hash: bool = False,
) -> dict[str, ViewDinoCache]:
    root = Path(cache_root)
    caches: dict[str, ViewDinoCache] = {}
    for view in world_views:
        expected_sha = (expected_manifests or {}).get(view)
        caches[view] = ViewDinoCache.open(
            root / view,
            view=view,
            episodes=episodes,
            rows=rows,
            expected_manifest_sha256=expected_sha,
            full_hash=full_hash,
        )
    return caches
