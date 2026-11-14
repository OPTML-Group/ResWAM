"""End-padded window dataset, exact-timestamp video decoder, and distributed sampler for ResWAM."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Sequence, Sized

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..models.dinov3_encoder import preprocess_camera
from ..models.system import CONTRACT_VIEW, VIEW_ORDER, resolve_world_views
from .augmentation import augment_window_cameras
from .codec import ZScoreCodec
from .dino_cache import ViewDinoCache, float_view, open_world_caches, sha256_file


def window_positions(length: int, start: int, horizon: int) -> dict[str, Any]:
    """Compute clamped action indices s..s+H-1 and endpoint index min(s+H, L-1)."""
    if not 0 <= start < length:
        raise IndexError(f"window start {start} outside [0, {length})")
    steps = np.arange(start, start + horizon)
    return {
        "actions": np.minimum(steps, length - 1),
        "state": start,
        "endpoint": min(start + horizon, length - 1),
        "action_padded": steps > length - 1,
        "endpoint_padded": start + horizon > length - 1,
    }


class VideoFrames:
    """Exact-timestamp frame decoding with a per-process LRU cache of open PyAV containers."""

    def __init__(self, entries: int = 64) -> None:
        self.entries = entries
        self.open: OrderedDict[str, Any] = OrderedDict()
        self.max_error_s = 0.0

    def _container(self, path: str):
        container = self.open.pop(path, None)
        if container is None:
            container = av.open(
                path, options={"threads": "1", "tilethreads": "1", "framethreads": "1"}
            )
            stream = container.streams.video[0]
            stream.thread_count = 1
            stream.codec_context.options = {
                "threads": "1",
                "tilethreads": "1",
                "framethreads": "1",
            }
            if len(self.open) >= self.entries:
                self.open.popitem(last=False)[1].close()
        self.open[path] = container
        return container

    def decode(self, path: str, timestamps: list[float], fps: float) -> list[np.ndarray]:
        container = self._container(path)
        stream = container.streams.video[0]
        tolerance = 0.5 / fps
        first, last = min(timestamps), max(timestamps)
        for seek_to in (first, 0.0):
            container.seek(
                int(max(seek_to - tolerance, 0.0) / stream.time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            best: dict[int, tuple[float, np.ndarray | None]] = {
                i: (np.inf, None) for i in range(len(timestamps))
            }
            started = False
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                if not started:
                    started = True
                    if t > first + tolerance:
                        break
                array = None
                for i, target in enumerate(timestamps):
                    error = abs(t - target)
                    if error < best[i][0]:
                        if array is None:
                            array = frame.to_ndarray(format="rgb24")
                        best[i] = (error, array)
                if t > last + tolerance:
                    break
            if all(b[1] is not None and b[0] <= tolerance for b in best.values()):
                self.max_error_s = max(self.max_error_s, max(b[0] for b in best.values()))
                return [best[i][1] for i in range(len(timestamps))]
        raise IndexError(f"frames at {timestamps} not found in {path}")


class LeRobotWindowDataset(Dataset):
    """End-padded window dataset over LeRobot v3.0 / v2.1 parquet and mp4 files.

    Supports:
      - Online mode (``dino_cache_root=None``): decodes both current frame ``o_t`` and future
        frame ``o_{t+H}`` from video files for online frozen DINOv3 encoding.
      - Cached mode (``dino_cache_root`` set): decodes only current policy frames and reads
        ``E(o_t)`` and ``E(o_{t+H})`` from memory-mapped FP32 shards.
      - Cleaned trajectory indexing (``clean_index_path`` set): indexes kept non-stationary
        frames per episode (LiLa-WAM protocol for RoboTwin 2.0).
      - Optional shared temporal image augmentation (``pi05_aug=True``): applies pi0.5 crop,
        rotation, and color jitter identically to ``(o_t, o_{t+H})`` on the primary view and
        color jitter on wrist views.
    """

    def __init__(
        self,
        contract: dict[str, Any],
        codec: ZScoreCodec,
        *,
        horizon: int,
        instructions: dict[int, str],
        world_views: Sequence[str] | None = None,
        task_index: int | Sequence[int] | None = None,
        episode_indices: Sequence[int] | None = None,
        dino_cache_root: str | Path | None = None,
        dino_cache_manifests: dict[str, str] | None = None,
        clean_index_path: str | Path | None = None,
        clean_index_sha256: str | None = None,
        end_padding: str = "repeat_last_frame_supervised",
        pi05_aug: bool = False,
        base_seed: int = 20260830,
    ) -> None:
        self.root = Path(contract["root"])
        self.contract = contract
        self.codec = codec
        self.horizon = int(horizon)
        self.end_padding = str(end_padding)
        self.pi05_aug = bool(pi05_aug)
        self.base_seed = int(base_seed)
        self.shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        if self.pi05_aug and dino_cache_root is not None:
            raise RuntimeError("pi05_aug requires online DINOv3 encoding and forbids static DINO cache")

        info = json.loads((self.root / "meta/info.json").read_text())
        codebase_ver = str(info.get("codebase_version", "v3.0"))
        if codebase_ver not in ("v3.0", "v2.1", "v2.0"):
            raise RuntimeError(f"Unsupported LeRobot codebase_version: {codebase_ver}")
        self.codebase_version = codebase_ver
        self.fps = float(info["fps"])
        self.video_path = info["video_path"]
        self.chunks_size = int(info.get("chunks_size", 1000))
        self.view_keys = {
            CONTRACT_VIEW.get(v, v): contract["views"][v]
            for v in contract["views"]
            if contract["views"].get(v)
        }
        self.views = tuple(v for v in VIEW_ORDER if v in self.view_keys)
        self.world_views = resolve_world_views(
            world_views or contract.get("world_views") or contract.get("future_views"),
            self.views,
        )
        self.instructions = instructions
        sk, ak = contract["state_key"], contract["action_key"]
        columns = [sk, ak, "index", "episode_index", "frame_index", "task_index"]
        parquet_files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_files:
            parquet_files = sorted((self.root / "data").glob("**/*.parquet"))
        parts = [pq.read_table(f, columns=columns) for f in parquet_files]
        index = np.concatenate([p.column("index").to_numpy() for p in parts])
        order = np.argsort(index, kind="stable")
        self.index = index[order]

        def stack(key: str, dtype) -> np.ndarray:
            rows = []
            for p in parts:
                column = p.column(key).combine_chunks()
                if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
                    flat = column.flatten().to_numpy(zero_copy_only=False)
                    rows.append(flat.reshape(len(column), -1).astype(dtype))
                else:
                    rows.append(column.to_numpy(zero_copy_only=False).astype(dtype))
            return np.concatenate(rows)[order]

        self.state = stack(sk, np.float32).reshape(len(order), -1)
        self.action = stack(ak, np.float32).reshape(len(order), -1)
        self.task_index = stack("task_index", np.int64)
        self.episode_index = stack("episode_index", np.int64)
        self.frame_index = stack("frame_index", np.int64)
        if self.action.shape[1] != codec.action_dim:
            raise RuntimeError(
                f"dataset action_dim {self.action.shape[1]} != codec {codec.action_dim}"
            )
        if (
            self.state.shape[1] != codec.state_dim
            and not (
                codec.state_featurization == "xyz_rot6d_fingers2_11d"
                and self.state.shape[1] == 8
                and codec.state_dim == 11
            )
        ):
            raise RuntimeError(
                f"dataset state_dim {self.state.shape[1]} incompatible with codec {codec.state_dim}"
            )

        meta: list[dict[str, Any]] = []
        ep_parquet_files = sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if ep_parquet_files:
            for f in ep_parquet_files:
                table = pq.read_table(f)
                meta.extend(
                    table.select(
                        [c for c in table.column_names if not c.startswith("stats/")]
                    ).to_pylist()
                )
        elif (self.root / "meta/episodes.jsonl").is_file():
            for line in (self.root / "meta/episodes.jsonl").read_text().splitlines():
                if line.strip():
                    meta.append(json.loads(line))
        else:
            raise FileNotFoundError(f"Episode metadata not found in {self.root / 'meta'}")

        episodes = sorted(meta, key=lambda r: int(r["episode_index"]))
        self.source_episodes: list[dict[str, Any]] = []
        self.source_rows: list[int] = []
        running_offset = 0
        for ep in episodes:
            ep_idx = int(ep["episode_index"])
            length = int(ep["length"])
            if "dataset_from_index" in ep:
                begin = int(np.searchsorted(self.index, ep["dataset_from_index"]))
            else:
                begin = running_offset
            running_offset = begin + length
            rows_slice = slice(begin, begin + length)
            if (self.episode_index[rows_slice] != ep_idx).any() or (
                self.frame_index[rows_slice] != np.arange(length)
            ).any():
                raise RuntimeError(f"episode {ep_idx} rows are not contiguous frames 0..L-1")
            self.source_episodes.append(ep)
            self.source_rows.append(begin)

        clean_map: dict[int, dict[str, Any]] | None = None
        if clean_index_path is not None:
            cpath = Path(clean_index_path)
            if not cpath.is_file():
                raise RuntimeError(f"clean_index_path missing: {cpath}")
            if clean_index_sha256:
                actual_sha = sha256_file(cpath)
                if actual_sha != clean_index_sha256:
                    raise RuntimeError(
                        f"clean_index sha256 mismatch: {actual_sha} != {clean_index_sha256}"
                    )
            clean_payload = json.loads(cpath.read_text())
            clean_map = {int(rec["episode_index"]): rec for rec in clean_payload["episodes"]}

        allowed_tasks: set[int] | None = None
        if task_index is not None:
            allowed_tasks = (
                {int(task_index)}
                if isinstance(task_index, (int, np.integer))
                else {int(t) for t in task_index}
            )
        allowed_episodes: set[int] | None = (
            {int(e) for e in episode_indices} if episode_indices is not None else None
        )
        self.episodes: list[dict[str, Any]] = []
        self.rows: list[int] = []
        self.short: list[int] = []
        self.kept_rows: list[np.ndarray | None] = []
        self.kept_frame_indices: list[np.ndarray | None] = []

        for ep, begin in zip(self.source_episodes, self.source_rows, strict=True):
            ep_idx = int(ep["episode_index"])
            if clean_map is not None and ep_idx not in clean_map:
                continue
            if allowed_episodes is not None and ep_idx not in allowed_episodes:
                continue
            length = int(ep["length"])
            rows_slice = slice(begin, begin + length)
            if allowed_tasks is not None:
                ep_tasks = set(int(t) for t in np.unique(self.task_index[rows_slice]))
                if not ep_tasks.issubset(allowed_tasks):
                    continue
            if clean_map is not None:
                rec = clean_map[ep_idx]
                k_rows = np.asarray(rec["kept_rows"], dtype=np.int64)
                k_frames = np.asarray(rec["kept_frame_indices"], dtype=np.int64)
                if len(k_rows) != int(rec["clean_length"]) or len(k_frames) != len(k_rows):
                    raise RuntimeError(f"clean_index length mismatch for episode {ep_idx}")
                if not np.array_equal(k_rows, begin + k_frames):
                    raise RuntimeError(
                        f"clean_index kept_rows != begin + kept_frame_indices for episode {ep_idx}"
                    )
                eff_len = len(k_rows)
                self.kept_rows.append(k_rows)
                self.kept_frame_indices.append(k_frames)
            else:
                eff_len = length
                self.kept_rows.append(None)
                self.kept_frame_indices.append(None)
            self.episodes.append(ep)
            self.rows.append(begin)
            if eff_len <= self.horizon:
                self.short.append(ep_idx)

        if not self.episodes:
            raise RuntimeError(
                f"no episodes matched task_index={task_index!r} episode_indices={episode_indices!r}"
            )
        if clean_map is not None and self.short:
            raise RuntimeError(
                f"clean dataset contains episodes <= horizon {self.horizon}: {self.short}"
            )
        self.lengths = [
            len(kr) if kr is not None else int(ep["length"])
            for ep, kr in zip(self.episodes, self.kept_rows, strict=True)
        ]
        self.cumulative = np.cumsum(self.lengths, dtype=np.int64)
        self.frames = VideoFrames()
        self.dino_caches: dict[str, ViewDinoCache] | None = None
        if dino_cache_root is not None:
            cache_path = Path(dino_cache_root)
            if cache_path.is_dir() and any((cache_path / v).is_dir() for v in self.world_views):
                cache_episodes = self.source_episodes if clean_map is not None else self.episodes
                cache_rows = self.source_rows if clean_map is not None else self.rows
                self.dino_caches = open_world_caches(
                    cache_path,
                    world_views=self.world_views,
                    episodes=cache_episodes,
                    rows=cache_rows,
                    expected_manifests=dino_cache_manifests,
                )

    def set_epoch(self, epoch: int) -> None:
        self.shared_epoch.fill_(int(epoch))

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def locate(self, index: int) -> tuple[int, int]:
        episode = int(np.searchsorted(self.cumulative, index, side="right"))
        start = index - (int(self.cumulative[episode - 1]) if episode else 0)
        return episode, start

    def _video(self, episode: dict[str, Any], key: str) -> tuple[str, float]:
        ep_idx = int(episode["episode_index"])
        chunk_key = f"videos/{key}/chunk_index"
        file_key = f"videos/{key}/file_index"
        ts_key = f"videos/{key}/from_timestamp"
        if chunk_key in episode and file_key in episode:
            path = self.root / self.video_path.format(
                video_key=key,
                chunk_index=episode[chunk_key],
                file_index=episode[file_key],
                episode_chunk=episode[chunk_key],
                episode_index=ep_idx,
            )
            offset = float(episode.get(ts_key, 0.0))
        else:
            ep_chunk = ep_idx // max(1, self.chunks_size)
            path = self.root / self.video_path.format(
                video_key=key,
                chunk_index=ep_chunk,
                file_index=ep_idx,
                episode_chunk=ep_chunk,
                episode_index=ep_idx,
            )
            offset = float(episode.get(ts_key, 0.0))
        return str(path), offset

    def __getitem__(self, index: int) -> dict[str, Any]:
        e, start = self.locate(int(index))
        episode, begin, length = self.episodes[e], self.rows[e], self.lengths[e]
        positions = window_positions(length, start, self.horizon)
        kr = self.kept_rows[e]
        kf = self.kept_frame_indices[e]
        if kr is not None and kf is not None:
            current_row = int(kr[positions["state"]])
            endpoint_row = int(kr[positions["endpoint"]])
            current_frame_idx = int(kf[positions["state"]])
            endpoint_frame_idx = int(kf[positions["endpoint"]])
            ep_actions = self.action[kr]
        else:
            current_row = int(begin + positions["state"])
            endpoint_row = int(begin + positions["endpoint"])
            current_frame_idx = int(positions["state"])
            endpoint_frame_idx = int(positions["endpoint"])
            ep_actions = self.action[begin : begin + length]

        current, endpoint = {}, {}
        dino_current, dino_endpoint = {}, {}
        for view in self.views:
            path, offset = self._video(episode, self.view_keys[view])
            times = [offset + current_frame_idx / self.fps]
            if view in self.world_views and self.dino_caches is None:
                times.append(offset + endpoint_frame_idx / self.fps)
            decoded_frames = self.frames.decode(path, times, self.fps)
            current[view] = torch.from_numpy(decoded_frames[0].copy()).permute(2, 0, 1)
            if view in self.world_views:
                if self.dino_caches is not None:
                    dino_current[view] = self.dino_caches[view].row(current_row)
                    dino_endpoint[view] = self.dino_caches[view].row(endpoint_row)
                else:
                    endpoint[view] = torch.from_numpy(decoded_frames[1].copy()).permute(2, 0, 1)

        if self.pi05_aug:
            current, endpoint = augment_window_cameras(
                current,
                endpoint,
                base_seed=self.base_seed,
                epoch=int(self.shared_epoch.item()),
                current_row=current_row,
                endpoint_row=endpoint_row,
            )

        raw_actions = ep_actions[positions["actions"]].copy()
        if (
            self.end_padding == "zero_delta_hold_gripper_supervised"
            and positions["action_padded"].any()
        ):
            padded_mask = positions["action_padded"]
            non_gripper = [
                i for i in range(raw_actions.shape[1]) if i not in self.codec.gripper_indices
            ]
            raw_actions[np.ix_(padded_mask, non_gripper)] = 0.0
            for g in self.codec.gripper_indices:
                raw_actions[padded_mask, int(g)] = ep_actions[-1, int(g)]

        out = {
            "cameras": current,
            "future_cameras": endpoint,
            "actions": torch.from_numpy(self.codec.encode_action(raw_actions)),
            "action_mask": torch.ones(self.horizon, dtype=torch.bool),
            "state": torch.from_numpy(self.codec.encode_state(self.state[current_row])),
            "state_mask": torch.ones(self.codec.state_dim, dtype=torch.bool),
            "instruction": self.instructions[int(self.task_index[current_row])],
            "episode_id": int(episode["episode_index"]),
            "window_start": int(start),
            "padded_action_steps": int(positions["action_padded"].sum()),
            "endpoint_padded": bool(positions["endpoint_padded"]),
        }
        if self.dino_caches is not None:
            out["dino_current"] = dino_current
            out["dino_endpoint"] = dino_endpoint
        return out


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate window samples into a batch with bicubic 256x256 camera tensors."""
    if not samples:
        raise ValueError("empty batch")
    views = tuple(samples[0]["cameras"])
    world_views = tuple(samples[0]["future_cameras"])
    current = {
        v: torch.stack([preprocess_camera(s["cameras"][v]) for s in samples]) for v in views
    }
    endpoint = {
        v: torch.stack([preprocess_camera(s["future_cameras"][v]) for s in samples])
        for v in world_views
    }
    batch = {
        "current_cameras": current,
        "endpoint_cameras": endpoint,
        "camera_order": views,
        "world_camera_order": world_views,
        "actions": torch.stack([s["actions"] for s in samples]),
        "action_mask": torch.stack([s["action_mask"] for s in samples]).bool(),
        "state": torch.stack([s["state"] for s in samples]),
        "state_mask": torch.stack([s["state_mask"] for s in samples]).bool(),
        "qwen_cameras": {
            v: [
                f.mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
                for f in current[v]
            ]
            for v in views
        },
        "instructions": [s["instruction"] for s in samples],
        "sample_metadata": [
            {
                k: s[k]
                for k in (
                    "episode_id",
                    "window_start",
                    "padded_action_steps",
                    "endpoint_padded",
                )
            }
            for s in samples
        ],
    }
    if "dino_current" in samples[0]:
        cached_views = tuple(samples[0]["dino_current"])
        batch["world_camera_order"] = cached_views
        batch["dino_current"] = {
            v: float_view(torch.stack([s["dino_current"][v] for s in samples]))
            for v in cached_views
        }
        batch["dino_endpoint"] = {
            v: float_view(torch.stack([s["dino_endpoint"][v] for s in samples]))
            for v in cached_views
        }
    return batch


class DistributedWindowSampler(Sampler[int]):
    """Seeded per-epoch permutation with rank striding and resumable position."""

    def __init__(
        self,
        dataset: Sized,
        *,
        rank: int,
        world_size: int,
        seed: int,
        microbatch: int,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("invalid rank")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.microbatch = microbatch
        self.epoch = self.position = 0
        if self._per_rank() == 0:
            raise ValueError(
                f"dataset ({len(dataset)} windows) smaller than world_size x microbatch"
            )

    def _per_rank(self) -> int:
        return (len(self.dataset) // self.world_size) // self.microbatch * self.microbatch

    def _indices(self) -> list[int]:
        permutation = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(self.seed + self.epoch),
        ).tolist()
        return permutation[self.rank :: self.world_size][: self._per_rank()]

    def __iter__(self) -> Iterator[int]:
        indices = self._indices()
        for position in range(self.position, len(indices)):
            self.position = position + 1
            yield indices[position]

    def __len__(self) -> int:
        return self._per_rank()

    def set_epoch(self, epoch: int, position: int = 0) -> None:
        self.epoch, self.position = int(epoch), int(position)
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(self.epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.set_epoch(state["epoch"], state["position"])


def _worker_init_fn(_worker_id: int) -> None:
    torch.set_num_threads(1)


def build_loader(
    dataset: LeRobotWindowDataset,
    *,
    rank: int,
    world_size: int,
    seed: int,
    microbatch: int,
    num_workers: int = 8,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DistributedWindowSampler]:
    sampler = DistributedWindowSampler(
        dataset, rank=rank, world_size=world_size, seed=seed, microbatch=microbatch
    )
    generator = torch.Generator().manual_seed(int(seed) + 8_000_033 * int(rank))
    kwargs = (
        {"prefetch_factor": prefetch_factor, "worker_init_fn": _worker_init_fn}
        if num_workers > 0
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=microbatch,
        sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        generator=generator,
        **kwargs,
    )
    return loader, sampler


def isolated_loader_iterator(loader: DataLoader, *, seed: int, rank: int, epoch: int):
    loader.generator.manual_seed(int(seed) + 8_000_033 * int(rank) + 104_729 * int(epoch))
    return iter(loader)
