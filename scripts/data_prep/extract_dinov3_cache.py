#!/usr/bin/env python3
"""Optional multi-GPU precomputation of frozen DINOv3 camera features for ResWAM datasets.

Note: ResWAM supports online frozen DINOv3 feature extraction during training out of the box
(``use_dino_cache: false``). Pre-extracting features with this script is optional and enables
faster training throughput by avoiding repeated video decoding of endpoint frames ``o_{t+H}``.

For each world view (e.g. ``head``), decodes all frames of each video file sequentially,
resizes to 256x256 with bicubic antialiasing, runs ``DINOv3DualViewEncoder`` in batches of 32,
and writes contiguous ``shard-XXX.f32`` files + ``DINO_CACHE_MANIFEST.json`` under
``<out_root>/<view>/``.

Usage:
    python scripts/data_prep/extract_dinov3_cache.py \
        --dataset-config ./data/prepared/robotwin2/reswam.json \
        --out-root ./cache/dino/robotwin2 \
        --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reswam.data.codec import ZScoreCodec  # noqa: E402
from reswam.data.dataset import LeRobotWindowDataset  # noqa: E402
from reswam.data.dino_cache import (  # noqa: E402
    CACHE_SCHEMA,
    MANIFEST_NAME,
    ROW_BYTES,
    ViewDinoCache,
    layout_sha256,
)
from reswam.models.dinov3_encoder import (  # noqa: E402
    DINO_REPO,
    DINO_REVISION,
    DINOv3DualViewEncoder,
    preprocess_camera,
)

CHUNK = 32
SHARD_ROWS = CHUNK * 1024  # 32,768 rows (~6.44 GB per shard)


def decode_episode_frames(
    video_path: str, from_timestamp: float, length: int, fps: float
) -> list[torch.Tensor]:
    """Decode ``length`` frames starting at ``from_timestamp`` at ``fps`` from ``video_path``."""
    timestamps = [from_timestamp + i / fps for i in range(length)]
    tolerance = 0.5 / fps
    first, last = timestamps[0], timestamps[-1]
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        for seek_to in (first, 0.0):
            container.seek(
                int(max(seek_to - tolerance, 0.0) / stream.time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            best: dict[int, tuple[float, np.ndarray | None]] = {
                i: (np.inf, None) for i in range(length)
            }
            started = False
            ptr = 0
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                if not started:
                    started = True
                    if t > first + tolerance:
                        break
                if t < first - tolerance:
                    continue
                if t > last + tolerance:
                    break
                while ptr + 1 < length and abs(timestamps[ptr + 1] - t) <= abs(timestamps[ptr] - t):
                    ptr += 1
                arr = None
                for i in range(max(0, ptr - 2), min(length, ptr + 3)):
                    err = abs(t - timestamps[i])
                    if err < best[i][0]:
                        if arr is None:
                            arr = frame.to_ndarray(format="rgb24")
                        best[i] = (err, arr)
            if all(b[1] is not None and b[0] <= tolerance for b in best.values()):
                return [
                    preprocess_camera(torch.from_numpy(best[i][1].copy()).permute(2, 0, 1))
                    for i in range(length)
                ]
    finally:
        container.close()
    raise IndexError(
        f"failed to decode {length} frames from {video_path} starting at {from_timestamp}"
    )


class EpisodeBatchDataset(Dataset):
    """Yields (begin_row, length, Tensor[length, 3, 256, 256] bf16) per episode."""

    def __init__(
        self,
        episodes_meta: list[dict[str, Any]],
        video_key: str,
        root: Path,
        video_path_tmpl: str,
        fps: float,
    ) -> None:
        self.episodes_meta = episodes_meta
        self.video_key = video_key
        self.root = root
        self.video_path_tmpl = video_path_tmpl
        self.fps = fps

    def __len__(self) -> int:
        return len(self.episodes_meta)

    def __getitem__(self, idx: int):
        torch.set_num_threads(1)
        ep = self.episodes_meta[idx]
        vpath = str(
            self.root
            / self.video_path_tmpl.format(
                video_key=self.video_key,
                chunk_index=ep[f"videos/{self.video_key}/chunk_index"],
                file_index=ep[f"videos/{self.video_key}/file_index"],
            )
        )
        offset = float(ep[f"videos/{self.video_key}/from_timestamp"])
        length = int(ep["length"])
        frames = decode_episode_frames(vpath, offset, length, self.fps)
        stacked = torch.stack(frames).to(torch.bfloat16)
        return int(ep["begin_row"]), length, stacked


def gpu_worker(gpu: int, shard_tasks: list[dict[str, Any]], cfg: dict[str, Any], out_q) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda", 0)
        dino = DINOv3DualViewEncoder(
            cfg["dino_model_id"],
            revision=cfg["dino_revision"],
            selected_blocks=(8, 10),
            image_size=256,
        ).to(device)
        dino.requires_grad_(False)
        dino.eval()

        for task in shard_tasks:
            view = task["view"]
            video_key = task["video_key"]
            shard = task["shard"]
            out_dir = Path(task["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            s_begin, s_end = int(shard["begin"]), int(shard["end"])
            path = out_dir / shard["file"]
            rec_path = path.with_suffix(".json")
            if (
                path.is_file()
                and rec_path.is_file()
                and path.stat().st_size == (s_end - s_begin) * ROW_BYTES
            ):
                out_q.put({"gpu": gpu, "view": view, "record": json.loads(rec_path.read_text())})
                continue

            ov_eps = [
                ep
                for ep in cfg["episodes"]
                if int(ep["begin_row"]) + int(ep["length"]) > s_begin
                and int(ep["begin_row"]) < s_end
            ]
            ds = EpisodeBatchDataset(
                ov_eps, video_key, Path(cfg["data_root"]), cfg["video_path_tmpl"], cfg["fps"]
            )
            loader = DataLoader(
                ds,
                batch_size=None,
                shuffle=False,
                num_workers=cfg["workers_per_gpu"],
                prefetch_factor=2 if cfg["workers_per_gpu"] > 0 else None,
                pin_memory=True,
            )
            tmp_path = path.with_name(f".{path.name}.tmp")
            tmp_path.unlink(missing_ok=True)
            t0 = time.perf_counter()
            digest = hashlib.sha256()
            written_rows = 0
            expected_row = s_begin
            buf_tensors: list[torch.Tensor] = []
            buf_count = 0

            with tmp_path.open("wb") as fh:
                def flush_batches(force: bool = False) -> None:
                    nonlocal buf_tensors, buf_count, written_rows
                    if not buf_tensors:
                        return
                    cat = torch.cat(buf_tensors, dim=0)
                    while cat.shape[0] >= CHUNK or (force and cat.shape[0] > 0):
                        take = min(CHUNK, int(cat.shape[0]))
                        chunk_t = cat[:take]
                        cat = cat[take:]
                        if take < CHUNK:
                            pad = chunk_t[-1:].expand(CHUNK - take, -1, -1, -1)
                            chunk_in = torch.cat([chunk_t, pad], dim=0)
                        else:
                            chunk_in = chunk_t
                        with torch.no_grad():
                            feats = dino(chunk_in.to(device, non_blocking=True))
                        if feats.shape != (CHUNK, 64, 768) or feats.dtype != torch.float32:
                            raise RuntimeError(f"bad DINO output {feats.shape} {feats.dtype}")
                        raw = feats[:take].cpu().numpy().astype("<f4", copy=False).tobytes()
                        digest.update(raw)
                        fh.write(raw)
                        written_rows += take
                    buf_tensors = [cat] if cat.shape[0] > 0 else []
                    buf_count = int(cat.shape[0])

                for ep_begin, ep_len, pixels in loader:
                    lo = max(0, s_begin - ep_begin)
                    hi = min(ep_len, s_end - ep_begin)
                    if ep_begin + lo != expected_row:
                        raise RuntimeError(
                            f"non-contiguous episode slice at {ep_begin + lo}, expected {expected_row}"
                        )
                    slice_px = pixels[lo:hi]
                    buf_tensors.append(slice_px)
                    buf_count += int(slice_px.shape[0])
                    expected_row += int(slice_px.shape[0])
                    if buf_count >= CHUNK * 4:
                        flush_batches(force=False)
                flush_batches(force=True)

            if written_rows != (s_end - s_begin):
                raise RuntimeError(f"wrote {written_rows} rows, expected {s_end - s_begin}")
            os.chmod(tmp_path, 0o444)
            os.replace(tmp_path, path)
            rec = {
                "file": shard["file"],
                "begin": s_begin,
                "end": s_end,
                "size": written_rows * ROW_BYTES,
                "sha256": digest.hexdigest(),
                "seconds": round(time.perf_counter() - t0, 2),
                "gpu": gpu,
            }
            rec_tmp = rec_path.with_name(f".{rec_path.name}.tmp")
            rec_tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
            os.chmod(rec_tmp, 0o444)
            os.replace(rec_tmp, rec_path)
            out_q.put({"gpu": gpu, "view": view, "record": rec})
        out_q.put({"gpu": gpu, "done": True})
    except BaseException as exc:
        import traceback

        out_q.put(
            {"gpu": gpu, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute frozen DINOv3 cache for ResWAM datasets")
    ap.add_argument("--dataset-config", type=Path, required=True, help="Path to prepared reswam.json")
    ap.add_argument("--out-root", type=Path, required=True, help="Output directory for <view>/shard-*.f32")
    ap.add_argument("--data-root", type=Path, help="Optional dataset root override")
    ap.add_argument("--dino-model", type=str, default=os.environ.get("RESWAM_DINOV3_PATH", DINO_REPO))
    ap.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    ap.add_argument("--workers-per-gpu", type=int, default=8)
    args = ap.parse_args()

    import multiprocessing as mp

    data_cfg = json.loads(args.dataset_config.read_text())
    contract = json.loads(Path(data_cfg["contract"]).read_text())
    if args.data_root:
        contract["root"] = str(args.data_root.resolve())
    codec = ZScoreCodec.from_file(data_cfg["codec_statistics"])
    ds = LeRobotWindowDataset(
        contract,
        codec,
        horizon=int(data_cfg["horizon"]),
        instructions={int(k): str(v) for k, v in data_cfg["instructions"].items()},
        world_views=data_cfg["world_views"],
        episode_indices=data_cfg.get("episode_indices"),
        clean_index_path=data_cfg.get("clean_index"),
        clean_index_sha256=data_cfg.get("clean_index_sha256"),
    )
    src_episodes = ds.source_episodes if ds.kept_rows and any(x is not None for x in ds.kept_rows) else ds.episodes
    src_rows = ds.source_rows if ds.kept_rows and any(x is not None for x in ds.kept_rows) else ds.rows
    frames = sum(int(ep["length"]) for ep in src_episodes)
    ep_layout_sha = layout_sha256(src_episodes, src_rows)
    episodes_list = [
        {**ep, "begin_row": int(begin)} for ep, begin in zip(src_episodes, src_rows, strict=True)
    ]

    all_tasks = []
    view_shards: dict[str, list[dict[str, Any]]] = {}
    for view in ds.world_views:
        v_out = args.out_root / view
        shards = [
            {"file": f"shard-{i:03d}.f32", "begin": b, "end": min(b + SHARD_ROWS, frames)}
            for i, b in enumerate(range(0, frames, SHARD_ROWS))
        ]
        view_shards[view] = shards
        if (v_out / MANIFEST_NAME).is_file():
            print(f"[extract_dinov3_cache] {view} already has {v_out / MANIFEST_NAME}; verifying...")
            ViewDinoCache.open(v_out, view=view, episodes=src_episodes, rows=src_rows, full_hash=False)
            continue
        for s in shards:
            all_tasks.append(
                {
                    "view": view,
                    "video_key": ds.view_keys[view],
                    "shard": s,
                    "out_dir": str(v_out),
                }
            )

    if all_tasks:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        cfg = {
            "episodes": episodes_list,
            "data_root": str(ds.root),
            "video_path_tmpl": ds.video_path,
            "fps": ds.fps,
            "workers_per_gpu": args.workers_per_gpu,
            "dino_model_id": args.dino_model,
            "dino_revision": DINO_REVISION,
        }
        procs = []
        for idx, gpu in enumerate(gpus):
            sub_tasks = all_tasks[idx :: len(gpus)]
            p = ctx.Process(target=gpu_worker, args=(gpu, sub_tasks, cfg, out_q))
            p.start()
            procs.append(p)

        done_gpus = set()
        t_start = time.perf_counter()
        while len(done_gpus) < len(gpus):
            msg = out_q.get(timeout=1800)
            if "error" in msg:
                for p in procs:
                    p.terminate()
                print(json.dumps({"event": "dino_precompute_error", **msg}), flush=True)
                return 1
            if msg.get("done"):
                done_gpus.add(msg["gpu"])
                continue
            print(
                json.dumps(
                    {
                        "event": "dino_shard_done",
                        "view": msg["view"],
                        **msg["record"],
                        "elapsed": round(time.perf_counter() - t_start, 1),
                    }
                ),
                flush=True,
            )
        for p in procs:
            p.join()

    manifests = {}
    for view in ds.world_views:
        v_out = args.out_root / view
        manifest_path = v_out / MANIFEST_NAME
        if not manifest_path.is_file():
            rows = []
            for s in view_shards[view]:
                rec = json.loads((v_out / Path(s["file"]).with_suffix(".json")).read_text())
                rows.append({k: rec[k] for k in ("file", "begin", "end", "size", "sha256")})
            manifest = {
                "schema_version": CACHE_SCHEMA,
                "complete": True,
                "view": view,
                "video_key": ds.view_keys[view],
                "dtype": "float32",
                "byte_order": "little",
                "row_shape": [64, 768],
                "frames": frames,
                "episode_layout_sha256": ep_layout_sha,
                "dino": {
                    "model_id": DINO_REPO,
                    "revision": DINO_REVISION,
                    "selected_blocks": [8, 10],
                    "spatial_pooling": "2x2",
                    "tokens": 64,
                },
                "shards": rows,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            tmp_m = v_out / f".{MANIFEST_NAME}.tmp"
            tmp_m.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.chmod(tmp_m, 0o444)
            os.replace(tmp_m, manifest_path)
        cache = ViewDinoCache.open(v_out, view=view, episodes=src_episodes, rows=src_rows, full_hash=True)
        manifests[view] = cache.manifest_sha256
        print(
            json.dumps(
                {
                    "event": "dino_view_verified",
                    "view": view,
                    "manifest": str(manifest_path),
                    "manifest_sha256": cache.manifest_sha256,
                    "frames": frames,
                }
            ),
            flush=True,
        )

    summary_path = args.out_root / "DINO_WORLD_CACHE_SUMMARY.json"
    summary_payload = {
        "schema_version": "reswam-dino-world-cache-summary-v1",
        "dataset_id": data_cfg["dataset_id"],
        "frames": frames,
        "world_views": list(ds.world_views),
        "manifests": manifests,
        "episode_layout_sha256": ep_layout_sha,
        "complete": True,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "dino_all_complete", **summary_payload}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
