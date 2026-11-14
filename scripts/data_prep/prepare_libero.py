#!/usr/bin/env python3
"""Prepare LIBERO LeRobot dataset metadata and normalization statistics for ResWAM.

Computes over all 273,465 frames across 1,693 episodes (`physical-intelligence/libero`):
  - 7-D delta-EEF action statistics (z-score over dims 0..5 with 12.0 sigma clamp, binary +-1 gripper on dim 6)
  - 11-D proprioceptive state statistics (3D EEF position + 6D continuous Rodrigues rotation + 2D finger qpos,
    z-scored with 10.0 sigma clamp)
  - Resolved dataset contract (dataset.json), codec statistics (codec_statistics.json),
    and ResWAM dataset configuration (reswam.json).

Usage:
    python scripts/data_prep/prepare_libero.py \
        --dataset-root ./data/libero \
        --output-dir ./data/prepared/libero
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reswam.data.codec import (  # noqa: E402
    DEFAULT_ENCODE_CLAMP_SIGMA,
    DEFAULT_LIBERO_STATE_CLAMP_SIGMA,
    STATISTICS_SCHEMA,
    describe,
    featurize_libero_state_11d,
)

HF_REPO = "physical-intelligence/libero"
HF_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


def download_from_hf(
    target_dir: Path, repo_id: str = HF_REPO, revision: str = HF_REVISION
) -> Path:
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_libero] Downloading {repo_id}@{revision} into {target_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target_dir),
    )
    return target_dir


def _read_parquet_columns(root: Path, columns: list[str]) -> dict[str, np.ndarray]:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    parts = [pq.read_table(f, columns=columns) for f in files]
    idx = np.concatenate([p.column("index").to_numpy() for p in parts])
    order = np.argsort(idx, kind="stable")
    out: dict[str, np.ndarray] = {"index": idx[order]}
    for col_name in columns:
        if col_name == "index":
            continue
        rows = []
        for p in parts:
            col = p.column(col_name).combine_chunks()
            if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
                flat = col.flatten().to_numpy(zero_copy_only=False)
                rows.append(flat.reshape(len(col), -1))
            else:
                rows.append(col.to_numpy(zero_copy_only=False))
        out[col_name] = np.concatenate(rows)[order]
    return out


def _read_tasks(root: Path) -> dict[str, str]:
    tasks_parquet = root / "meta/tasks.parquet"
    tasks_jsonl = root / "meta/tasks.jsonl"
    if tasks_parquet.is_file():
        df = pq.read_table(tasks_parquet).to_pandas()
        if "task" in df.columns:
            return {str(int(r.task_index)): str(r.task) for r in df.itertuples()}
        return {str(int(idx)): str(task) for task, idx in zip(df.index, df["task_index"])}
    if tasks_jsonl.is_file():
        rows = [json.loads(line) for line in tasks_jsonl.read_text().splitlines() if line.strip()]
        return {str(int(r["task_index"])): str(r["task"]) for r in rows}
    raise FileNotFoundError(f"Neither meta/tasks.parquet nor meta/tasks.jsonl found in {root}")


def _read_episodes(root: Path) -> list[dict[str, Any]]:
    ep_files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if ep_files:
        meta = []
        for f in ep_files:
            table = pq.read_table(f)
            keep = [c for c in table.column_names if not c.startswith("stats/")]
            meta.extend(table.select(keep).to_pylist())
        return sorted(meta, key=lambda r: int(r["episode_index"]))
    ep_jsonl = root / "meta/episodes.jsonl"
    if ep_jsonl.is_file():
        rows = [json.loads(line) for line in ep_jsonl.read_text().splitlines() if line.strip()]
        return sorted(rows, key=lambda r: int(r["episode_index"]))
    raise FileNotFoundError(f"Episode metadata not found in {root / 'meta'}")


def prepare_libero(
    dataset_root: Path,
    output_dir: Path,
    *,
    horizon: int = 8,
    head_key: str = "observation.images.image",
    wrist_key: str = "observation.images.wrist_image",
    state_key: str = "observation.state",
    action_key: str = "action",
    download: bool = False,
    force: bool = False,
) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    reswam_json = output_dir / "reswam.json"
    if reswam_json.is_file() and not force:
        print(f"[prepare_libero] {reswam_json} already exists (pass --force to rebuild)")
        return reswam_json

    if download and not (dataset_root / "meta/info.json").is_file():
        download_from_hf(dataset_root)

    info = json.loads((dataset_root / "meta/info.json").read_text())
    columns = ["index", "episode_index", "frame_index", "task_index", state_key, action_key]
    data = _read_parquet_columns(dataset_root, columns)
    raw_actions = np.asarray(data[action_key], dtype=np.float64).reshape(len(data["index"]), -1)
    raw_states = np.asarray(data[state_key], dtype=np.float64).reshape(len(data["index"]), -1)

    if raw_actions.shape[1] != 7:
        raise RuntimeError(f"Expected 7-D LIBERO actions, got shape {raw_actions.shape}")
    if raw_states.shape[1] == 8:
        featurized_states = featurize_libero_state_11d(raw_states)
        state_featurization = "xyz_rot6d_fingers2_11d"
        state_dim = 11
    else:
        featurized_states = raw_states
        state_featurization = "identity"
        state_dim = int(raw_states.shape[1])

    grippers = [6]
    state_grippers: list[int] = []
    stats = {
        "schema_version": STATISTICS_SCHEMA,
        "source": f"all {len(raw_actions)} frames of LIBERO ({dataset_root.name})",
        "gripper_indices": grippers,
        "state_gripper_indices": state_grippers,
        "gripper_mode": "binary_pm1",
        "encode_clamp_sigma": float(DEFAULT_ENCODE_CLAMP_SIGMA),
        "state_clamp_sigma": float(DEFAULT_LIBERO_STATE_CLAMP_SIGMA),
        "state_featurization": state_featurization,
        "action": describe(raw_actions, grippers, fixed_gripper_range=(0.0, 1.0)),
        "state": describe(featurized_states, state_grippers),
    }

    episodes = _read_episodes(dataset_root)
    tasks = _read_tasks(dataset_root)
    lengths = [int(ep["length"]) for ep in episodes]

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "codec_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    contract = {
        "schema": "reswam-dataset-contract-v1",
        "dataset": str(dataset_root),
        "dataset_id": "libero",
        "root": str(dataset_root),
        "codebase_version": info.get("codebase_version", "v3.0"),
        "fps": int(info.get("fps", 20)),
        "state_key": state_key,
        "action_key": action_key,
        "state_dim": state_dim,
        "raw_state_dim": int(raw_states.shape[1]),
        "action_dim": 7,
        "action_space": "eef_delta",
        "gripper_indices": grippers,
        "arms": 1,
        "views": {
            "base": head_key,
            "left_wrist": wrist_key,
            "right_wrist": None,
        },
        "world_views": ["head"],
        "tasks": tasks,
        "total_episodes": len(episodes),
        "total_frames": int(sum(lengths)),
        "episodes": [
            {
                "episode_index": int(ep["episode_index"]),
                "length": int(ep["length"]),
            }
            for ep in episodes
        ],
    }
    contract_path = output_dir / "dataset.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    reswam_cfg = {
        "schema": "reswam-dataset-config-v1",
        "dataset": str(dataset_root),
        "dataset_id": "libero",
        "root": str(dataset_root),
        "contract": str(contract_path),
        "fps": int(info.get("fps", 20)),
        "action_dim": 7,
        "state_dim": state_dim,
        "raw_state_dim": int(raw_states.shape[1]),
        "state_featurization": state_featurization,
        "state_dropout_prob": 0.3,
        "horizon": int(horizon),
        "execute_before_replan": int(horizon),
        "views": ["head", "left_wrist"],
        "view_keys": {"head": head_key, "left_wrist": wrist_key},
        "world_view": "head",
        "world_views": ["head"],
        "world_view_keys": {"head": head_key},
        "action_space": "eef_delta",
        "end_padding": "zero_delta_hold_gripper_supervised",
        "gripper_indices": grippers,
        "state_gripper_indices": state_grippers,
        "gripper_mode": "binary_pm1",
        "codec_statistics": str(stats_path),
        "codec_statistics_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "instructions": tasks,
        "episodes": len(lengths),
        "frames": int(sum(lengths)),
        "windows": int(sum(lengths)),
    }
    reswam_json.write_text(json.dumps(reswam_cfg, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "libero_prepared",
                "config": str(reswam_json),
                "episodes": len(lengths),
                "frames": int(sum(lengths)),
            }
        )
    )
    return reswam_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare LIBERO LeRobot dataset for ResWAM")
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data/libero"),
        help="Path to LIBERO LeRobot dataset directory",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/prepared/libero"),
        help="Output directory for prepared contract and codec statistics",
    )
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--head-key", type=str, default="observation.images.image")
    ap.add_argument("--wrist-key", type=str, default="observation.images.wrist_image")
    ap.add_argument("--state-key", type=str, default="observation.state")
    ap.add_argument("--action-key", type=str, default="action")
    ap.add_argument("--download", action="store_true", help="Download dataset from Hugging Face if absent")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prepare_libero(
        args.dataset_root,
        args.output_dir,
        horizon=args.horizon,
        head_key=args.head_key,
        wrist_key=args.wrist_key,
        state_key=args.state_key,
        action_key=args.action_key,
        download=args.download,
        force=args.force,
    )


if __name__ == "__main__":
    main()
