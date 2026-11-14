#!/usr/bin/env python3
"""Prepare RoboDojo LeRobot dataset and normalization statistics for ResWAM.

Supports:
  - Optional download from Hugging Face (`RoboDojo-Benchmark/RoboDojo` @ `35efbc7`)
  - Computing 14-D z-score joint + affine [0, 1] -> [-1, 1] gripper statistics across all 35 training tasks
    (3,500 episodes, 1,856,102 frames, all-frames variant)
  - Writing dataset.json, codec_statistics.json, and reswam.json.

Usage:
    python scripts/data_prep/prepare_robodojo.py \
        --dataset-root ./data/robodojo \
        --output-dir ./data/prepared/robodojo
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

from reswam.data.codec import STATISTICS_SCHEMA, describe  # noqa: E402

HF_REPO = "RoboDojo-Benchmark/RoboDojo"
HF_REVISION = "35efbc7dedfdbeeb6e95fb749bd885d73d483e41"
SUBDIR = "data/RoboDojo_lerobot_v21_video"
GRIPPER_INDICES = [6, 13]


def download_from_hf(target_dir: Path, revision: str = HF_REVISION) -> Path:
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_robodojo] Downloading {HF_REPO}@{revision} into {target_dir}...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{SUBDIR}/**"],
        local_dir=str(target_dir),
    )
    nested = target_dir / SUBDIR
    return nested if (nested / "meta/info.json").is_file() else target_dir


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
    raise FileNotFoundError(f"Task metadata not found in {root / 'meta'}")


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


def prepare_robodojo(
    dataset_root: Path,
    output_dir: Path,
    *,
    horizon: int = 32,
    execute_before_replan: int = 10,
    head_key: str = "observation.images.cam_high",
    left_wrist_key: str = "observation.images.cam_left_wrist",
    right_wrist_key: str = "observation.images.cam_right_wrist",
    state_key: str = "observation.state",
    action_key: str = "action",
    download: bool = False,
    force: bool = False,
) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    reswam_json = output_dir / "reswam.json"
    if reswam_json.is_file() and not force:
        print(f"[prepare_robodojo] {reswam_json} already exists (pass --force to rebuild)")
        return reswam_json

    if download and not (dataset_root / "meta/info.json").is_file():
        dataset_root = download_from_hf(dataset_root)
    elif not (dataset_root / "meta/info.json").is_file() and (
        dataset_root / SUBDIR / "meta/info.json"
    ).is_file():
        dataset_root = (dataset_root / SUBDIR).resolve()

    info = json.loads((dataset_root / "meta/info.json").read_text())
    columns = ["index", "episode_index", "frame_index", "task_index", state_key, action_key]
    data = _read_parquet_columns(dataset_root, columns)
    actions_all = np.asarray(data[action_key], dtype=np.float64).reshape(len(data["index"]), -1)
    states_all = np.asarray(data[state_key], dtype=np.float64).reshape(len(data["index"]), -1)

    if actions_all.shape[1] != 14 or states_all.shape[1] != 14:
        raise RuntimeError(
            f"RoboDojo requires 14-D state and action, got state={states_all.shape}, action={actions_all.shape}"
        )

    episodes = _read_episodes(dataset_root)
    tasks = _read_tasks(dataset_root)
    lengths = [int(ep["length"]) for ep in episodes]

    stats = {
        "schema_version": STATISTICS_SCHEMA,
        "source": f"RoboDojo all-frames ({len(actions_all)} frames across {len(episodes)} episodes)",
        "gripper_indices": GRIPPER_INDICES,
        "state_gripper_indices": GRIPPER_INDICES,
        "gripper_mode": "affine",
        "encode_clamp_sigma": 12.0,
        "state_featurization": "identity",
        "action": describe(actions_all, GRIPPER_INDICES, fixed_gripper_range=(0.0, 1.0)),
        "state": describe(states_all, GRIPPER_INDICES, fixed_gripper_range=(0.0, 1.0)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "codec_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    contract = {
        "schema": "reswam-dataset-contract-v1",
        "dataset": str(dataset_root),
        "dataset_id": "robodojo",
        "root": str(dataset_root),
        "codebase_version": info.get("codebase_version", "v3.0"),
        "fps": int(info.get("fps", 25)),
        "state_key": state_key,
        "action_key": action_key,
        "state_dim": 14,
        "action_dim": 14,
        "action_space": "joint_absolute",
        "gripper_indices": GRIPPER_INDICES,
        "arms": 2,
        "views": {
            "base": head_key,
            "left_wrist": left_wrist_key,
            "right_wrist": right_wrist_key,
        },
        "world_views": ["head"],
        "tasks": tasks,
        "total_episodes": len(episodes),
        "total_frames": int(sum(lengths)),
        "episodes": [
            {"episode_index": int(ep["episode_index"]), "length": int(ep["length"])}
            for ep in episodes
        ],
    }
    contract_path = output_dir / "dataset.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    reswam_cfg = {
        "schema": "reswam-dataset-config-v1",
        "dataset": str(dataset_root),
        "dataset_id": "robodojo",
        "root": str(dataset_root),
        "contract": str(contract_path),
        "fps": int(info.get("fps", 25)),
        "action_dim": 14,
        "state_dim": 14,
        "state_dropout_prob": 0.0,
        "horizon": int(horizon),
        "execute_before_replan": int(execute_before_replan),
        "views": ["head", "left_wrist", "right_wrist"],
        "view_keys": {
            "head": head_key,
            "left_wrist": left_wrist_key,
            "right_wrist": right_wrist_key,
        },
        "world_view": "head",
        "world_views": ["head"],
        "world_view_keys": {"head": head_key},
        "action_space": "joint_absolute",
        "end_padding": "repeat_last_frame_supervised",
        "gripper_indices": GRIPPER_INDICES,
        "state_gripper_indices": GRIPPER_INDICES,
        "codec_statistics": str(stats_path),
        "codec_statistics_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "instructions": tasks,
        "episodes": len(episodes),
        "frames": int(sum(lengths)),
        "windows": int(sum(lengths)),
        "stable_v2_codec_compatible": True,
    }
    reswam_json.write_text(json.dumps(reswam_cfg, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "robodojo_prepared",
                "config": str(reswam_json),
                "episodes": len(episodes),
                "frames": int(sum(lengths)),
            }
        )
    )
    return reswam_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare RoboDojo LeRobot dataset for ResWAM")
    ap.add_argument("--dataset-root", type=Path, default=Path("./data/robodojo"))
    ap.add_argument("--output-dir", type=Path, default=Path("./data/prepared/robodojo"))
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--execute-before-replan", type=int, default=10)
    ap.add_argument("--head-key", type=str, default="observation.images.cam_high")
    ap.add_argument("--left-wrist-key", type=str, default="observation.images.cam_left_wrist")
    ap.add_argument("--right-wrist-key", type=str, default="observation.images.cam_right_wrist")
    ap.add_argument("--download", action="store_true", help="Download dataset from Hugging Face if absent")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prepare_robodojo(
        args.dataset_root,
        args.output_dir,
        horizon=args.horizon,
        execute_before_replan=args.execute_before_replan,
        head_key=args.head_key,
        left_wrist_key=args.left_wrist_key,
        right_wrist_key=args.right_wrist_key,
        download=args.download,
        force=args.force,
    )


if __name__ == "__main__":
    main()
