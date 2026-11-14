#!/usr/bin/env python3
"""Official zero-shot evaluation and aggregation script for ResWAM on LIBERO-Plus and LIBERO.

Evaluates all 4 task suites (libero_spatial: 2402, libero_object: 2518, libero_goal: 2591,
libero_10: 2519; total 10,030 episodes) across the 7 perturbation dimensions of LIBERO-Plus:
  1. Camera Viewpoints
  2. Robot Initial States
  3. Language Instructions
  4. Light Conditions
  5. Background Textures
  6. Sensor Noise
  7. Objects Layout

Supports:
  - Single-GPU / sharded worker evaluation (`worker` mode)
  - Multi-GPU parallel evaluation across suites and shards (`run` mode)
  - Standalone Websocket policy server (`server` mode)
  - Official-horizon (220/280/300/520) and executed-horizon (250/300/320/550) aggregation (`aggregate` mode)

Usage:
    python scripts/eval/eval_libero_plus.py run \
        --checkpoint /path/to/model.pt \
        --libero-plus-root ./third_party/LIBERO-plus \
        --output-dir ./outputs/libero/eval_libero_plus \
        --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reswam.data.codec import ZScoreCodec, featurize_libero_state_11d  # noqa: E402
from reswam.models.system import ResWAMSystem, system_config  # noqa: E402
from reswam.utils.dist import configure_determinism  # noqa: E402
from reswam.utils.websocket_policy import WebsocketPolicyServer  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_PLUS_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
OFFICIAL_HORIZON = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
EXECUTED_HORIZON = {
    "libero_spatial": 250,
    "libero_object": 300,
    "libero_goal": 320,
    "libero_10": 550,
}
PERTURBATION_DIMENSIONS = (
    ("Camera Viewpoints", "Camera"),
    ("Robot Initial States", "Robot"),
    ("Language Instructions", "Language"),
    ("Light Conditions", "Light"),
    ("Background Textures", "Background"),
    ("Sensor Noise", "Noise"),
    ("Objects Layout", "Layout"),
)
DIM_SHORT = dict(PERTURBATION_DIMENSIONS)
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def official_rollout_seed(suite_name: str, task_id: int, init_state_idx: int) -> int:
    identity = f"{suite_name}\0{int(task_id)}\0{int(init_state_idx)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


def derive_chunk_seed(
    suite: str, task_id: int, init_state_idx: int, rollout_seed: int, chunk_index: int
) -> int:
    identity = (
        f"{suite}\0{int(task_id)}\0{int(init_state_idx)}\0"
        f"{int(rollout_seed)}\0{int(chunk_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


def extract_raw_state_8d(obs: dict[str, Any]) -> np.ndarray:
    fingers = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2]
    return np.concatenate(
        (np.asarray(obs["robot0_eef_pos"], dtype=np.float32), quat2axisangle(obs["robot0_eef_quat"]), fingers)
    ).astype(np.float32)


def extract_cameras_180flip(obs: dict[str, Any]) -> dict[str, list[np.ndarray]]:
    primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1], dtype=np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1], dtype=np.uint8)
    return {"head": [primary], "left_wrist": [wrist]}


def format_libero_env_action(decoded_action: np.ndarray) -> list[float]:
    act = np.asarray(decoded_action, dtype=np.float32).reshape(-1).copy()
    if act.shape != (7,):
        raise ValueError(f"Expected 7-D LIBERO action, got {act.shape}")
    gripper_raw = float(act[6])
    # In ResWAM LIBERO codec, raw > 0.5 was encoded to +1 (close) and <= 0.5 to -1 (open)
    # Robosuite expects +1 for close and -1 for open after invert_gripper_action
    act[6] = 1.0 if gripper_raw >= 0.0 else -1.0
    return act.tolist()


def load_reswam_checkpoint(
    checkpoint_path: Path,
    *,
    codec_path: Path | None = None,
    device: str = "cuda",
) -> tuple[ResWAMSystem, ZScoreCodec, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if codec_path is not None:
        codec = ZScoreCodec.from_file(codec_path)
    elif "codec" in payload:
        codec = ZScoreCodec.from_metadata(payload["codec"])
    elif (checkpoint_path.parent / "codec_statistics.json").is_file():
        # Released Hugging Face checkpoints ship the codec as a sibling codec_statistics.json.
        codec = ZScoreCodec.from_file(checkpoint_path.parent / "codec_statistics.json")
    else:
        raise RuntimeError(
            "Checkpoint does not contain codec metadata and no codec_statistics.json sits next "
            "to it; pass --codec-statistics"
        )

    cfg = payload.get("config", {})
    sys_cfg = cfg.get("system")
    if sys_cfg is None:
        sys_cfg = system_config(
            action_dim=codec.action_dim,
            state_dim=codec.state_dim,
            horizon=8,
            views=("head", "left_wrist"),
            world_views=("head",),
            world_loss_weight=0.5,
            action_tau_shift=1.0,
            world_tau_shift=1.0,
            solver_steps=10,
            state_dropout_prob=0.3,
            prompt_style="libero_cot",
        )
    system = ResWAMSystem(sys_cfg)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    system.load_state_dict(state_dict, strict=True)
    system.to(device).eval()
    return system, codec, payload


class LiberoPolicyWrapper:
    """Policy wrapper supporting both direct evaluation and Websocket serving."""

    def __init__(self, system: ResWAMSystem, codec: ZScoreCodec, device: str = "cuda") -> None:
        self.system = system
        self.codec = codec
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict_chunk(
        self,
        *,
        obs: dict[str, Any],
        instruction: str,
        seed: int,
    ) -> np.ndarray:
        cameras = extract_cameras_180flip(obs)
        raw_state = extract_raw_state_8d(obs)
        norm_state = torch.from_numpy(self.codec.encode_state(raw_state)).unsqueeze(0).to(self.device)
        norm_actions = self.system.sample_actions(
            cameras=cameras,
            instructions=[str(instruction)],
            normalized_state=norm_state,
            seeds=[int(seed)],
        )
        return self.codec.decode_action(norm_actions[0].detach().cpu().numpy())

    def predict_action(self, **msg: Any) -> dict[str, Any]:
        examples = msg.get("examples", [msg])
        images, wrists, instructions, seeds, states = [], [], [], [], []
        for ex in examples:
            images.append(np.asarray(ex["primary_image"][0], dtype=np.uint8))
            wrists.append(np.asarray(ex["wrist_image"][0], dtype=np.uint8))
            instructions.append(str(ex["lang"]))
            seeds.append(int(ex.get("rollout_seed", 0)))
            raw_st = np.asarray(ex.get("proprio_state_raw", ex.get("state")), dtype=np.float32).reshape(-1)
            states.append(self.codec.encode_state(raw_st))
        norm_state = torch.from_numpy(np.stack(states)).to(self.device)
        norm_actions = self.system.sample_actions(
            cameras={"head": images, "left_wrist": wrists},
            instructions=instructions,
            normalized_state=norm_state,
            seeds=seeds,
        )
        return {"normalized_actions": norm_actions.detach().cpu().numpy()}


def load_task_classification(libero_plus_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    cls_path = libero_plus_root / "libero/libero/benchmark/task_classification.json"
    if not cls_path.is_file():
        return {}
    raw = json.loads(cls_path.read_text())
    return {
        suite: {int(item["id"]) - 1: item for item in raw.get(suite, [])}
        for suite in SUITES
    }


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_worker(args: argparse.Namespace) -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    lp_root = Path(args.libero_plus_root).expanduser().resolve()
    if lp_root.is_dir() and str(lp_root) not in sys.path:
        sys.path.insert(0, str(lp_root))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    configure_determinism(int(args.seed))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    system, codec, _ = load_reswam_checkpoint(
        Path(args.checkpoint),
        codec_path=Path(args.codec_statistics) if args.codec_statistics else None,
        device=device,
    )
    policy = LiberoPolicyWrapper(system, codec, device=device)
    classification = load_task_classification(lp_root)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    n_tasks = int(task_suite.n_tasks)
    if args.max_tasks is not None:
        n_tasks = min(n_tasks, int(args.max_tasks))
    task_ids = [i for i in range(n_tasks) if i % int(args.num_shards) == int(args.shard)]

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "episodes.partial.jsonl"
    records: list[dict[str, Any]] = []
    if partial_path.is_file():
        records = [
            json.loads(line)
            for line in partial_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed_keys = {(int(r["task_id"]), int(r["episode_idx"])) for r in records}
    max_steps = int(EXECUTED_HORIZON[args.suite])

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        meta = classification.get(args.suite, {}).get(task_id, {})
        category = meta.get("category", "Unknown")
        task_name = meta.get("name", getattr(task, "name", f"task_{task_id}"))
        difficulty = meta.get("difficulty_level")

        for ep_idx in range(int(args.trials_per_task)):
            if (task_id, ep_idx) in completed_keys:
                continue
            init_state_idx = int(ep_idx % len(initial_states))
            ep_seed = official_rollout_seed(args.suite, task_id, init_state_idx)
            np.random.seed(ep_seed % (2**32))
            random.seed(ep_seed)

            bddl_file = (
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            )
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            env.seed(int(args.seed))
            try:
                env.reset()
                obs = env.set_init_state(initial_states[init_state_idx])
                for _ in range(int(args.num_steps_wait)):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                num_actions = 0
                chunk_idx = 0
                done = False
                while num_actions < max_steps:
                    c_seed = derive_chunk_seed(
                        args.suite, task_id, init_state_idx, ep_seed, chunk_idx
                    )
                    chunk = policy.predict_chunk(
                        obs=obs, instruction=task.language, seed=c_seed
                    )
                    chunk_idx += 1
                    exec_len = min(
                        int(args.execute_horizon), chunk.shape[0], max_steps - num_actions
                    )
                    for k in range(exec_len):
                        env_act = format_libero_env_action(chunk[k])
                        obs, _, done_flag, _ = env.step(env_act)
                        num_actions += 1
                        if bool(done_flag):
                            done = True
                            break
                    if done:
                        break

                rec = {
                    "suite": args.suite,
                    "task_id": int(task_id),
                    "task_name": str(task_name),
                    "task_description": str(task.language),
                    "category": str(category),
                    "difficulty_level": difficulty,
                    "episode_idx": int(ep_idx),
                    "initial_state_idx": int(init_state_idx),
                    "rollout_seed": int(ep_seed),
                    "success": bool(done),
                    "num_actions": int(num_actions),
                }
                records.append(rec)
                completed_keys.add((task_id, ep_idx))
                atomic_write_jsonl(partial_path, records)
            finally:
                env.close()

    total_ep = len(records)
    total_succ = sum(int(r["success"]) for r in records)
    summary = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "task_suite_name": args.suite,
        "shard": int(args.shard),
        "num_shards": int(args.num_shards),
        "total_episodes": total_ep,
        "total_successes": total_succ,
        "total_success_rate": total_succ / max(1, total_ep),
    }
    atomic_write_jsonl(out_dir / "episodes.jsonl", records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "shard_complete", **summary}), flush=True)
    return 0


def wilson_ci(successes: int, episodes: int, z: float = 1.959964) -> list[float]:
    if episodes <= 0:
        return [0.0, 0.0]
    rate = successes / episodes
    denom = 1.0 + z * z / episodes
    centre = (rate + z * z / (2.0 * episodes)) / denom
    half = z * math.sqrt(rate * (1.0 - rate) / episodes + z * z / (4.0 * episodes * episodes)) / denom
    return [centre - half, centre + half]


def aggregate_results(
    output_dir: Path,
    libero_plus_root: Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    classification = load_task_classification(libero_plus_root)
    records_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for suite in SUITES:
        suite_dir = output_dir / suite
        if not suite_dir.is_dir():
            continue
        shard_dirs = sorted(suite_dir.glob("shard-*")) or [suite_dir]
        for sdir in shard_dirs:
            ep_file = sdir / "episodes.jsonl"
            if not ep_file.is_file():
                ep_file = sdir / "episodes.partial.jsonl"
            if not ep_file.is_file():
                continue
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                task_id = int(rec["task_id"])
                ep_idx = int(rec.get("episode_idx", 0))
                meta = classification.get(suite, {}).get(task_id, {})
                rec["suite"] = suite
                rec["category"] = meta.get("category", rec.get("category", "Unknown"))
                rec["difficulty_level"] = meta.get("difficulty_level", rec.get("difficulty_level"))
                records_by_key[(suite, task_id, ep_idx)] = rec

    records = list(records_by_key.values())
    if not allow_incomplete and len(records) != 10030:
        raise RuntimeError(
            f"Incomplete LIBERO-Plus evaluation: found {len(records)}/10030 episodes "
            f"(pass --allow-incomplete for partial summary)"
        )

    def _compute_metrics(limits: dict[str, int]) -> dict[str, Any]:
        groups: dict[str, dict[str, list[int]]] = {
            "dimension": defaultdict(lambda: [0, 0]),
            "suite": defaultdict(lambda: [0, 0]),
            "difficulty": defaultdict(lambda: [0, 0]),
        }
        total_succ = 0
        for rec in records:
            suite = rec["suite"]
            succ = int(bool(rec["success"]) and int(rec["num_actions"]) <= limits[suite])
            total_succ += succ
            for kind, key in (
                ("dimension", str(rec.get("category", "Unknown"))),
                ("suite", suite),
                ("difficulty", f"level{rec.get('difficulty_level')}"),
            ):
                groups[kind][key][0] += succ
                groups[kind][key][1] += 1

        def _rates(d: dict[str, list[int]]) -> dict[str, Any]:
            return {
                k: {"successes": v[0], "episodes": v[1], "rate": v[0] / max(1, v[1])}
                for k, v in sorted(d.items())
            }

        by_dim = _rates(groups["dimension"])
        dim_rates = [v["rate"] for v in by_dim.values()]
        return {
            "total": {
                "successes": total_succ,
                "episodes": len(records),
                "rate": total_succ / max(1, len(records)),
                "wilson95": wilson_ci(total_succ, len(records)),
            },
            "mean_of_dimensions": sum(dim_rates) / max(1, len(dim_rates)),
            "by_dimension": by_dim,
            "by_suite": _rates(groups["suite"]),
            "by_difficulty": _rates(groups["difficulty"]),
        }

    result = {
        "schema_version": "reswam-libero-plus-eval-result-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "episodes": len(records),
        "complete": len(records) == 10030,
        "official": _compute_metrics(OFFICIAL_HORIZON),
        "executed_horizon": _compute_metrics(EXECUTED_HORIZON),
    }
    out_file = output_dir / "libero_plus_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "libero_plus_aggregated", "output": str(out_file), "official_total": result["official"]["total"]}), flush=True)
    return result


def run_parallel(args: argparse.Namespace) -> int:
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    out_dir = Path(args.output_dir).expanduser().resolve()
    suites = args.suites.split(",") if args.suites else list(SUITES)
    shards_per_suite = int(args.shards_per_suite)

    work_units = []
    for suite in suites:
        for shard in range(shards_per_suite):
            shard_dir = out_dir / suite / f"shard-{shard:03d}of{shards_per_suite:03d}"
            if (shard_dir / "summary.json").is_file():
                continue
            work_units.append((suite, shard, shards_per_suite, shard_dir))

    print(f"[eval_libero_plus] Launching {len(work_units)} pending shard units across GPUs {gpus}")
    active: dict[int, tuple[subprocess.Popen, tuple[str, int]]] = {}
    idx = 0
    failed = 0

    while idx < len(work_units) or active:
        for gpu in list(active):
            proc, (suite, shard) = active[gpu]
            rc = proc.poll()
            if rc is not None:
                del active[gpu]
                if rc != 0:
                    failed += 1
                    print(f"[eval_libero_plus] Unit {suite} shard {shard} failed on GPU {gpu} (rc={rc})")

        for gpu in gpus:
            if gpu not in active and idx < len(work_units):
                suite, shard, num_shards, shard_dir = work_units[idx]
                idx += 1
                shard_dir.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--checkpoint",
                    str(args.checkpoint),
                    "--libero-plus-root",
                    str(args.libero_plus_root),
                    "--suite",
                    suite,
                    "--shard",
                    str(shard),
                    "--num-shards",
                    str(num_shards),
                    "--trials-per-task",
                    str(args.trials_per_task),
                    "--execute-horizon",
                    str(args.execute_horizon),
                    "--output-dir",
                    str(shard_dir),
                ]
                if args.codec_statistics:
                    cmd += ["--codec-statistics", str(args.codec_statistics)]
                if args.max_tasks is not None:
                    cmd += ["--max-tasks", str(args.max_tasks)]
                log_fh = (shard_dir / "worker.log").open("a")
                proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
                log_fh.close()
                active[gpu] = (proc, (suite, shard))
        time.sleep(2.0)

    if failed > 0:
        return 1
    aggregate_results(
        out_dir,
        Path(args.libero_plus_root),
        allow_incomplete=args.max_tasks is not None or len(suites) < 4,
    )
    return 0


def run_server(args: argparse.Namespace) -> int:
    configure_determinism(20260907)
    system, codec, _ = load_reswam_checkpoint(
        Path(args.checkpoint),
        codec_path=Path(args.codec_statistics) if args.codec_statistics else None,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    policy = LiberoPolicyWrapper(system, codec)
    metadata = {
        "framework_name": "reswam_libero_plus",
        "action_horizon": 8,
        "replan_interval": 8,
        "solver": "synchronous_jacobi_euler",
        "rounds": 10,
    }
    WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, idle_timeout=-1, metadata=metadata
    ).serve_forever()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ResWAM LIBERO-Plus Zero-Shot Evaluation and Aggregation")
    sub = p.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="Run parallel multi-GPU evaluation across all LIBERO-Plus suites")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--codec-statistics", type=Path, default=None)
    r.add_argument("--libero-plus-root", type=Path, default=Path("./third_party/LIBERO-plus"))
    r.add_argument("--output-dir", type=Path, default=Path("./outputs/libero/eval_libero_plus"))
    r.add_argument("--gpus", type=str, default="0")
    r.add_argument("--suites", type=str, default="libero_spatial,libero_object,libero_goal,libero_10")
    r.add_argument("--shards-per-suite", type=int, default=4)
    r.add_argument("--trials-per-task", type=int, default=1)
    r.add_argument("--execute-horizon", type=int, default=8)
    r.add_argument("--max-tasks", type=int, default=None)

    w = sub.add_parser("worker", help="Evaluate a single suite shard on one GPU")
    w.add_argument("--checkpoint", type=Path, required=True)
    w.add_argument("--codec-statistics", type=Path, default=None)
    w.add_argument("--libero-plus-root", type=Path, default=Path("./third_party/LIBERO-plus"))
    w.add_argument("--suite", type=str, required=True, choices=SUITES)
    w.add_argument("--shard", type=int, default=0)
    w.add_argument("--num-shards", type=int, default=1)
    w.add_argument("--trials-per-task", type=int, default=1)
    w.add_argument("--execute-horizon", type=int, default=8)
    w.add_argument("--num-steps-wait", type=int, default=10)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--max-tasks", type=int, default=None)
    w.add_argument("--output-dir", type=Path, required=True)

    a = sub.add_parser("aggregate", help="Aggregate completed shard results into libero_plus_results.json")
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--libero-plus-root", type=Path, default=Path("./third_party/LIBERO-plus"))
    a.add_argument("--allow-incomplete", action="store_true")

    s = sub.add_parser("server", help="Start Websocket policy server for external LIBERO clients")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--codec-statistics", type=Path, default=None)
    s.add_argument("--host", type=str, default="127.0.0.1")
    s.add_argument("--port", type=int, default=10093)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "run":
        return run_parallel(args)
    if args.mode == "worker":
        return run_worker(args)
    if args.mode == "aggregate":
        aggregate_results(args.output_dir, args.libero_plus_root, allow_incomplete=args.allow_incomplete)
        return 0
    if args.mode == "server":
        return run_server(args)
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
