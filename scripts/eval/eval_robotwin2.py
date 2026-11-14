#!/usr/bin/env python3
"""Official 50-task RoboTwin 2.0 evaluation and aggregation script for ResWAM.

Protocol:
  - 50 canonical bimanual manipulation tasks x {clean (demo_clean), randomized (demo_randomized)}
  - 100 episodes per task/setting (10,000 total episodes)
  - Candidate seeds searched starting from ``100000 * (1 + base_seed)``; a seed is accepted iff
    the expert planner succeeds
  - Unseen language instruction sampling per accepted seed
  - Prediction horizon H = 32, execution horizon = 16 (p32e16) with degree-3 8-control-point
    B-spline trajectory smoothing over the 32-step chunk.

Supports:
  - Single-task worker evaluation (`worker` mode)
  - Multi-GPU parallel evaluation across all 50x2 task-settings (`run` mode)
  - Standalone Websocket policy server (`server` mode)
  - Full 50-task Clean / Randomized / Macro SR aggregation (`aggregate` mode)

Usage:
    python scripts/eval/eval_robotwin2.py run \
        --checkpoint /path/to/model.pt \
        --robotwin-root ./third_party/RoboTwin \
        --output-dir ./outputs/robotwin2/eval_p32e16 \
        --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reswam.data.codec import ZScoreCodec  # noqa: E402
from reswam.models.system import ResWAMSystem, system_config  # noqa: E402
from reswam.utils.dist import configure_determinism  # noqa: E402
from reswam.utils.websocket_policy import WebsocketPolicyServer, bspline_smooth  # noqa: E402

CANONICAL_TASKS: tuple[str, ...] = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
SETTINGS = ("clean", "randomized")
CAMERA_ORDER = ("head", "left_wrist", "right_wrist")


def derive_rollout_seed(
    *,
    task: str,
    split: str,
    episode_index: int,
    base_seed: int,
    chunk_index: int,
) -> int:
    """Per-chunk flow-noise seed of the official RoboTwin evaluation.

    ``sha256("task\\0split\\0episode_index\\0base_seed\\0chunk_index")[:8] mod (2**63 - 1)``
    with ``split`` in {"clean", "randomized"}, ``episode_index`` the 0-based accepted episode
    and ``chunk_index`` the 0-based replan count within the episode.
    """
    identity = "\0".join(
        str(value)
        for value in (task, split, int(episode_index), int(base_seed), int(chunk_index))
    ).encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


def load_robotwin_checkpoint(
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
            action_dim=14,
            state_dim=14,
            horizon=32,
            views=CAMERA_ORDER,
            world_views=("head",),
            world_loss_weight=0.1,
            action_tau_shift=1.0,
            world_tau_shift=5.0,
            solver_steps=10,
            prompt_style="multi_camera",
        )
    system = ResWAMSystem(sys_cfg)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    system.load_state_dict(state_dict, strict=True)
    system.to(device).eval()
    return system, codec, payload


class RoboTwinPolicy:
    """ResWAM policy wrapper for RoboTwin 2.0 with optional B-spline smoothing."""

    def __init__(
        self,
        system: ResWAMSystem,
        codec: ZScoreCodec,
        *,
        execute_horizon: int = 16,
        use_bspline: bool = True,
        device: str = "cuda",
    ) -> None:
        self.system = system
        self.codec = codec
        self.execute_horizon = int(execute_horizon)
        self.use_bspline = bool(use_bspline)
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict_chunk(
        self,
        *,
        cameras: dict[str, list[np.ndarray]],
        state: np.ndarray,
        instruction: str,
        seed: int,
    ) -> np.ndarray:
        norm_state = (
            torch.from_numpy(self.codec.encode_state(np.asarray(state, dtype=np.float32)))
            .unsqueeze(0)
            .to(self.device)
        )
        ordered_cameras = {name: cameras[name] for name in CAMERA_ORDER}
        norm_actions = self.system.sample_actions(
            cameras=ordered_cameras,
            instructions=[str(instruction)],
            normalized_state=norm_state,
            seeds=[int(seed)],
        )
        decoded = self.codec.decode_action(norm_actions[0].detach().cpu().numpy())
        if self.use_bspline:
            decoded = bspline_smooth(decoded, degree=3, num_ctrl_pts=8)
        return decoded[: self.execute_horizon]

    def predict_action(self, **payload: Any) -> dict[str, Any]:
        state = np.asarray(payload["state"], dtype=np.float32)
        cameras = {
            name: [np.asarray(payload["cameras"][name], dtype=np.uint8)]
            for name in CAMERA_ORDER
        }
        seed = int(
            payload.get(
                "derived_seed",
                derive_rollout_seed(
                    task=payload.get("task", ""),
                    split=payload.get("split", ""),
                    episode_index=payload.get("episode_index", 0),
                    base_seed=payload.get("base_seed", 0),
                    chunk_index=payload.get("chunk_index", 0),
                ),
            )
        )
        norm_state = (
            torch.from_numpy(self.codec.encode_state(state)).unsqueeze(0).to(self.device)
        )
        norm_actions = self.system.sample_actions(
            cameras=cameras,
            instructions=[str(payload["instruction"])],
            normalized_state=norm_state,
            seeds=[seed],
        )
        decoded = self.codec.decode_action(norm_actions[0].detach().cpu().numpy())
        if self.use_bspline:
            decoded = bspline_smooth(decoded, degree=3, num_ctrl_pts=8)
        return {"actions": decoded, "execute_steps": self.execute_horizon}


def select_instruction(
    task_name: str,
    instruction_type: str,
    episode_info: dict[str, Any],
    seed: int,
) -> str:
    from generate_episode_instructions import generate_episode_descriptions  # type: ignore

    # The official evaluation seeds Python's ``random`` with the episode seed before the
    # instruction is generated, so every checkpoint and protocol sees the same text per seed.
    random.seed(int(seed))
    descriptions = generate_episode_descriptions(task_name, [episode_info], 1)
    if not descriptions:
        return ""
    pool = descriptions[0].get(str(instruction_type), [])
    if not pool:
        pool = descriptions[0].get("unseen", []) or descriptions[0].get("seen", [])
    if not pool:
        return ""
    if len(pool) == 1:
        return str(pool[0])
    rng = np.random.default_rng(int(seed))
    return str(pool[int(rng.integers(0, len(pool)))])


def run_worker(args: argparse.Namespace) -> int:
    rt_root = Path(args.robotwin_root).expanduser().resolve()
    if not rt_root.is_dir():
        raise FileNotFoundError(
            f"RoboTwin 2.0 root not found at {rt_root}. Run scripts/env/setup_robotwin2.sh first."
        )
    os.chdir(rt_root)
    if str(rt_root) not in sys.path:
        sys.path.insert(0, str(rt_root))

    from envs import CONFIGS_PATH  # type: ignore
    from envs.utils.create_actor import UnStableError  # type: ignore
    from script.eval_policy import (  # type: ignore
        class_decorator,
        get_camera_config,
        get_embodiment_config,
    )

    configure_determinism(int(args.base_seed))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    system, codec, _ = load_robotwin_checkpoint(
        Path(args.checkpoint).expanduser().resolve(),
        codec_path=Path(args.codec_statistics).expanduser().resolve() if args.codec_statistics else None,
        device=device,
    )
    policy = RoboTwinPolicy(
        system,
        codec,
        execute_horizon=int(args.execute_horizon),
        use_bspline=not args.no_bspline,
        device=device,
    )

    task_config_name = f"demo_{args.setting}"
    with open(rt_root / f"task_config/{task_config_name}.yml", "r", encoding="utf-8") as fh:
        task_args = yaml.safe_load(fh)
    task_args["task_name"] = args.task
    task_args["task_config"] = task_config_name
    task_args["eval_mode"] = True
    task_args["save_data"] = False
    task_args["eval_video_log"] = False

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as fh:
        embodiment_types = yaml.safe_load(fh)
    emb_type = task_args["embodiment"]
    if len(emb_type) == 1:
        task_args["left_robot_file"] = embodiment_types[emb_type[0]]["file_path"]
        task_args["right_robot_file"] = embodiment_types[emb_type[0]]["file_path"]
        task_args["dual_arm_embodied"] = True
    else:
        task_args["left_robot_file"] = embodiment_types[emb_type[0]]["file_path"]
        task_args["right_robot_file"] = embodiment_types[emb_type[1]]["file_path"]
        task_args["embodiment_dis"] = emb_type[2]
        task_args["dual_arm_embodied"] = False
    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    cam_cfg = get_camera_config(task_args["camera"]["head_camera_type"])
    task_args["head_camera_h"] = cam_cfg["h"]
    task_args["head_camera_w"] = cam_cfg["w"]

    task_env = class_decorator(args.task)
    candidate_seed = 100_000 * (1 + int(args.base_seed))
    ordinal = 0
    episodes_detail: list[dict[str, Any]] = []
    successes = 0

    while len(episodes_detail) < int(args.trials):
        seed = candidate_seed + ordinal
        ordinal += 1
        probe_args = {**task_args, "render_freq": 0, "eval_video_save_dir": None}
        try:
            task_env.setup_demo(now_ep_num=len(episodes_detail), seed=seed, is_test=True, **probe_args)
            ep_info = task_env.play_once()
            task_env.close_env()
            valid = bool(task_env.plan_success and task_env.check_success())
        except (UnStableError, Exception):
            try:
                task_env.close_env()
            except Exception:
                pass
            continue
        if not valid or ep_info is None:
            continue

        ep_idx = len(episodes_detail)
        try:
            task_env.setup_demo(now_ep_num=ep_idx, seed=seed, is_test=True, **task_args)
            instruction = select_instruction(
                args.task, args.instruction_type, dict(ep_info.get("info", {})), seed
            )
            task_env.set_instruction(instruction=instruction)
            chunk_idx = 0
            while not bool(task_env.eval_success) and int(task_env.take_action_cnt) < int(task_env.step_lim):
                obs = task_env.get_obs()
                src = obs["observation"]
                cameras = {
                    "head": [np.asarray(src["head_camera"]["rgb"], dtype=np.uint8)],
                    "left_wrist": [np.asarray(src["left_camera"]["rgb"], dtype=np.uint8)],
                    "right_wrist": [np.asarray(src["right_camera"]["rgb"], dtype=np.uint8)],
                }
                state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
                c_seed = derive_rollout_seed(
                    task=args.task,
                    split=args.setting,
                    episode_index=ep_idx,
                    base_seed=int(args.base_seed),
                    chunk_index=chunk_idx,
                )
                chunk_idx += 1
                chunk = policy.predict_chunk(
                    cameras=cameras, state=state, instruction=instruction, seed=c_seed
                )
                for act in chunk:
                    task_env.take_action(np.asarray(act, dtype=np.float32), action_type="qpos")
                    if bool(task_env.eval_success) or int(task_env.take_action_cnt) >= int(task_env.step_lim):
                        break
            succ = bool(task_env.eval_success)
            steps = int(task_env.take_action_cnt)
        finally:
            try:
                task_env.close_env(clear_cache=((ep_idx + 1) % int(task_args.get("clear_cache_freq", 5)) == 0))
            except Exception:
                pass

        successes += int(succ)
        episodes_detail.append(
            {
                "episode_index": ep_idx,
                "seed": seed,
                "seed_search_ordinal": ordinal - 1,
                "condition": "clean" if args.setting == "clean" else "random",
                "success": succ,
                "steps": steps,
            }
        )

    out_path = Path(args.output_dir).expanduser().resolve() / f"{args.task}-{args.setting}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "reswam-robotwin2-item-v1",
        "task": args.task,
        "setting": args.setting,
        "episodes_count": len(episodes_detail),
        "successes": successes,
        "success_rate": successes / max(1, len(episodes_detail)),
        "execute_horizon": int(args.execute_horizon),
        "bspline": not args.no_bspline,
        "episodes": episodes_detail,
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "robotwin2_item_done", "task": args.task, "setting": args.setting, "sr": result["success_rate"]}), flush=True)
    return 0


def aggregate_results(output_dir: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    per_task: dict[str, dict[str, float]] = {s: {} for s in SETTINGS}
    for task in CANONICAL_TASKS:
        for setting in SETTINGS:
            item_file = output_dir / f"{task}-{setting}.json"
            if not item_file.is_file():
                if not allow_incomplete:
                    raise RuntimeError(f"Missing evaluation result: {item_file}")
                continue
            row = json.loads(item_file.read_text())
            per_task[setting][task] = float(row["success_rate"])

    clean_vals = list(per_task["clean"].values())
    rand_vals = list(per_task["randomized"].values())
    clean_mean = sum(clean_vals) / len(clean_vals) if clean_vals else 0.0
    rand_mean = sum(rand_vals) / len(rand_vals) if rand_vals else 0.0
    overall = (clean_mean + rand_mean) / 2.0

    summary = {
        "schema_version": "reswam-robotwin2-eval-result-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "complete": len(clean_vals) == 50 and len(rand_vals) == 50,
        "metrics": {
            "clean": clean_mean,
            "randomized": rand_mean,
            "overall": overall,
        },
        "tasks_evaluated": {
            "clean": len(clean_vals),
            "randomized": len(rand_vals),
        },
        "per_task": per_task,
    }
    out_file = output_dir / "robotwin2_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "robotwin2_aggregated", "output": str(out_file), "metrics": summary["metrics"]}), flush=True)
    return summary


def run_parallel(args: argparse.Namespace) -> int:
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",") if args.tasks else list(CANONICAL_TASKS)
    settings = args.settings.split(",") if args.settings else list(SETTINGS)

    work_items = []
    for task in tasks:
        for setting in settings:
            if (out_dir / f"{task}-{setting}.json").is_file():
                continue
            work_items.append((task, setting))

    print(f"[eval_robotwin2] Scheduling {len(work_items)} task-setting items across GPUs {gpus}")
    active: dict[int, tuple[subprocess.Popen, tuple[str, str]]] = {}
    idx = 0
    failed = 0

    while idx < len(work_items) or active:
        for gpu in list(active):
            proc, (task, setting) = active[gpu]
            rc = proc.poll()
            if rc is not None:
                del active[gpu]
                if rc != 0:
                    failed += 1
                    print(f"[eval_robotwin2] Failed {task}-{setting} on GPU {gpu} (rc={rc})")

        for gpu in gpus:
            if gpu not in active and idx < len(work_items):
                task, setting = work_items[idx]
                idx += 1
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--checkpoint",
                    str(args.checkpoint),
                    "--robotwin-root",
                    str(args.robotwin_root),
                    "--task",
                    task,
                    "--setting",
                    setting,
                    "--trials",
                    str(args.trials),
                    "--execute-horizon",
                    str(args.execute_horizon),
                    "--instruction-type",
                    str(args.instruction_type),
                    "--base-seed",
                    str(args.base_seed),
                    "--output-dir",
                    str(out_dir),
                ]
                if args.codec_statistics:
                    cmd += ["--codec-statistics", str(args.codec_statistics)]
                if args.no_bspline:
                    cmd += ["--no-bspline"]
                log_fh = (out_dir / f"{task}-{setting}.log").open("a")
                proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
                log_fh.close()
                active[gpu] = (proc, (task, setting))
        time.sleep(2.0)

    if failed > 0:
        return 1
    aggregate_results(out_dir, allow_incomplete=len(tasks) < 50 or len(settings) < 2)
    return 0


def run_server(args: argparse.Namespace) -> int:
    configure_determinism(20260907)
    system, codec, _ = load_robotwin_checkpoint(
        Path(args.checkpoint),
        codec_path=Path(args.codec_statistics) if args.codec_statistics else None,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    policy = RoboTwinPolicy(
        system,
        codec,
        execute_horizon=int(args.execute_horizon),
        use_bspline=not args.no_bspline,
    )
    metadata = {
        "schema_version": "reswam-robotwin2-server-v1",
        "prediction_horizon": 32,
        "execute_before_replan": int(args.execute_horizon),
        "camera_order": list(CAMERA_ORDER),
        "action_representation": "14-D absolute joint/qpos",
    }
    WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, idle_timeout=-1, metadata=metadata
    ).serve_forever()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ResWAM RoboTwin 2.0 Official 50-Task Evaluator")
    sub = p.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="Run parallel multi-GPU evaluation across all 50 tasks (Clean & Randomized)")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--codec-statistics", type=Path, default=None)
    r.add_argument("--robotwin-root", type=Path, default=Path("./third_party/RoboTwin"))
    r.add_argument("--output-dir", type=Path, default=Path("./outputs/robotwin2/eval_p32e16"))
    r.add_argument("--gpus", type=str, default="0")
    r.add_argument("--tasks", type=str, default="")
    r.add_argument("--settings", type=str, default="clean,randomized")
    r.add_argument("--trials", type=int, default=100)
    r.add_argument("--execute-horizon", type=int, default=16)
    r.add_argument("--instruction-type", type=str, default="unseen", choices=("seen", "unseen"))
    r.add_argument("--base-seed", type=int, default=0)
    r.add_argument("--no-bspline", action="store_true")

    w = sub.add_parser("worker", help="Evaluate a single (task, setting) pair on one GPU")
    w.add_argument("--checkpoint", type=Path, required=True)
    w.add_argument("--codec-statistics", type=Path, default=None)
    w.add_argument("--robotwin-root", type=Path, default=Path("./third_party/RoboTwin"))
    w.add_argument("--task", type=str, required=True, choices=CANONICAL_TASKS)
    w.add_argument("--setting", type=str, required=True, choices=SETTINGS)
    w.add_argument("--trials", type=int, default=100)
    w.add_argument("--execute-horizon", type=int, default=16)
    w.add_argument("--instruction-type", type=str, default="unseen", choices=("seen", "unseen"))
    w.add_argument("--base-seed", type=int, default=0)
    w.add_argument("--no-bspline", action="store_true")
    w.add_argument("--output-dir", type=Path, required=True)

    a = sub.add_parser("aggregate", help="Aggregate per-task results into robotwin2_results.json")
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--allow-incomplete", action="store_true")

    s = sub.add_parser("server", help="Start Websocket policy server for RoboTwin bridge")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--codec-statistics", type=Path, default=None)
    s.add_argument("--execute-horizon", type=int, default=16)
    s.add_argument("--no-bspline", action="store_true")
    s.add_argument("--host", type=str, default="127.0.0.1")
    s.add_argument("--port", type=int, default=6694)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "run":
        return run_parallel(args)
    if args.mode == "worker":
        return run_worker(args)
    if args.mode == "aggregate":
        aggregate_results(args.output_dir, allow_incomplete=args.allow_incomplete)
        return 0
    if args.mode == "server":
        return run_server(args)
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
