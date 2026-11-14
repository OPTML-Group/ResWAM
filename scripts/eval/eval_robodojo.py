#!/usr/bin/env python3
"""Official 42-task (54-config, 2,100-episode) RoboDojo evaluation and aggregation script for ResWAM.

Protocol:
  - 42 manipulation tasks across 5 capability dimensions:
      1. Generalization (12 tasks x {standard, _random} x 25 episodes = 600 episodes)
      2. Memory (6 tasks x 50 episodes = 300 episodes)
      3. Precision (8 tasks x 50 episodes = 400 episodes)
      4. Long-Horizon (8 tasks x 50 episodes = 400 episodes)
      5. Open (8 tasks x 50 episodes = 400 episodes)
  - Total: 54 runnable configurations, 2,100 episodes at seed 0
  - ARX X5 14-D bimanual joint control, H = 32 prediction, execute_horizon = 10 (p32e10)
  - Overall SR and Score are computed as the unweighted mean across the 5 capability dimensions.

Supports:
  - Installing the ResWAM XPolicyLab adapter into a RoboDojo checkout and running all 54 configs (`run` mode)
  - Evaluating a single RoboDojo config (`worker` mode)
  - Standalone Websocket policy server (`server` mode)
  - Aggregating all 54 config results into `robodojo_results.json` (`aggregate` mode)

Usage:
    python scripts/eval/eval_robodojo.py run \
        --checkpoint /path/to/model.pt \
        --robodojo-root ./third_party/RoboDojo \
        --output-dir ./outputs/robodojo/eval_p32e10 \
        --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reswam.data.codec import ZScoreCodec  # noqa: E402
from reswam.models.system import ResWAMSystem, system_config  # noqa: E402
from reswam.utils.dist import configure_determinism  # noqa: E402
from reswam.utils.websocket_policy import WebsocketPolicyServer  # noqa: E402

ROBODOJO_CAMERAS = {
    "head": "cam_head",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}
CAMERA_ORDER = ("head", "left_wrist", "right_wrist")
GRIPPERS = (6, 13)
HORIZON = 32
PAIRED_HALF, STANDALONE = 25, 50
# Noise-seed base used by the official ResWAM RoboDojo evaluation (deploy.yml ``base_seed``).
DEFAULT_BASE_SEED = 2609


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()

DIMENSIONS: dict[str, tuple[str, ...]] = {
    "generalization": (
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ),
    "memory": (
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ),
    "precision": (
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ),
    "long-horizon": (
        "fill_pen_holder",
        "classify_objects",
        "put_bottles_into_dustbin",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ),
    "open": (
        "align_blocks",
        "general_pickup",
        "solve_equation",
        "stack_blocks_by_language",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ),
}
GENERALIZATION = set(DIMENSIONS["generalization"])
CONFIGS: tuple[str, ...] = tuple(
    config
    for tasks in DIMENSIONS.values()
    for task in tasks
    for config in ((task, f"{task}_random") if task in GENERALIZATION else (task,))
)


def expected_episodes(config: str) -> int:
    return PAIRED_HALF if config.removesuffix("_random") in GENERALIZATION else STANDALONE


def derive_seed(**fields: Any) -> int:
    raw = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") >> 1


def pack_state(state: dict[str, Any]) -> np.ndarray:
    parts = [
        np.asarray(state[key], dtype=np.float32).reshape(-1)
        for key in (
            "left_arm_joint_state",
            "left_ee_joint_state",
            "right_arm_joint_state",
            "right_ee_joint_state",
        )
    ]
    packed = np.concatenate(parts)
    if packed.shape != (14,):
        raise RuntimeError(f"RoboDojo arx_x5 joint state must be 14-D, got {packed.shape}")
    return packed


def unpack_action(row: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "left_arm_joint_state": row[0:6],
        "left_ee_joint_state": row[6:7],
        "right_arm_joint_state": row[7:13],
        "right_ee_joint_state": row[13:14],
    }


class RoboDojoPolicy:
    """ResWAM policy wrapper for RoboDojo ARX X5 evaluation."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        codec_path: Path | None = None,
        execute_horizon: int = 10,
        device: str = "cuda",
    ) -> None:
        configure_determinism(20260907)
        self.checkpoint = checkpoint.expanduser().resolve()
        # The per-chunk flow-noise seed includes the checkpoint SHA256 (official protocol).
        self.checkpoint_sha256 = os.environ.get("RESWAM_CHECKPOINT_SHA256") or sha256_file(
            self.checkpoint
        )
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if codec_path is not None:
            self.codec = ZScoreCodec.from_file(codec_path)
        elif "codec" in payload:
            self.codec = ZScoreCodec.from_metadata(payload["codec"])
        elif (self.checkpoint.parent / "codec_statistics.json").is_file():
            self.codec = ZScoreCodec.from_file(self.checkpoint.parent / "codec_statistics.json")
        else:
            raise RuntimeError(
                "Checkpoint lacks codec metadata and no codec_statistics.json sits next to it; "
                "pass --codec-statistics"
            )

        cfg = payload.get("config", {})
        sys_cfg = cfg.get("system")
        if sys_cfg is None:
            sys_cfg = system_config(
                action_dim=14,
                state_dim=14,
                horizon=HORIZON,
                views=CAMERA_ORDER,
                world_views=("head",),
                world_loss_weight=0.1,
                action_tau_shift=1.0,
                world_tau_shift=5.0,
                solver_steps=10,
                state_dropout_prob=0.0,
                prompt_style="multi_camera",
            )
        self.system = ResWAMSystem(sys_cfg)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.system.load_state_dict(state_dict, strict=True)
        self.device = torch.device(device)
        self.system.to(self.device).eval()
        self.execute_horizon = int(execute_horizon)

    @torch.inference_mode()
    def predict(
        self,
        observations: Sequence[dict[str, Any]],
        seeds: Sequence[int],
    ) -> np.ndarray:
        cameras = {
            name: [
                np.asarray(obs["vision"][key]["color"], dtype=np.uint8)[..., :3]
                for obs in observations
            ]
            for name, key in ROBODOJO_CAMERAS.items()
        }
        cameras = {name: cameras[name] for name in CAMERA_ORDER}
        state = np.stack([pack_state(obs["state"]) for obs in observations])
        normalized = self.system.sample_actions(
            cameras=cameras,
            instructions=[str(obs["instruction"]) for obs in observations],
            normalized_state=torch.from_numpy(self.codec.encode_state(state).astype(np.float32)).to(
                self.device
            ),
            seeds=list(seeds),
        )
        chunks = np.stack(
            [self.codec.decode_action(row.float().cpu().numpy()) for row in normalized]
        ).astype(np.float32)
        if chunks.shape != (len(observations), HORIZON, 14) or not np.isfinite(chunks).all():
            raise RuntimeError(f"bad action chunk {chunks.shape}")
        chunks[..., GRIPPERS] = np.clip(chunks[..., GRIPPERS], 0.0, 1.0)
        return chunks

    def predict_action(self, **msg: Any) -> dict[str, Any]:
        observations = msg.get("observations", [msg])
        seeds = msg.get("seeds", [0] * len(observations))
        chunks = self.predict(observations, seeds)
        return {"actions": chunks[:, : self.execute_horizon]}


_POLICY_SERVER_SH = """#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"
: "${RESWAM_CHECKPOINT:?}"
RESWAM_REPO="${RESWAM_REPO:-@REPO_ROOT@}"

exec env \\
  PYTHONWARNINGS=ignore::UserWarning \\
  PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${RESWAM_REPO}" \\
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \\
  "${policy_env}/bin/python" "${XPL_ROOT}/setup_policy_server.py" \\
    --config_path "${yaml_file}" \\
    --overrides \\
      port="${policy_server_port}" \\
      host="${policy_server_host}" \\
      bench_name="${bench_name}" \\
      task_name="${task_name}" \\
      ckpt_name="${ckpt_name}" \\
      env_cfg_type="${env_cfg_type}" \\
      seed="${seed}" \\
      policy_name="${policy_name}" \\
      action_type="${action_type}"
"""

_ENV_CLIENT_SH = """#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

bash "${UTILS_DIR}/setup_env_client.sh" \\
  "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" \\
  "${policy_server_port}" "${bench_name}" "${task_name}" \\
  "${env_cfg_type}" "${policy_name}" "${additional_info}" \\
  "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
"""

_EVAL_SH = """#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
env_gpu_id=${8}
policy_conda_env=${9}
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \\
  "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \\
  "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \\
  "${policy_server_port}" &
SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" \\
  "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \\
  "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \\
  "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \\
  "${additional_info}" "${policy_server_port}" "${policy_server_ip}"
"""


def install_xpolicylab_adapter(robodojo_root: Path, execute_horizon: int) -> Path:
    """Install the ResWAM policy adapter into ``<robodojo_root>/XPolicyLab/policy/ResWAM``."""
    target = robodojo_root / "XPolicyLab/policy/ResWAM"
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").write_text("from .model import Model\n__all__ = ['Model']\n")
    (target / "deploy.yml").write_text(
        "policy_name: ResWAM\n"
        "protocol: ws\n"
        "host: localhost\n"
        "port: null\n"
        "bench_name: null\n"
        "task_name: null\n"
        "ckpt_name: null\n"
        "env_cfg_type: null\n"
        "seed: null\n"
        "action_type: joint\n"
        "gpu_id: null\n"
        "eval_batch: true\n"
        f"execute_horizon: {int(execute_horizon)}\n"
        f"base_seed: {DEFAULT_BASE_SEED}\n"
    )
    # XPolicyLab launcher contract used by RoboDojo's scripts/internal/run_policy_eval.sh.
    # The policy server runs from the ResWAM environment (``POLICY_ENV``: a venv/conda prefix,
    # invoked as ``$POLICY_ENV/bin/python``); the simulator client runs in RoboDojo's Isaac Sim
    # conda environment (``EVAL_ENV``), exactly as in the official evaluation.
    (target / "setup_eval_policy_server.sh").write_text(
        _POLICY_SERVER_SH.replace("@REPO_ROOT@", str(REPO_ROOT))
    )
    (target / "setup_eval_env_client.sh").write_text(_ENV_CLIENT_SH)
    (target / "eval.sh").write_text(_EVAL_SH)
    for name in ("setup_eval_policy_server.sh", "setup_eval_env_client.sh", "eval.sh"):
        (target / name).chmod(0o755)
    (target / "deploy.py").write_text(
        """def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_actions = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_actions, env_idx_list)
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [index for index, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[index] for index in active]
            env_idx_list = [env_idx_list[index] for index in active]
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
"""
    )
    (target / "model.py").write_text(
        f"""from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, {repr(str(REPO_ROOT))})
from XPolicyLab.model_template import ModelTemplate
from scripts.eval.eval_robodojo import DEFAULT_BASE_SEED, RoboDojoPolicy, derive_seed, unpack_action

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.execute_horizon = int(model_cfg.get("execute_horizon", {int(execute_horizon)}))
        codec_env = os.environ.get("RESWAM_CODEC_STATISTICS")
        self.policy = RoboDojoPolicy(
            Path(os.environ["RESWAM_CHECKPOINT"]),
            codec_path=Path(codec_env) if codec_env else None,
            execute_horizon=self.execute_horizon,
        )
        self.seed_fields = {{
            "base_seed": int(
                DEFAULT_BASE_SEED if model_cfg.get("base_seed") is None else model_cfg["base_seed"]
            ),
            "task": str(model_cfg.get("task_name", "")),
            "eval_seed": str(model_cfg.get("seed", 0)),
            "checkpoint_sha256": self.policy.checkpoint_sha256,
        }}
        self.observations = {{}}
        self.calls = {{}}
        self.resets = 0

    def update_obs(self, obs):
        self.observations[int(obs.get("env_idx", 0))] = obs

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self.update_obs(obs)

    def _act(self, env_indices):
        obs_seq = [self.observations[i] for i in env_indices]
        seeds = []
        for i in env_indices:
            c = self.calls.get(i, 0)
            self.calls[i] = c + 1
            seeds.append(derive_seed(**self.seed_fields, reset=self.resets, env_idx=i, call=c))
        chunks = self.policy.predict(obs_seq, seeds)
        return [[unpack_action(row) for row in chunk[:self.execute_horizon]] for chunk in chunks]

    def get_action(self):
        return self._act(list(self.observations))[0]

    def get_action_batch(self, env_idx_list=None):
        indices = list(env_idx_list) if env_idx_list is not None else sorted(self.observations)
        return self._act([int(i) for i in indices])

    def reset(self):
        self.observations.clear()
        self.calls.clear()
        self.resets += 1
"""
    )
    return target


def find_latest_config_result(
    robodojo_root: Path, config: str, ckpt_tag: str, eval_seed: int = 0
) -> dict[str, Any] | None:
    rdir = (
        robodojo_root
        / "eval_result/RoboDojo"
        / config
        / "ResWAM/arx_x5"
        / f"{eval_seed}_ckpt_name={ckpt_tag},action_type=joint"
    )
    runs = sorted(rdir.glob("*/_result.json")) if rdir.is_dir() else []
    if not runs:
        return None
    payload = json.loads(runs[-1].read_text())
    if len(payload.get("details", {})) < expected_episodes(config):
        return None
    return payload


def aggregate_results(
    output_dir: Path,
    robodojo_root: Path,
    *,
    ckpt_tag: str = "reswam_p32e10",
    execute_horizon: int = 10,
    eval_seed: int = 0,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    def _stats(details: dict[str, Any], count: int) -> tuple[float, float]:
        rows = [details[k] for k in sorted(details, key=int)][:count]
        if not rows:
            return 0.0, 0.0
        return (
            100.0 * sum(bool(r["success"]) for r in rows) / len(rows),
            100.0 * sum(float(r["score"]) for r in rows) / len(rows),
        )

    tasks: dict[str, dict[str, Any]] = {}
    for dimension, names in DIMENSIONS.items():
        for task in names:
            if task in GENERALIZATION:
                base_res = find_latest_config_result(robodojo_root, task, ckpt_tag, eval_seed)
                rand_res = find_latest_config_result(
                    robodojo_root, f"{task}_random", ckpt_tag, eval_seed
                )
                if base_res is None or rand_res is None:
                    if not allow_incomplete:
                        raise RuntimeError(f"Missing RoboDojo results for {task} or {task}_random")
                    continue
                base = base_res["details"]
                rand = rand_res["details"]
                base_rows = [base[k] for k in sorted(base, key=int)][:PAIRED_HALF]
                rand_rows = [rand[k] for k in sorted(rand, key=int)][:PAIRED_HALF]
                merged = {str(i): r for i, r in enumerate(base_rows + rand_rows)}
                sr, score = _stats(merged, 2 * PAIRED_HALF)
                sr_std, _ = _stats(base, PAIRED_HALF)
                sr_rand, _ = _stats(rand, PAIRED_HALF)
                tasks[task] = {
                    "dimension": dimension,
                    "sr": sr,
                    "score": score,
                    "sr_standard": sr_std,
                    "sr_random": sr_rand,
                }
            else:
                res = find_latest_config_result(robodojo_root, task, ckpt_tag, eval_seed)
                if res is None:
                    if not allow_incomplete:
                        raise RuntimeError(f"Missing RoboDojo result for {task}")
                    continue
                sr, score = _stats(res["details"], STANDALONE)
                tasks[task] = {"dimension": dimension, "sr": sr, "score": score}

    dims: dict[str, dict[str, float]] = {}
    for dimension, names in DIMENSIONS.items():
        present = [t for t in names if t in tasks]
        if present:
            dims[dimension] = {
                "sr": sum(tasks[t]["sr"] for t in present) / len(present),
                "score": sum(tasks[t]["score"] for t in present) / len(present),
            }

    overall_sr = sum(d["sr"] for d in dims.values()) / max(1, len(dims))
    overall_score = sum(d["score"] for d in dims.values()) / max(1, len(dims))
    result = {
        "schema_version": "reswam-robodojo-eval-result-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "protocol": "RoboDojo official 42 tasks (54 configs, 2100 episodes), seed 0, unweighted mean of 5 dimension means",
        "execute_horizon": int(execute_horizon),
        "complete": len(tasks) == 42,
        "overall": {"sr": overall_sr, "score": overall_score},
        "dimensions": dims,
        "tasks": tasks,
    }
    out_file = output_dir / "robodojo_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "robodojo_aggregated", "output": str(out_file), "overall": result["overall"]}), flush=True)
    return result


def run_parallel(args: argparse.Namespace) -> int:
    rd_root = Path(args.robodojo_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = install_xpolicylab_adapter(rd_root, int(args.execute_horizon))
    ckpt_tag = args.ckpt_tag or f"reswam_p32e{args.execute_horizon}"
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    configs = args.configs.split(",") if args.configs else list(CONFIGS)

    pending = [
        cfg
        for cfg in configs
        if find_latest_config_result(rd_root, cfg, ckpt_tag, int(args.eval_seed)) is None
    ]
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_sha256 = os.environ.get("RESWAM_CHECKPOINT_SHA256") or sha256_file(checkpoint_path)
    print(f"[eval_robodojo] Scheduling {len(pending)} pending configs across GPUs {gpus}")

    active: dict[int, tuple[subprocess.Popen, str]] = {}
    idx = 0
    failed = 0

    while idx < len(pending) or active:
        for gpu in list(active):
            proc, cfg_name = active[gpu]
            rc = proc.poll()
            if rc is not None:
                del active[gpu]
                if rc != 0:
                    failed += 1
                    print(f"[eval_robodojo] Config {cfg_name} failed on GPU {gpu} (rc={rc})")

        for gpu in gpus:
            if gpu not in active and idx < len(pending):
                cfg_name = pending[idx]
                idx += 1
                env = dict(os.environ)
                env.update(
                    {
                        "RESWAM_CHECKPOINT": str(checkpoint_path),
                        "RESWAM_CHECKPOINT_SHA256": checkpoint_sha256,
                        "OMNI_KIT_ACCEPT_EULA": "YES",
                        "ACCEPT_EULA": "Y",
                        "PRIVACY_CONSENT": "Y",
                    }
                )
                if args.codec_statistics:
                    env["RESWAM_CODEC_STATISTICS"] = str(
                        Path(args.codec_statistics).expanduser().resolve()
                    )
                cmd = [
                    "bash",
                    str(rd_root / "scripts/internal/run_policy_eval.sh"),
                    str(policy_dir),
                    "RoboDojo",
                    cfg_name,
                    ckpt_tag,
                    "arx_x5",
                    "joint",
                    str(args.eval_seed),
                    str(gpu),
                    str(gpu),
                    str(Path(args.policy_env).expanduser().resolve()),
                    str(args.sim_env),
                ]
                log_fh = (out_dir / f"{cfg_name}.log").open("a")
                proc = subprocess.Popen(cmd, cwd=rd_root, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
                log_fh.close()
                active[gpu] = (proc, cfg_name)
        time.sleep(2.0)

    if failed > 0:
        return 1
    aggregate_results(
        out_dir,
        rd_root,
        ckpt_tag=ckpt_tag,
        execute_horizon=int(args.execute_horizon),
        eval_seed=int(args.eval_seed),
        allow_incomplete=len(configs) < 54,
    )
    return 0


def run_server(args: argparse.Namespace) -> int:
    policy = RoboDojoPolicy(
        Path(args.checkpoint),
        codec_path=Path(args.codec_statistics) if args.codec_statistics else None,
        execute_horizon=int(args.execute_horizon),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    metadata = {
        "schema_version": "reswam-robodojo-server-v1",
        "prediction_horizon": HORIZON,
        "execute_horizon": int(args.execute_horizon),
        "camera_order": list(CAMERA_ORDER),
    }
    WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, idle_timeout=-1, metadata=metadata
    ).serve_forever()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ResWAM RoboDojo Official 42-Task (54-Config) Evaluator")
    sub = p.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="Run parallel multi-GPU evaluation across all 54 RoboDojo configs")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--codec-statistics", type=Path, default=None)
    r.add_argument("--robodojo-root", type=Path, default=Path("./third_party/RoboDojo"))
    r.add_argument(
        "--policy-env",
        type=Path,
        default=Path("./envs/reswam"),
        help="ResWAM policy environment prefix (venv or conda prefix; $POLICY_ENV/bin/python)",
    )
    r.add_argument(
        "--sim-env",
        type=str,
        default="RoboDojo",
        help="RoboDojo Isaac Sim conda environment (name or prefix) created by scripts/install.sh",
    )
    r.add_argument("--output-dir", type=Path, default=Path("./outputs/robodojo/eval_p32e10"))
    r.add_argument("--gpus", type=str, default="0")
    r.add_argument("--configs", type=str, default="")
    r.add_argument("--execute-horizon", type=int, default=10)
    r.add_argument("--eval-seed", type=int, default=0)
    r.add_argument("--ckpt-tag", type=str, default="")

    a = sub.add_parser("aggregate", help="Aggregate 54-config results into robodojo_results.json")
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--robodojo-root", type=Path, default=Path("./third_party/RoboDojo"))
    a.add_argument("--ckpt-tag", type=str, default="reswam_p32e10")
    a.add_argument("--execute-horizon", type=int, default=10)
    a.add_argument("--eval-seed", type=int, default=0)
    a.add_argument("--allow-incomplete", action="store_true")

    s = sub.add_parser("server", help="Start Websocket policy server for RoboDojo")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--codec-statistics", type=Path, default=None)
    s.add_argument("--execute-horizon", type=int, default=10)
    s.add_argument("--host", type=str, default="127.0.0.1")
    s.add_argument("--port", type=int, default=10095)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "run":
        return run_parallel(args)
    if args.mode == "aggregate":
        aggregate_results(
            args.output_dir,
            args.robodojo_root,
            ckpt_tag=args.ckpt_tag,
            execute_horizon=args.execute_horizon,
            eval_seed=args.eval_seed,
            allow_incomplete=args.allow_incomplete,
        )
        return 0
    if args.mode == "server":
        return run_server(args)
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
