"""Distributed initialization, deterministic RNG seeding, and structured logging."""
from __future__ import annotations

import datetime
import json
import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as _c10d


def init_distributed(timeout_seconds: int = 3600) -> tuple[int, int, int]:
    """Initialize torch.distributed process group and return (world_size, rank, local_rank)."""
    timeout = datetime.timedelta(
        seconds=int(
            os.environ.get(
                "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
                os.environ.get("NCCL_TIMEOUT", str(timeout_seconds)),
            )
        )
    )
    if hasattr(_c10d, "default_pg_nccl_timeout"):
        _c10d.default_pg_nccl_timeout = timeout
    if hasattr(_c10d, "default_pg_timeout"):
        _c10d.default_pg_timeout = timeout

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, timeout=timeout)
    return world_size, rank, local_rank


def configure_determinism(seed: int, rank: int = 0) -> int:
    """Seed Python, NumPy, PyTorch, and CUDA RNGs deterministically for a given rank."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank_seed = int(seed) + int(rank) * 1_000_003
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    return rank_seed


def virtual_rank_seed(seed: int, rank: int, microstep: int, step: int, world_size: int) -> int:
    """Deterministic per-(rank, microstep, step) flow matching RNG seed."""
    return int(seed) + (int(rank) + int(microstep) * int(world_size)) * 1_000_003 + int(step) * 97_409


def rng_state() -> dict[str, Any]:
    """Capture complete RNG state across Python, NumPy, PyTorch, and CUDA."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    """Restore RNG state captured by ``rng_state``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def log_json(rank: int, payload: dict[str, Any]) -> None:
    if rank == 0:
        print(json.dumps(payload, default=str), flush=True)
