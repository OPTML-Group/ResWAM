"""Distributed FSDP training engine, LR scheduler, and atomic checkpointing for ResWAM."""
from __future__ import annotations

from .checkpoint import init_sync_group, latest_step, load, prune_resume_states, save, save_hashed
from .fsdp import (
    EXPECTED_COUNTS,
    assert_optimizer_fp32_contract,
    assert_recomputation_disabled,
    assert_trainable_parameter_dtype,
    build_optimizer,
    prepare_fp32_trainable_parameters,
    wrap_fsdp,
)
from .schedule import ConstantFloor, WarmupCosine, build_scheduler, warmup_cosine_value

__all__ = [
    "ConstantFloor",
    "EXPECTED_COUNTS",
    "WarmupCosine",
    "assert_optimizer_fp32_contract",
    "assert_recomputation_disabled",
    "assert_trainable_parameter_dtype",
    "build_optimizer",
    "build_scheduler",
    "init_sync_group",
    "latest_step",
    "load",
    "prepare_fp32_trainable_parameters",
    "prune_resume_states",
    "save",
    "save_hashed",
    "warmup_cosine_value",
    "wrap_fsdp",
]
