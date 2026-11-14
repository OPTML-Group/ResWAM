"""FSDP FULL_SHARD wrapping and FP32 master parameter / AdamW optimizer contracts."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import torch
from torch import nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

from ..models.dit_modules import StableJointBlock

EXPECTED_COUNTS = {"text": 16, "vision": 24, "expert": 12}


def prepare_fp32_trainable_parameters(system: nn.Module) -> dict[str, int]:
    """Cast all floating trainable parameters to FP32 master weights before FSDP wrapping."""
    with torch.no_grad():
        for parameter in system.parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.to(torch.float32)
    return assert_trainable_parameter_dtype(system, stage="before_fsdp")


def assert_trainable_parameter_dtype(system: nn.Module, *, stage: str) -> dict[str, int]:
    summary: Counter[str] = Counter()
    failures = []
    for name, parameter in system.named_parameters():
        if not parameter.requires_grad or not parameter.is_floating_point():
            continue
        summary[str(parameter.dtype)] += parameter.numel()
        if parameter.dtype != torch.float32:
            failures.append(f"{name}={parameter.dtype}")
    if failures:
        raise RuntimeError(
            f"FP32 trainable-parameter contract failed at {stage}: " + ", ".join(failures[:20])
        )
    if not summary:
        raise RuntimeError(f"no floating trainable parameters found at {stage}")
    return dict(sorted(summary.items()))


def build_optimizer(
    vlm: Iterable[nn.Parameter],
    expert: Iterable[nn.Parameter],
    *,
    vlm_peak_lr: float = 1e-5,
    vlm_min_lr: float = 1e-6,
    expert_peak_lr: float = 1e-4,
    expert_min_lr: float = 1e-5,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 1e-8,
) -> torch.optim.AdamW:
    """Construct two-group AdamW optimizer (VLM 1e-5 -> 1e-6, Expert 1e-4 -> 1e-5)."""
    groups = [
        {
            "name": "vlm",
            "params": list(vlm),
            "peak_lr": float(vlm_peak_lr),
            "min_lr": float(vlm_min_lr),
            "lr": 0.0,
        },
        {
            "name": "expert",
            "params": list(expert),
            "peak_lr": float(expert_peak_lr),
            "min_lr": float(expert_min_lr),
            "lr": 0.0,
        },
    ]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    if any(not group["params"] for group in groups) or len(ids) != len(set(ids)):
        raise ValueError("optimizer groups must be nonempty and disjoint")
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        groups,
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(weight_decay),
        fused=use_fused,
    )


def assert_optimizer_fp32_contract(
    optimizer: torch.optim.Optimizer, *, require_initialized: bool
) -> dict[str, Any]:
    """Verify AdamW parameters and both moments are FP32."""
    parameter_summary: Counter[str] = Counter()
    moment_summary: Counter[str] = Counter()
    parameter_failures = []
    moment_failures = []
    initialized = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not parameter.is_floating_point():
                continue
            parameter_summary[str(parameter.dtype)] += parameter.numel()
            if parameter.dtype != torch.float32:
                parameter_failures.append(f"{group.get('name', '<unnamed>')}={parameter.dtype}")
            state = optimizer.state.get(parameter, {})
            has_first = "exp_avg" in state
            has_second = "exp_avg_sq" in state
            if require_initialized and parameter.numel() > 0 and not has_first:
                moment_failures.append(
                    f"{group.get('name', '<unnamed>')} parameter has no Adam moments"
                )
            if has_first != has_second:
                moment_failures.append("incomplete Adam moment pair")
            if has_first:
                initialized += 1
                for key in ("exp_avg", "exp_avg_sq"):
                    value = state[key]
                    moment_summary[f"{key}:{value.dtype}"] += value.numel()
                    if value.dtype != torch.float32:
                        moment_failures.append(f"{key}={value.dtype}")
    if parameter_failures:
        raise RuntimeError(
            "FP32 optimizer-parameter contract failed: " + ", ".join(parameter_failures[:20])
        )
    if moment_failures or (require_initialized and initialized == 0):
        detail = moment_failures[:20] or ["Adam moments were not initialized"]
        raise RuntimeError("FP32 optimizer-moment contract failed: " + ", ".join(detail))
    return {
        "parameters": dict(sorted(parameter_summary.items())),
        "moments": dict(sorted(moment_summary.items())),
        "initialized_parameter_states": initialized,
    }


def assert_recomputation_disabled(system: nn.Module) -> dict[str, int]:
    wrappers = [
        name
        for name, module in system.named_modules()
        if "CheckpointWrapper" in type(module).__name__
        or "checkpoint_wrapper" in type(module).__name__.lower()
    ]
    flags = [
        name
        for name, module in system.named_modules()
        if any(
            bool(getattr(module, flag, False))
            for flag in (
                "is_gradient_checkpointing",
                "gradient_checkpointing",
                "activation_checkpointing",
                "selective_recompute",
            )
        )
    ]
    counts = {"text": 0, "vision": 0, "expert": 0}
    for module in system.modules():
        name = type(module).__name__
        counts["text"] += int(name.endswith("DecoderLayer"))
        counts["vision"] += int(
            "VisionBlock" in name or name.endswith("VisionTransformerBlock")
        )
        counts["expert"] += int(isinstance(module, StableJointBlock))
    if wrappers or flags or counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"recomputation fail-closed: wrappers={wrappers}, flags={flags}, counts={counts}"
        )
    return counts


def wrap_fsdp(
    system: nn.Module,
    local_rank: int,
    *,
    activation_checkpointing: bool = False,
) -> FSDP:
    """Wrap ResWAMSystem with FSDP FULL_SHARD (bf16 compute/reduce, FP32 master params, DINO ignored)."""
    if activation_checkpointing:
        raise RuntimeError("activation checkpointing is disabled in ResWAM")
    if any(parameter.requires_grad for parameter in system.dino.parameters()):
        raise RuntimeError("DINO encoder must remain frozen")
    assert_recomputation_disabled(system)
    kwargs = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        "cpu_offload": CPUOffload(offload_params=False),
        "device_id": torch.device("cuda", local_rank),
        "use_orig_params": True,
        "limit_all_gathers": True,
        "forward_prefetch": False,
        "sync_module_states": True,
    }
    wrapped = {"text": 0, "vision": 0, "expert": 0}

    def recurse(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            recurse(child)
            class_name = type(child).__name__
            kind = None
            if isinstance(child, StableJointBlock):
                kind = "expert"
            elif class_name.endswith("DecoderLayer"):
                kind = "text"
            elif "VisionBlock" in class_name or class_name.endswith("VisionTransformerBlock"):
                kind = "vision"
            if kind and any(parameter.requires_grad for parameter in child.parameters()):
                setattr(module, name, FSDP(child, **kwargs))
                wrapped[kind] += 1

    recurse(system)
    if wrapped != EXPECTED_COUNTS:
        raise RuntimeError(f"FSDP boundary mismatch: {wrapped}")
    result = FSDP(system, ignored_modules=[system.dino], **kwargs)
    assert_recomputation_disabled(result)
    return result
