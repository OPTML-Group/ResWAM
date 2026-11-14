"""Trajectory stationary-segment and joint-outlier cleaning utilities (LiLa-WAM protocol)."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

THRESHOLD = 1e-6
MARGIN = 1
DEFAULT_GRIPPER_INDICES = (6, 13)


def detect_stationary_segments(
    actions: np.ndarray, threshold: float = THRESHOLD, min_duration: int = 1
) -> list[tuple[int, int]]:
    """Detect contiguous segments where max per-step action change is below ``threshold``."""
    total = actions.shape[0]
    if total < 2:
        return []
    diffs = np.abs(np.diff(actions, axis=0))
    max_diff_per_frame = np.max(diffs, axis=1)
    still_flags = max_diff_per_frame < threshold
    segments: list[tuple[int, int]] = []
    i = 0
    while i < len(still_flags):
        if still_flags[i]:
            j = i
            while j < len(still_flags) and still_flags[j]:
                j += 1
            duration = j - i + 1
            if duration >= min_duration:
                segments.append((i, j + 1))
            i = j
        else:
            i += 1
    return segments


def compute_delete_mask(
    total: int, segments: list[tuple[int, int]], margin: int = MARGIN
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Compute boolean keep mask by removing interior frames of stationary segments."""
    delete_mask = np.zeros(total, dtype=bool)
    delete_segments: list[tuple[int, int]] = []
    for segment_start, segment_end in segments:
        actual_start = segment_start + margin
        actual_end = segment_end - margin
        if actual_start >= actual_end:
            continue
        actual_start = max(0, actual_start)
        actual_end = min(total, actual_end)
        delete_mask[actual_start:actual_end] = True
        delete_segments.append((actual_start, actual_end))
    return ~delete_mask, delete_segments


def static_keep_mask(
    actions: np.ndarray,
    *,
    threshold: float = THRESHOLD,
    margin: int = MARGIN,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    values = np.asarray(actions)
    if values.ndim != 2:
        raise ValueError("cleaner requires episode-local [T, D] actions")
    return compute_delete_mask(
        len(values),
        detect_stationary_segments(values, threshold=threshold, min_duration=1),
        margin=margin,
    )


def is_outlier_episode(
    actions_after_static: np.ndarray,
    *,
    gripper_indices: Sequence[int] = DEFAULT_GRIPPER_INDICES,
    limit: float = np.pi,
) -> bool:
    values = np.asarray(actions_after_static)
    if values.ndim != 2:
        raise ValueError("outlier rule requires [T, D] actions")
    non_gripper = [i for i in range(values.shape[1]) if i not in set(gripper_indices)]
    if not non_gripper:
        return False
    joints = values[:, non_gripper]
    return bool(np.any(joints > limit) or np.any(joints < -limit))


@dataclass(frozen=True)
class CleanResult:
    keep_indices: np.ndarray
    delete_segments: tuple[tuple[int, int], ...]
    outlier_episode: bool


def clean_episode(
    actions: np.ndarray,
    *,
    threshold: float = THRESHOLD,
    margin: int = MARGIN,
    gripper_indices: Sequence[int] = DEFAULT_GRIPPER_INDICES,
    check_outliers: bool = True,
) -> CleanResult:
    """Apply stationary frame removal followed by optional joint-range outlier rejection."""
    keep, segments = static_keep_mask(actions, threshold=threshold, margin=margin)
    indices = np.flatnonzero(keep).astype(np.int32)
    outlier = (
        is_outlier_episode(np.asarray(actions)[indices], gripper_indices=gripper_indices)
        if (check_outliers and len(indices) > 0)
        else False
    )
    if outlier:
        indices = np.empty(0, dtype=np.int32)
    return CleanResult(indices, tuple(segments), outlier)


def rule_identity() -> dict[str, Any]:
    local_source = inspect.getsource(detect_stationary_segments)
    local_source += inspect.getsource(compute_delete_mask)
    local_source += inspect.getsource(is_outlier_episode)
    payload = {
        "threshold": THRESHOLD,
        "margin": MARGIN,
        "gripper_indices_participate_in_static": list(DEFAULT_GRIPPER_INDICES),
        "outlier_excluded_indices": list(DEFAULT_GRIPPER_INDICES),
        "order": ["static_cleaner", "outlier_episode_deletion"],
        "local_rule_source_sha256": hashlib.sha256(local_source.encode()).hexdigest(),
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload
