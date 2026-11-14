"""Action and state normalization codecs for ResWAM across LIBERO, RoboTwin 2.0, and RoboDojo."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CODEC_VERSION = "reswam-absolute-zscore-joints-affine-grippers-v1"
STATISTICS_SCHEMA = "reswam-zscore-statistics-v1"
LIBERO_CODEC_VERSION = "reswam-libero-delta-eef-zscore6-binary-gripper-pm1-v1"
LIBERO_STATE_FEATURIZATION = "reswam-libero-state11-xyz-rot6d-fingers2-v1"
DEFAULT_ENCODE_CLAMP_SIGMA = 12.0
DEFAULT_LIBERO_STATE_CLAMP_SIGMA = 10.0
MIN_STD = 1e-6


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues' formula in float64 mapping [N, 3] axis-angle to [N, 3, 3] rotation matrices."""
    vector = np.asarray(axis_angle, dtype=np.float64)
    if vector.ndim != 2 or vector.shape[1] != 3:
        raise ValueError("axis-angle must be [N, 3]")
    angle = np.linalg.norm(vector, axis=1)
    small = angle < 1e-12
    axis = vector / np.where(small, 1.0, angle)[:, None]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=1).reshape(-1, 3, 3)
    sin = np.sin(angle)[:, None, None]
    cos = np.cos(angle)[:, None, None]
    matrix = np.eye(3)[None] + sin * skew + (1.0 - cos) * (skew @ skew)
    if small.any():
        vx, vy, vz = vector[small, 0], vector[small, 1], vector[small, 2]
        zs = np.zeros_like(vx)
        first_order = np.stack([zs, -vz, vy, vz, zs, -vx, -vy, vx, zs], axis=1).reshape(-1, 3, 3)
        matrix[small] = np.eye(3)[None] + first_order
    return matrix


def featurize_libero_state_11d(raw: np.ndarray) -> np.ndarray:
    """Map raw 8-D LIBERO state [xyz(3), axis_angle(3), gripper_qpos(2)] to 11-D [xyz(3), rot6d(6), fingers(2)]."""
    arr = np.asarray(raw, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"raw LIBERO state must have shape [..., 8], got {raw.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("raw LIBERO state contains non-finite values")
    matrix = axis_angle_to_matrix(arr[:, 3:6])
    rotation6d = np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], axis=1)
    out = np.concatenate([arr[:, :3], rotation6d, arr[:, 6:8]], axis=1)
    return out[0] if single else out


def build_libero_state_statistics(
    raw: np.ndarray, *, clamp_sigma: float = DEFAULT_LIBERO_STATE_CLAMP_SIGMA
) -> dict[str, Any]:
    """Compute float64 population mean and std for 11-D featurized LIBERO proprioceptive state."""
    features = featurize_libero_state_11d(np.asarray(raw, dtype=np.float32))
    mean = features.mean(axis=0)
    std = np.where(features.std(axis=0) < MIN_STD, 1.0, features.std(axis=0))
    return {
        "featurization": LIBERO_STATE_FEATURIZATION,
        "frames": int(len(features)),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "clamp_sigma": float(clamp_sigma),
    }


@dataclass(frozen=True)
class StateNormalizer11D:
    """Normalizer for 11-D LIBERO proprioceptive state (3D pos + 6D rot + 2D finger qpos)."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    clamp_sigma: float = DEFAULT_LIBERO_STATE_CLAMP_SIGMA

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateNormalizer11D":
        return cls(
            tuple(float(x) for x in payload["mean"]),
            tuple(float(x) for x in payload["std"]),
            float(payload.get("clamp_sigma", DEFAULT_LIBERO_STATE_CLAMP_SIGMA)),
        )

    def encode(self, raw_state: np.ndarray) -> np.ndarray:
        feat = featurize_libero_state_11d(raw_state)
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        z = (feat - mean) / std
        return np.clip(z, -self.clamp_sigma, self.clamp_sigma).astype(np.float32)

    def __call__(self, raw_state: np.ndarray) -> np.ndarray:
        return self.encode(raw_state)


@dataclass(frozen=True)
class LiberoZScoreCodec:
    """7-D LIBERO delta-EEF action codec (z-score on dims 0..5, binary +-1 gripper on dim 6)."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    state_normalizer: StateNormalizer11D | None = None
    encode_clamp_sigma: float = DEFAULT_ENCODE_CLAMP_SIGMA

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def state_dim(self) -> int:
        return 11 if self.state_normalizer is not None else 8

    def encode_action(self, raw_actions: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_actions, dtype=np.float32)
        if raw.shape[-1] != 7 or not np.isfinite(raw).all():
            raise ValueError("LIBERO raw actions must be finite [..., 7]")
        output = np.empty_like(raw)
        z = (raw[..., :6].astype(np.float64) - np.asarray(self.mean, np.float64)) / np.asarray(
            self.std, np.float64
        )
        output[..., :6] = np.clip(z, -self.encode_clamp_sigma, self.encode_clamp_sigma).astype(
            np.float32
        )
        output[..., 6] = np.where(raw[..., 6] > 0.5, 1.0, -1.0).astype(np.float32)
        return output

    def decode_action(self, normalized_actions: np.ndarray) -> np.ndarray:
        normalized = np.asarray(normalized_actions, dtype=np.float32)
        if normalized.shape[-1] != 7 or not np.isfinite(normalized).all():
            raise ValueError("LIBERO normalized actions must be finite [..., 7]")
        output = np.empty_like(normalized)
        output[..., :6] = (
            normalized[..., :6].astype(np.float64) * np.asarray(self.std, np.float64)
            + np.asarray(self.mean, np.float64)
        ).astype(np.float32)
        output[..., 6] = np.where(normalized[..., 6] >= 0.0, 1.0, -1.0).astype(np.float32)
        return output

    def encode_state(self, raw_state: np.ndarray) -> np.ndarray:
        if self.state_normalizer is not None:
            return self.state_normalizer.encode(raw_state)
        return np.asarray(raw_state, dtype=np.float32)

    @property
    def metadata(self) -> dict[str, Any]:
        base = {
            "version": LIBERO_CODEC_VERSION,
            "mean": list(self.mean),
            "std": list(self.std),
            "encode_clamp_sigma": self.encode_clamp_sigma,
            "state_11d": (
                {
                    "mean": list(self.state_normalizer.mean),
                    "std": list(self.state_normalizer.std),
                    "clamp_sigma": self.state_normalizer.clamp_sigma,
                }
                if self.state_normalizer is not None
                else None
            ),
        }
        return {**base, "contract_sha256": _sha(base)}


@dataclass(frozen=True)
class ZScoreCodec:
    """Dimension-generic z-score joint + affine/binary gripper codec for ResWAM.

    Joint dimensions are z-score normalized using population mean and std in float64 and clamped
    to ``+-encode_clamp_sigma`` (or ``+-state_clamp_sigma`` for state) during encoding.
    Gripper dimensions support:
      - ``gripper_mode="affine"`` (default): mapped affinely from ``[lo, hi]`` to ``[-1, 1]`` and
        clipped on encode; decoding never clips.
      - ``gripper_mode="binary_pm1"`` (LIBERO): ``raw > 0.5 -> +1.0, else -1.0`` on encode;
        ``normalized >= 0.0 -> +1.0, else -1.0`` on decode.
    """

    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    action_gripper_range: tuple[tuple[float, float], ...]
    state_gripper_range: tuple[tuple[float, float], ...]
    gripper_indices: tuple[int, ...]
    state_gripper_indices: tuple[int, ...]
    encode_clamp_sigma: float = DEFAULT_ENCODE_CLAMP_SIGMA
    state_clamp_sigma: float | None = None
    state_featurization: str = "identity"
    gripper_mode: str = "affine"
    source: str = ""

    @classmethod
    def from_statistics(cls, payload: dict[str, Any]) -> "ZScoreCodec":
        if payload.get("schema_version") != STATISTICS_SCHEMA:
            raise RuntimeError(f"Unsupported statistics schema: {payload.get('schema_version')}")
        featurization = str(payload.get("state_featurization", "identity"))
        default_state_clamp = (
            DEFAULT_LIBERO_STATE_CLAMP_SIGMA
            if featurization == "xyz_rot6d_fingers2_11d"
            else float(payload.get("encode_clamp_sigma", DEFAULT_ENCODE_CLAMP_SIGMA))
        )
        return cls(
            tuple(payload["action"]["mean"]),
            tuple(payload["action"]["std"]),
            tuple(payload["state"]["mean"]),
            tuple(payload["state"]["std"]),
            tuple(map(tuple, payload["action"]["gripper_range"])),
            tuple(map(tuple, payload["state"]["gripper_range"])),
            tuple(payload["gripper_indices"]),
            tuple(payload["state_gripper_indices"]),
            float(payload.get("encode_clamp_sigma", DEFAULT_ENCODE_CLAMP_SIGMA)),
            float(payload.get("state_clamp_sigma", default_state_clamp)),
            featurization,
            str(
                payload.get(
                    "gripper_mode",
                    "binary_pm1" if featurization == "xyz_rot6d_fingers2_11d" else "affine",
                )
            ),
            str(payload.get("source", "")),
        )

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "ZScoreCodec":
        featurization = str(payload.get("state_featurization", "identity"))
        default_state_clamp = (
            DEFAULT_LIBERO_STATE_CLAMP_SIGMA
            if featurization == "xyz_rot6d_fingers2_11d"
            else float(payload.get("encode_clamp_sigma", DEFAULT_ENCODE_CLAMP_SIGMA))
        )
        return cls(
            tuple(payload["action_mean"]),
            tuple(payload["action_std"]),
            tuple(payload["state_mean"]),
            tuple(payload["state_std"]),
            tuple(
                map(
                    tuple,
                    payload.get(
                        "action_gripper_range",
                        [[0.0, 1.0]] * len(payload["gripper_indices"]),
                    ),
                )
            ),
            tuple(
                map(
                    tuple,
                    payload.get(
                        "state_gripper_range",
                        [[0.0, 1.0]]
                        * len(payload.get("state_gripper_indices", payload["gripper_indices"])),
                    ),
                )
            ),
            tuple(payload["gripper_indices"]),
            tuple(payload.get("state_gripper_indices", payload["gripper_indices"])),
            float(payload.get("encode_clamp_sigma", DEFAULT_ENCODE_CLAMP_SIGMA)),
            float(payload.get("state_clamp_sigma", default_state_clamp)),
            featurization,
            str(
                payload.get(
                    "gripper_mode",
                    "binary_pm1" if featurization == "xyz_rot6d_fingers2_11d" else "affine",
                )
            ),
            str(payload.get("source", "")),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ZScoreCodec":
        payload = json.loads(Path(path).read_text())
        if "action_mean" in payload and "action" not in payload:
            return cls.from_metadata(payload)
        return cls.from_statistics(payload)

    def __post_init__(self) -> None:
        for kind in ("action", "state"):
            mean = np.asarray(getattr(self, f"{kind}_mean"))
            std = np.asarray(getattr(self, f"{kind}_std"))
            if mean.shape != std.shape or mean.ndim != 1:
                raise RuntimeError(f"{kind} mean/std shape mismatch")
            if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
                raise RuntimeError(f"invalid {kind} mean/std")
            grippers = self.gripper_indices if kind == "action" else self.state_gripper_indices
            ranges = getattr(self, f"{kind}_gripper_range")
            if len(ranges) != len(grippers) or any(hi <= lo for lo, hi in ranges):
                raise RuntimeError(f"invalid {kind} gripper ranges")
        if self.gripper_mode not in ("affine", "binary_pm1"):
            raise RuntimeError(f"invalid gripper_mode: {self.gripper_mode}")

    @property
    def action_dim(self) -> int:
        return len(self.action_mean)

    @property
    def state_dim(self) -> int:
        return len(self.state_mean)

    @property
    def effective_state_clamp_sigma(self) -> float:
        if self.state_clamp_sigma is not None:
            return float(self.state_clamp_sigma)
        if self.state_featurization == "xyz_rot6d_fingers2_11d":
            return DEFAULT_LIBERO_STATE_CLAMP_SIGMA
        return float(self.encode_clamp_sigma)

    def _arrays(self, kind: str):
        mean = np.asarray(getattr(self, f"{kind}_mean"), np.float64)
        std = np.asarray(getattr(self, f"{kind}_std"), np.float64)
        grippers = list(self.gripper_indices if kind == "action" else self.state_gripper_indices)
        ranges = np.asarray(getattr(self, f"{kind}_gripper_range"), np.float64).reshape(-1, 2)
        joints = [i for i in range(len(mean)) if i not in grippers]
        return mean, std, joints, grippers, ranges

    def _encode(self, values: np.ndarray, kind: str) -> np.ndarray:
        if kind == "state" and self.state_featurization == "xyz_rot6d_fingers2_11d":
            arr = np.asarray(values, dtype=np.float32)
            if arr.shape[-1] == 8:
                values = featurize_libero_state_11d(arr).astype(np.float32)
        mean, std, joints, grippers, ranges = self._arrays(kind)
        raw = np.asarray(values, dtype=np.float32)
        if raw.shape[-1] != len(mean) or not np.isfinite(raw).all():
            raise ValueError(f"{kind} must be finite [..., {len(mean)}], got {raw.shape}")
        clamp = self.effective_state_clamp_sigma if kind == "state" else float(self.encode_clamp_sigma)
        z = (raw.astype(np.float64) - mean) / std
        encoded = z.astype(np.float32)
        encoded[..., joints] = np.clip(z[..., joints], -clamp, clamp).astype(np.float32)
        if grippers:
            if kind == "action" and self.gripper_mode == "binary_pm1":
                encoded[..., grippers] = np.where(
                    raw[..., grippers] > 0.5, 1.0, -1.0
                ).astype(np.float32)
            else:
                lo, hi = ranges[:, 0], ranges[:, 1]
                unit = (raw[..., grippers].astype(np.float64) - lo) / (hi - lo)
                encoded[..., grippers] = np.clip(2.0 * unit - 1.0, -1.0, 1.0).astype(np.float32)
        return encoded

    def _decode(self, values: np.ndarray, kind: str) -> np.ndarray:
        mean, std, _, grippers, ranges = self._arrays(kind)
        normalized = np.asarray(values, dtype=np.float32)
        if normalized.shape[-1] != len(mean) or not np.isfinite(normalized).all():
            raise ValueError(f"normalized {kind} must be finite [..., {len(mean)}]")
        decoded = (normalized.astype(np.float64) * std + mean).astype(np.float32)
        if grippers:
            if kind == "action" and self.gripper_mode == "binary_pm1":
                decoded[..., grippers] = np.where(
                    normalized[..., grippers] >= 0.0, 1.0, -1.0
                ).astype(np.float32)
            else:
                lo, hi = ranges[:, 0], ranges[:, 1]
                decoded[..., grippers] = (
                    (normalized[..., grippers] + 1.0) * 0.5 * (hi - lo) + lo
                ).astype(np.float32)
        return decoded

    def encode_action(self, values: np.ndarray) -> np.ndarray:
        return self._encode(values, "action")

    def encode_state(self, values: np.ndarray) -> np.ndarray:
        return self._encode(values, "state")

    def decode_action(self, values: np.ndarray) -> np.ndarray:
        return self._decode(values, "action")

    def decode_state(self, values: np.ndarray) -> np.ndarray:
        return self._decode(values, "state")

    @property
    def metadata(self) -> dict[str, Any]:
        base = {
            "version": CODEC_VERSION,
            "joint_normalization": "zscore_mean_std_all_frames_population_std",
            "gripper_normalization": (
                "binary_pm1_threshold_0.5_decode_0.0"
                if self.gripper_mode == "binary_pm1"
                else "affine_[min,max]->[-1,1]_clip_on_encode"
            ),
            "gripper_mode": self.gripper_mode,
            "action_mean": list(self.action_mean),
            "action_std": list(self.action_std),
            "state_mean": list(self.state_mean),
            "state_std": list(self.state_std),
            "action_gripper_range": [list(r) for r in self.action_gripper_range],
            "state_gripper_range": [list(r) for r in self.state_gripper_range],
            "gripper_indices": list(self.gripper_indices),
            "state_gripper_indices": list(self.state_gripper_indices),
            "encode_clamp_sigma": self.encode_clamp_sigma,
            "state_clamp_sigma": self.effective_state_clamp_sigma,
            "state_featurization": self.state_featurization,
            "decode_clip": False,
            "source": self.source,
            "padding": "end-of-episode windows repeat the last frame; supervised",
        }
        return {**base, "contract_sha256": _sha(base)}


def describe(
    values: np.ndarray,
    grippers: list[int],
    *,
    fixed_gripper_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute float64 population mean, std, min, max, and gripper ranges over all frames."""
    x = np.asarray(values, np.float64)
    std = x.std(0)
    floored = [int(i) for i in np.where(std < MIN_STD)[0] if i not in grippers]
    std = np.where(std < MIN_STD, 1.0, std)
    lo, hi = x.min(0), x.max(0)
    ranges = []
    for i in grippers:
        if fixed_gripper_range is not None:
            ranges.append([float(fixed_gripper_range[0]), float(fixed_gripper_range[1])])
        else:
            a, b = float(lo[i]), float(hi[i])
            ranges.append([a, b] if b > a else [a, a + 1.0])
    return {
        "mean": x.mean(0).tolist(),
        "std": std.tolist(),
        "minimum": lo.tolist(),
        "maximum": hi.tolist(),
        "gripper_range": ranges,
        "constant_dims_std_set_to_1": floored,
        "frames": int(x.shape[0]),
    }
