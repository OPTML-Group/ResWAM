"""Shared temporal image augmentation (pi0.5 / openpi protocol) for ResWAM.

Applies identical random crop, rotation, and color jitter parameters to the current frame ``o_t``
and future endpoint frame ``o_{t+H}`` on the primary camera view, and color jitter on wrist views:
  - Primary (``head``):
      1. RandomCrop(round(0.95 * H), round(0.95 * W)) -> 243x243 from 256x256
      2. Resize(H, W) with bilinear antialiasing
      3. RandomRotation((-5.0, +5.0)) degrees around image center (fill=0)
      4. ColorJitter(brightness=[0.7, 1.3], contrast=[0.8, 1.2], saturation=[0.5, 1.5])
    Sampled once per window from a deterministic per-(base_seed, epoch, current_row, endpoint_row)
    CPU generator and applied jointly to ``[o_t, o_{t+H}]`` so that:
      - VLM ``o_t`` and DINOv3 ``o_t`` receive the exact same augmented uint8 image byte-for-byte;
      - DINOv3 ``o_t`` and ``o_{t+H}`` share identical spatial and photometric transforms.
  - Wrist (``left_wrist``, ``right_wrist``):
      ColorJitter(brightness=[0.7, 1.3], contrast=[0.8, 1.2], saturation=[0.5, 1.5]) only
      (no crop and no rotation).
  - Evaluation (``eval`` mode):
      Disabled (identity).
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any

import torch
from torchvision.transforms import functional as TF

AUGMENTATION_VERSION = "reswam-pi05-shared-temporal-aug-v1"
PRIMARY_CROP_SCALE = 0.95
PRIMARY_ROTATE_DEGREES = (-5.0, 5.0)
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.8, 1.2)
SATURATION_RANGE = (0.5, 1.5)

_PRIMARY_STREAM_SALT = 0x5049_3035_5052_494D
_WRIST_STREAM_SALT = 0x5049_3035_5752_5354


def augmentation_spec() -> dict[str, Any]:
    return {
        "version": AUGMENTATION_VERSION,
        "reference": "openpi/src/openpi/models/model.py::preprocess_observation",
        "primary": {
            "temporal_sharing": "identical parameters applied to (o_t, o_{t+H})",
            "vlm_dino_alignment": "exact same augmented uint8 o_t fed to both VLM processor and online DINOv3",
            "random_crop_scale": PRIMARY_CROP_SCALE,
            "crop_size_for_256": [
                int(round(256 * PRIMARY_CROP_SCALE)),
                int(round(256 * PRIMARY_CROP_SCALE)),
            ],
            "resize_to": [256, 256],
            "interpolation": "bilinear",
            "random_rotation_degrees": list(PRIMARY_ROTATE_DEGREES),
            "rotation_fill": 0.0,
            "color_jitter": {
                "brightness": list(BRIGHTNESS_RANGE),
                "contrast": list(CONTRAST_RANGE),
                "saturation": list(SATURATION_RANGE),
                "order": ["brightness", "contrast", "saturation"],
            },
        },
        "wrist": {
            "random_crop": False,
            "random_rotation": False,
            "color_jitter": {
                "brightness": list(BRIGHTNESS_RANGE),
                "contrast": list(CONTRAST_RANGE),
                "saturation": list(SATURATION_RANGE),
                "order": ["brightness", "contrast", "saturation"],
            },
        },
        "eval_mode": "disabled (identity)",
        "rng": "deterministic per-(base_seed, epoch, frame_row, future_row) local CPU torch.Generator",
    }


def derive_sample_seed(
    *,
    base_seed: int,
    epoch: int,
    frame_row: int,
    future_row: int,
    stream_salt: int,
) -> int:
    payload = struct.pack(
        "<QQQQQ",
        int(base_seed) & 0xFFFFFFFFFFFFFFFF,
        int(epoch) & 0xFFFFFFFFFFFFFFFF,
        int(frame_row) & 0xFFFFFFFFFFFFFFFF,
        int(future_row) & 0xFFFFFFFFFFFFFFFF,
        int(stream_salt) & 0xFFFFFFFFFFFFFFFF,
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(
        torch.empty((), dtype=torch.float32)
        .uniform_(float(low), float(high), generator=generator)
        .item()
    )


def _validate_frames_uint8(frames_uint8: torch.Tensor, name: str) -> tuple[int, int]:
    if not isinstance(frames_uint8, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(frames_uint8)}")
    if frames_uint8.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8, got {frames_uint8.dtype}")
    if frames_uint8.ndim != 4 or frames_uint8.shape[1] != 3 or frames_uint8.shape[0] < 1:
        raise ValueError(f"{name} must have shape [N, 3, H, W], got {tuple(frames_uint8.shape)}")
    height, width = int(frames_uint8.shape[2]), int(frames_uint8.shape[3])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} has invalid spatial dimensions {height}x{width}")
    return height, width


def augment_primary_pair_uint8(
    frames_uint8: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Apply shared pi0.5 primary augmentation (crop 0.95 + resize + rotate [-5, 5] + color jitter)
    identically across all frames in ``frames_uint8`` (e.g. ``[o_t, o_{t+H}]``)."""
    height, width = _validate_frames_uint8(frames_uint8, "primary_frames")
    crop_h = max(1, int(round(height * PRIMARY_CROP_SCALE)))
    crop_w = max(1, int(round(width * PRIMARY_CROP_SCALE)))
    max_top = max(0, height - crop_h)
    max_left = max(0, width - crop_w)
    top = int(torch.randint(0, max_top + 1, (), generator=generator).item())
    left = int(torch.randint(0, max_left + 1, (), generator=generator).item())
    angle = _uniform(generator, PRIMARY_ROTATE_DEGREES[0], PRIMARY_ROTATE_DEGREES[1])
    brightness = _uniform(generator, BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
    contrast = _uniform(generator, CONTRAST_RANGE[0], CONTRAST_RANGE[1])
    saturation = _uniform(generator, SATURATION_RANGE[0], SATURATION_RANGE[1])

    x = frames_uint8.to(dtype=torch.float32) / 255.0
    x = TF.resized_crop(
        x,
        top=top,
        left=left,
        height=crop_h,
        width=crop_w,
        size=[height, width],
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True,
    )
    x = TF.rotate(
        x,
        angle=angle,
        interpolation=TF.InterpolationMode.BILINEAR,
        expand=False,
        fill=0.0,
    )
    x = TF.adjust_brightness(x, brightness)
    x = TF.adjust_contrast(x, contrast)
    x = TF.adjust_saturation(x, saturation)
    out = (x.clamp(0.0, 1.0) * 255.0).round().to(dtype=torch.uint8).contiguous()
    params: dict[str, float | int] = {
        "crop_top": top,
        "crop_left": left,
        "crop_height": crop_h,
        "crop_width": crop_w,
        "angle_degrees": angle,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
    }
    return out, params


def augment_wrist_pair_uint8(
    frames_uint8: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply shared pi0.5 wrist augmentation (color jitter only; no crop or rotation)."""
    _validate_frames_uint8(frames_uint8, "wrist_frames")
    brightness = _uniform(generator, BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
    contrast = _uniform(generator, CONTRAST_RANGE[0], CONTRAST_RANGE[1])
    saturation = _uniform(generator, SATURATION_RANGE[0], SATURATION_RANGE[1])

    x = frames_uint8.to(dtype=torch.float32) / 255.0
    x = TF.adjust_brightness(x, brightness)
    x = TF.adjust_contrast(x, contrast)
    x = TF.adjust_saturation(x, saturation)
    out = (x.clamp(0.0, 1.0) * 255.0).round().to(dtype=torch.uint8).contiguous()
    params: dict[str, float] = {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
    }
    return out, params


def augment_window_cameras(
    current: dict[str, torch.Tensor],
    endpoint: dict[str, torch.Tensor],
    *,
    base_seed: int,
    epoch: int,
    current_row: int,
    endpoint_row: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply shared pi0.5 augmentation to a window sample's ``current`` and ``endpoint`` cameras."""
    aug_cur: dict[str, torch.Tensor] = {}
    aug_end: dict[str, torch.Tensor] = {}
    for idx, (view, cur_frame) in enumerate(current.items()):
        is_wrist = "wrist" in view
        salt = (_WRIST_STREAM_SALT + idx) if is_wrist else (_PRIMARY_STREAM_SALT + idx)
        seed = derive_sample_seed(
            base_seed=base_seed,
            epoch=epoch,
            frame_row=current_row,
            future_row=endpoint_row,
            stream_salt=salt,
        )
        gen = torch.Generator().manual_seed(seed)
        if view in endpoint:
            pair = torch.stack([cur_frame, endpoint[view]], dim=0)
            aug_pair, _ = (
                augment_wrist_pair_uint8(pair, gen)
                if is_wrist
                else augment_primary_pair_uint8(pair, gen)
            )
            aug_cur[view] = aug_pair[0]
            aug_end[view] = aug_pair[1]
        else:
            single = cur_frame.unsqueeze(0)
            aug_single, _ = (
                augment_wrist_pair_uint8(single, gen)
                if is_wrist
                else augment_primary_pair_uint8(single, gen)
            )
            aug_cur[view] = aug_single[0]
    return aug_cur, aug_end


__all__ = [
    "AUGMENTATION_VERSION",
    "BRIGHTNESS_RANGE",
    "CONTRAST_RANGE",
    "PRIMARY_CROP_SCALE",
    "PRIMARY_ROTATE_DEGREES",
    "SATURATION_RANGE",
    "augment_primary_pair_uint8",
    "augment_window_cameras",
    "augment_wrist_pair_uint8",
    "augmentation_spec",
    "derive_sample_seed",
]
