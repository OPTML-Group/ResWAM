"""Frozen DINOv3-ViT-B/16 encoder and spatial/timestep positional embeddings."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

DINO_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DINO_DIM = 768
GRID_SIZE = 8
TOKENS_PER_VIEW = GRID_SIZE * GRID_SIZE


def sinusoidal_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    """1-D sinusoidal embedding of shape [B, dim] from timesteps [B]."""
    if timestep.ndim != 1 or dim % 2:
        raise ValueError("timestep must be [B] and dim must be even")
    half = dim // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angle = timestep.float()[:, None] * frequency[None]
    return torch.cat((angle.sin(), angle.cos()), dim=-1)


def spatial_sincos_2d(grid_size: int, dim: int) -> torch.Tensor:
    """2-D spatial sinusoidal positional embedding of shape [grid_size^2, dim]."""
    if dim % 4:
        raise ValueError("2D positional dimension must be divisible by four")
    coordinate = torch.linspace(0.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.cat(
        (
            sinusoidal_embedding(xx.reshape(-1), dim // 2),
            sinusoidal_embedding(yy.reshape(-1), dim // 2),
        ),
        dim=-1,
    )


def preprocess_camera(image: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    """Deterministic HxW -> 256x256 direct bicubic antialias resize in [0, 1]."""
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) <= 0:
        raise ValueError("camera frame must be RGB CHW")
    pixels = image.float().unsqueeze(0)
    if image.dtype == torch.uint8:
        pixels = pixels / 255.0
    elif bool((pixels.detach().amin() < 0) or (pixels.detach().amax() > 1)):
        raise ValueError("floating camera frame must be in [0, 1]")
    return (
        F.interpolate(
            pixels,
            (image_size, image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .squeeze(0)
        .clamp(0.0, 1.0)
    )


class DINOv3DualViewEncoder(nn.Module):
    """Frozen DINOv3-ViT-B/16 with arithmetic mean of blocks (8, 10) and 2x2 spatial pooling.

    Maps RGB images [B, 3, H, W] to [B, 64, 768] spatial tokens.
    """

    def __init__(
        self,
        model_id: str = DINO_REPO,
        *,
        selected_blocks: Sequence[int] = (8, 10),
        image_size: int = 256,
        revision: str | None = DINO_REVISION,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if model is None:
            from pathlib import Path
            from transformers import AutoModel

            is_local = Path(str(model_id)).exists()
            model = AutoModel.from_pretrained(
                str(model_id),
                revision=None if is_local else revision,
                trust_remote_code=True,
                dtype=torch.float32,
            )
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.image_size = int(image_size)
        self.selected_blocks = tuple(int(index) for index in selected_blocks)
        config = self.model.config
        self.num_blocks = int(config.num_hidden_layers)
        self.hidden_size = int(config.hidden_size)
        self.patch_size = int(config.patch_size)
        self.register_tokens = int(getattr(config, "num_register_tokens", 0))
        if self.hidden_size != DINO_DIM:
            raise RuntimeError(f"ResWAM requires DINO width {DINO_DIM}, got {self.hidden_size}")
        if self.patch_size != 16 or self.image_size // self.patch_size != 16:
            raise RuntimeError("ResWAM requires a 16x16 DINO patch grid at 256x256 resolution")
        if not self.selected_blocks or any(
            index < 0 or index >= self.num_blocks for index in self.selected_blocks
        ):
            raise RuntimeError(
                f"invalid DINO block selection {self.selected_blocks} for {self.num_blocks} blocks"
            )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINO images must be [B, 3, H, W]")
        pixels = images.float()
        if images.dtype == torch.uint8:
            pixels = pixels / 255.0
        elif bool((pixels.detach().amin() < 0) or (pixels.detach().amax() > 1)):
            raise ValueError("floating DINO images must be in [0, 1]")
        pixels = F.interpolate(
            pixels,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
        return (pixels - self.imagenet_mean) / self.imagenet_std

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = self.preprocess(images)
        output = self.model(pixel_values=pixels, output_hidden_states=True)
        hidden_states = output.hidden_states
        if len(hidden_states) != self.num_blocks + 1:
            raise RuntimeError("DINO hidden-state count does not match its config")
        selected = torch.stack(
            [hidden_states[index + 1] for index in self.selected_blocks], dim=0
        ).mean(dim=0)
        patch_start = 1 + self.register_tokens
        patches = selected[:, patch_start:]
        expected = (self.image_size // self.patch_size) ** 2
        if patches.shape[1:] != (expected, self.hidden_size):
            raise RuntimeError(f"unexpected DINO patch shape {tuple(patches.shape)}")
        patches = patches.reshape(-1, 16, 16, self.hidden_size)
        patches = patches.reshape(
            -1, GRID_SIZE, 2, GRID_SIZE, 2, self.hidden_size
        ).mean(dim=(2, 4))
        return patches.reshape(-1, TOKENS_PER_VIEW, self.hidden_size).detach()
