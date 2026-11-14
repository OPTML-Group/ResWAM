"""Dual-stream Joint-Attention DiT Expert for ResWAM (Action & World Residual branches)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dinov3_encoder import DINO_DIM, TOKENS_PER_VIEW, sinusoidal_embedding, spatial_sincos_2d
from .dit_modules import (
    AttentionVariant,
    FinalLayer,
    QKNormCrossContext,
    StableJointBlock,
    stable_time_mlp,
)

JOINT_DEPTH = 12


@dataclass(frozen=True)
class Prediction:
    action: torch.Tensor
    primary: torch.Tensor


class ResWAMExpert(nn.Module):
    """12-layer bidirectional joint-attention DiT over Action and World Residual tokens.

    Parameterized by ``action_dim``, ``state_dim``, ``horizon``, and ``num_world_views``.
    Supports both standard masked state cross-attention and learned ``state_null`` replacement
    for proprioceptive state dropout (used in LIBERO training).
    """

    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        horizon: int,
        num_world_views: int = 1,
        dim: int = 1024,
        heads: int = 16,
        ffn_dim: int = 4096,
        depth: int = JOINT_DEPTH,
        vlm_dim: int = 2048,
        variant: AttentionVariant = "bidirectional",
        use_state_null_tokens: bool = False,
        layout: str = "robotwin",
    ) -> None:
        """``layout`` selects the parameterization of the released checkpoints.

        - ``"robotwin"`` (RoboTwin 2.0 / RoboDojo, ``StableRoboTwinExpert``): state modules are
          built in the middle of the module order; a ``view_identity`` embedding exists only
          when ``num_world_views > 1``.
        - ``"libero"`` (LIBERO, ``StableLiberoText16ProprioExpert``): every action/world module
          is built first, in the stable-v2 order, with a learned ``view_identity`` [1, 1, dim]
          added to the world position embedding even for a single world view; the state
          modules (``state_input``, ``state_position``, ``state_null``, ``action_state``,
          ``world_state``) are built afterwards.
        """
        super().__init__()
        if depth != JOINT_DEPTH:
            raise ValueError(f"ResWAM expert depth is locked to {JOINT_DEPTH}")
        if int(num_world_views) < 1:
            raise ValueError("num_world_views must be >= 1")
        if layout not in ("robotwin", "libero"):
            raise ValueError(f"unknown expert layout {layout!r}")
        self.layout = str(layout)
        self.dim, self.depth = int(dim), int(depth)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.num_world_views = int(num_world_views)
        self.world_tokens = self.num_world_views * TOKENS_PER_VIEW
        self.use_state_null_tokens = bool(use_state_null_tokens)
        self.has_view_identity = self.layout == "libero" or self.num_world_views > 1

        self.action_input = nn.Linear(self.action_dim, dim)
        self.residual_input = nn.Linear(DINO_DIM, dim)
        self.current_input = nn.Linear(DINO_DIM, dim)
        self.action_position = nn.Parameter(torch.randn(1, self.horizon, dim) * 0.02)
        self.register_buffer("primary_position", spatial_sincos_2d(8, dim).unsqueeze(0))
        if self.layout == "libero":
            self.view_identity = nn.Parameter(torch.randn(self.num_world_views, 1, dim) * 0.02)
        else:
            self._build_state_modules(dim, heads, cross=False)
        self.action_time = stable_time_mlp(dim)
        self.world_time = stable_time_mlp(dim)
        self.action_vlm = QKNormCrossContext(dim, heads, vlm_dim)
        self.world_vlm = QKNormCrossContext(dim, heads, vlm_dim)
        if self.layout == "robotwin":
            self.action_state = QKNormCrossContext(dim, heads, dim)
            self.world_state = QKNormCrossContext(dim, heads, dim)
        self.action_static = QKNormCrossContext(dim, heads, dim)
        self.world_static = QKNormCrossContext(dim, heads, dim)
        self.blocks = nn.ModuleList(
            StableJointBlock(dim, heads, ffn_dim, variant=variant) for _ in range(depth)
        )
        self.action_output = FinalLayer(dim, self.action_dim)
        self.world_output = FinalLayer(dim, DINO_DIM)
        if self.layout == "libero":
            self._build_state_modules(dim, heads, cross=True)
        elif self.has_view_identity:
            self.view_identity = nn.Parameter(torch.randn(self.num_world_views, 1, dim) * 0.02)

    def _build_state_modules(self, dim: int, heads: int, *, cross: bool) -> None:
        self.state_input = nn.Linear(1, dim)
        self.state_position = nn.Parameter(torch.randn(1, self.state_dim, dim) * 0.02)
        if self.use_state_null_tokens:
            self.state_null = nn.Parameter(torch.randn(1, self.state_dim, dim) * 0.02)
        if cross:
            self.action_state = QKNormCrossContext(dim, heads, dim)
            self.world_state = QKNormCrossContext(dim, heads, dim)

    def _world_position(self) -> torch.Tensor:
        if not self.has_view_identity:
            return self.primary_position
        return (self.primary_position + self.view_identity).reshape(1, self.world_tokens, self.dim)

    def _build_state_context(
        self, state: torch.Tensor, state_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = self.state_input(state.unsqueeze(-1)) + self.state_position
        if self.use_state_null_tokens and hasattr(self, "state_null"):
            null = self.state_null.to(tokens.dtype).expand_as(tokens)
            return torch.where(state_mask.bool().unsqueeze(-1), tokens, null), None
        return tokens, state_mask

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        noisy_primary: torch.Tensor,
        current_primary: torch.Tensor,
        tau_action: torch.Tensor,
        tau_world: torch.Tensor,
        vlm_context: torch.Tensor,
        vlm_mask: torch.Tensor | None,
        state: torch.Tensor,
        state_mask: torch.Tensor,
    ) -> Prediction:
        batch = noisy_action.shape[0]
        if noisy_action.shape != (batch, self.horizon, self.action_dim):
            raise ValueError(f"action must be [B, {self.horizon}, {self.action_dim}]")
        if noisy_primary.shape != (batch, self.world_tokens, DINO_DIM):
            raise ValueError(f"primary world must be [B, {self.world_tokens}, {DINO_DIM}]")
        if current_primary.shape != noisy_primary.shape:
            raise ValueError("current/target primary shape mismatch")
        if state.shape != (batch, self.state_dim) or state_mask.shape != state.shape:
            raise ValueError(f"state contract violated: expected [B, {self.state_dim}], got {tuple(state.shape)}")

        action_t = self.action_time(sinusoidal_embedding(tau_action, self.dim).to(noisy_action.dtype))
        world_t = self.world_time(sinusoidal_embedding(tau_world, self.dim).to(noisy_action.dtype))
        world_pos = self._world_position()

        action = self.action_input(noisy_action) + self.action_position
        primary = self.residual_input(noisy_primary) + world_pos
        static = self.current_input(current_primary) + world_pos
        state_context, effective_state_mask = self._build_state_context(state, state_mask)

        action = self.action_vlm(action, vlm_context, vlm_mask)
        primary = self.world_vlm(primary, vlm_context, vlm_mask)
        action = self.action_state(action, state_context, effective_state_mask)
        primary = self.world_state(primary, state_context, effective_state_mask)
        action = self.action_static(action, static, None)
        primary = self.world_static(primary, static, None)

        empty = primary[:, :0]
        for block in self.blocks:
            action, primary, empty = block(action, primary, empty, action_t, world_t)
        if empty.shape[1] != 0:
            raise RuntimeError("unexpected extra world tokens materialized")
        return Prediction(self.action_output(action, action_t), self.world_output(primary, world_t))


DIM_DEPENDENT_LAYERS = {
    "action_dim": (
        "action_input.weight",
        "action_input.bias",
        "action_output.linear.weight",
        "action_output.linear.bias",
    ),
    "horizon": ("action_position",),
    "state_dim": ("state_position", "state_null"),
    "num_world_views": ("view_identity",),
}
