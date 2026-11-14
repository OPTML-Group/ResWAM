"""Conditional Flow Matching objective and 10-step Euler / Jacobi sampler for ResWAM."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .dinov3_encoder import DINO_DIM, TOKENS_PER_VIEW
from .expert import Prediction, ResWAMExpert


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    prediction: Prediction


def masked_action_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Dimension-generic masked action MSE in FP32."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("action tensors must be [B, H, D]")
    if mask.shape not in {prediction.shape, prediction.shape[:-1]}:
        raise ValueError("mask must be [B, H] or [B, H, D]")
    expanded = mask.unsqueeze(-1) if mask.ndim == prediction.ndim - 1 else mask
    expanded = expanded.expand_as(prediction).to(prediction.dtype)
    if expanded.sum() <= 0:
        raise ValueError("empty action mask")
    return ((prediction.float() - target.float()).square() * expanded).sum() / expanded.sum()


def shifted_schedule(shift: float, steps: int = 10, *, device: torch.device | None = None) -> torch.Tensor:
    """Monotonic schedule on [0, 1] with ``steps + 1`` grid points."""
    base = torch.linspace(0.0, 1.0, steps + 1, device=device)
    shift = float(shift)
    if shift == 1.0:
        return base
    return shift * base / (1.0 + (shift - 1.0) * base)


def apply_tau_shift(u: torch.Tensor, shift: float) -> torch.Tensor:
    shift = float(shift)
    if shift == 1.0:
        return u
    return shift * u / (1.0 + (shift - 1.0) * u)


class ResWAMFlowObjective(nn.Module):
    """Joint Action and Latent World Residual Conditional Flow Matching objective.

    Computes the normalized DINOv3 residual target:
        Delta_t = LayerNorm(E(o_{t+H})) - LayerNorm(E(o_t))
    and optimizes:
        L = L_action + lambda_w * L_world
    """

    def __init__(
        self,
        expert: ResWAMExpert,
        world_loss_weight: float = 0.1,
        *,
        action_tau_shift: float = 1.0,
        world_tau_shift: float = 5.0,
        solver_steps: int = 10,
    ) -> None:
        super().__init__()
        self.expert = expert
        self.world_loss_weight = float(world_loss_weight)
        self.action_tau_shift = float(action_tau_shift)
        self.world_tau_shift = float(world_tau_shift)
        self.solver_steps = int(solver_steps)

    @staticmethod
    def residual(current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Per-token LayerNorm residual Delta_t = LN(E(o_{t+H})) - LN(E(o_t)) in FP32."""
        return F.layer_norm(future.float(), (DINO_DIM,)) - F.layer_norm(
            current.float(), (DINO_DIM,)
        )

    def forward(
        self,
        *,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        primary_target: torch.Tensor,
        current_primary: torch.Tensor,
        vlm_context: torch.Tensor,
        vlm_mask: torch.Tensor | None,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> ObjectiveOutput:
        batch, device = actions.shape[0], actions.device
        base_action = torch.rand(batch, device=device, generator=generator)
        base_world = torch.rand(batch, device=device, generator=generator)
        tau_action = apply_tau_shift(base_action, self.action_tau_shift)
        tau_world = apply_tau_shift(base_world, self.world_tau_shift)

        action_noise = torch.randn(
            actions.shape, device=device, dtype=actions.dtype, generator=generator
        )
        world_noise = torch.randn(
            primary_target.shape,
            device=device,
            dtype=primary_target.dtype,
            generator=generator,
        )
        prediction = self.expert(
            noisy_action=action_noise.lerp(actions, tau_action[:, None, None].to(actions.dtype)),
            noisy_primary=world_noise.lerp(
                primary_target, tau_world[:, None, None].to(primary_target.dtype)
            ),
            current_primary=current_primary,
            tau_action=tau_action,
            tau_world=tau_world,
            vlm_context=vlm_context,
            vlm_mask=vlm_mask,
            state=state,
            state_mask=state_mask,
        )
        action_raw = masked_action_mse(prediction.action, actions - action_noise, action_mask)
        world_raw = F.mse_loss(prediction.primary.float(), (primary_target - world_noise).float())
        world_weighted = self.world_loss_weight * world_raw
        total = action_raw + world_weighted
        metrics = {
            "loss/total_weighted": total.detach(),
            "loss/action_raw": action_raw.detach(),
            "loss/action_weighted": action_raw.detach(),
            "loss/primary_raw": world_raw.detach(),
            "loss/primary_weighted": world_weighted.detach(),
            "tau/action_mean": tau_action.mean().detach(),
            "tau/world_mean": tau_world.mean().detach(),
        }
        num_world_views = int(getattr(self.expert, "num_world_views", 1))
        if num_world_views > 1:
            pred_views = prediction.primary.float().chunk(num_world_views, dim=1)
            vel_views = (primary_target - world_noise).float().chunk(num_world_views, dim=1)
            for idx, (pv, vv) in enumerate(zip(pred_views, vel_views)):
                metrics[f"loss/world_view_{idx}_raw"] = F.mse_loss(pv, vv).detach()
        return ObjectiveOutput(total, metrics, prediction)


@torch.inference_mode()
def synchronous_sample(
    expert: ResWAMExpert,
    *,
    current_primary: torch.Tensor,
    vlm_context: torch.Tensor,
    vlm_mask: torch.Tensor | None,
    state: torch.Tensor,
    state_mask: torch.Tensor,
    seeds: Sequence[int],
    action_shift: float = 1.0,
    world_shift: float = 5.0,
    steps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """10-step synchronous Euler / Jacobi flow ODE integration for action and world residual."""
    batch, device = state.shape[0], state.device
    if len(seeds) != batch:
        raise ValueError("one seed per rollout required")
    world_tokens = int(getattr(expert, "world_tokens", TOKENS_PER_VIEW))
    noise_a, noise_w = [], []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed) % (2**63 - 1))
        noise_a.append(torch.randn(expert.horizon, expert.action_dim, device=device, generator=generator))
        noise_w.append(torch.randn(world_tokens, DINO_DIM, device=device, generator=generator))
    action, world = torch.stack(noise_a), torch.stack(noise_w)
    parameter_dtype = next(expert.parameters()).dtype
    model_dtype = torch.bfloat16 if device.type == "cuda" else parameter_dtype
    sa = shifted_schedule(action_shift, steps=steps, device=device)
    sw = shifted_schedule(world_shift, steps=steps, device=device)
    nfe = 0
    for index in range(steps):
        compute = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with compute:
            prediction = expert(
                noisy_action=action.to(model_dtype),
                noisy_primary=world.to(model_dtype),
                current_primary=current_primary.to(model_dtype),
                tau_action=sa[index].expand(batch).to(model_dtype),
                tau_world=sw[index].expand(batch).to(model_dtype),
                vlm_context=vlm_context.to(model_dtype),
                vlm_mask=vlm_mask,
                state=state.to(model_dtype),
                state_mask=state_mask,
            )
        nfe += 1
        action = action + (sa[index + 1] - sa[index]) * prediction.action.float()
        world = world + (sw[index + 1] - sw[index]) * prediction.primary.float()
    return action.float(), world.float(), nfe
