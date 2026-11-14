"""Top-level ResWAMSystem combining RynnBrain-1.1-2B, frozen DINOv3-B/16, and Dual-Stream DiT."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .dinov3_encoder import DINO_REPO, DINO_REVISION, DINOv3DualViewEncoder, preprocess_camera
from .expert import ResWAMExpert
from .objective import ObjectiveOutput, ResWAMFlowObjective, synchronous_sample
from .rynnbrain_encoder import RYNN_REPO, RYNN_REVISION, RynnBrainText16Encoder

VIEW_ORDER = ("head", "left_wrist", "right_wrist")
CONTRACT_VIEW = {"base": "head", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
DROPOUT_SEED_SALT = 0x5354_4154_45


def resolve_world_views(
    world_views: Sequence[str] | str | None,
    active_views: Sequence[str],
) -> tuple[str, ...]:
    """Map contract or canonical view names to canonical ``VIEW_ORDER`` subset."""
    if world_views is None:
        raw = ["head"]
    elif isinstance(world_views, str):
        raw = [world_views]
    else:
        raw = list(world_views)
    if not raw:
        raise ValueError("world_views must not be empty")
    mapped = [CONTRACT_VIEW.get(str(v), str(v)) for v in raw]
    invalid = [v for v in mapped if v not in VIEW_ORDER]
    if invalid:
        raise ValueError(f"invalid world_views {invalid}; allowed: {VIEW_ORDER} or {tuple(CONTRACT_VIEW)}")
    missing = [v for v in mapped if v not in active_views]
    if missing:
        raise ValueError(f"world_views {missing} not present in active views {tuple(active_views)}")
    return tuple(v for v in VIEW_ORDER if v in mapped)


def unview_float32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor
    if tensor.dtype != torch.int32:
        raise RuntimeError("bit-exact float payload must arrive as int32 or float32")
    return tensor.view(torch.float32)


def draw_state_keep_mask(
    seed: int,
    batch: int,
    state_dim: int,
    dropout_prob: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if dropout_prob <= 0.0:
        return torch.ones((batch, state_dim), dtype=torch.bool, device=device)
    generator = torch.Generator().manual_seed((int(seed) * 1_000_003 + DROPOUT_SEED_SALT) % (2**63 - 1))
    keep = torch.rand(batch, generator=generator) >= float(dropout_prob)
    return keep[:, None].expand(batch, state_dim).to(device=device)


class ResWAMSystem(nn.Module):
    """Complete ResWAM policy and latent residual world model system.

    Supports both:
      1. Online mode: computes frozen DINOv3 features on-the-fly for ``o_t`` and ``o_{t+H}``
         and extracts the normalized residual ``Delta_t = LN(E(o_{t+H})) - LN(E(o_t))``.
      2. Cached mode: reads pre-extracted DINOv3 features from ``batch['dino_current']``
         and ``batch['dino_endpoint']`` for high-throughput training.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vlm: nn.Module | None = None,
        dino: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        self.views = tuple(config["views"])
        if not self.views or self.views[0] != "head" or any(v not in VIEW_ORDER for v in self.views):
            raise ValueError(f"views must start with 'head' and be a subset of {VIEW_ORDER}")
        self.world_views = resolve_world_views(
            config.get("world_views") or config.get("future_views") or config.get("world_view"),
            self.views,
        )
        self.action_dim = int(config["action_dim"])
        self.state_dim = int(config["state_dim"])
        self.horizon = int(config["horizon"])
        self.state_dropout_prob = float(config.get("state_dropout_prob", 0.0))

        rynn_cfg = model_cfg["rynnbrain"]
        if vlm is not None:
            self.vlm = vlm
        else:
            rynn_path = rynn_cfg.get("path") or rynn_cfg.get("model_id") or RYNN_REPO
            self.vlm = RynnBrainText16Encoder(
                rynn_path,
                revision=rynn_cfg.get("revision", RYNN_REVISION),
                dtype=torch.bfloat16,
                activation_checkpointing=False,
                views=self.views,
                prompt_style=rynn_cfg.get("prompt_style", "multi_camera"),
            )

        dino_cfg = model_cfg["dino"]
        if dino is not None:
            self.dino = dino
        else:
            dino_path = dino_cfg.get("path") or dino_cfg.get("model_id") or DINO_REPO
            self.dino = DINOv3DualViewEncoder(
                dino_path,
                revision=dino_cfg.get("revision", DINO_REVISION),
                selected_blocks=tuple(dino_cfg.get("selected_blocks", (8, 10))),
                image_size=int(dino_cfg.get("image_size", 256)),
            )
        self.dino.requires_grad_(False)

        expert_cfg = model_cfg["expert"]
        flow_cfg = model_cfg.get("flow", {})
        world_loss_weight = float(
            config.get("world_loss_weight", flow_cfg.get("world_loss_weight", 0.1))
        )
        action_tau_shift = float(flow_cfg.get("action_tau_shift", 1.0))
        world_tau_shift = float(flow_cfg.get("world_tau_shift", 5.0))
        solver_steps = int(flow_cfg.get("solver_steps", 10))

        expert = ResWAMExpert(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            horizon=self.horizon,
            num_world_views=len(self.world_views),
            dim=int(expert_cfg.get("hidden_dim", 1024)),
            heads=int(expert_cfg.get("heads", 16)),
            ffn_dim=int(expert_cfg.get("ffn_dim", 4096)),
            depth=int(expert_cfg.get("depth", 12)),
            variant=expert_cfg.get("variant", "bidirectional"),
            use_state_null_tokens=bool(
                expert_cfg.get("use_state_null_tokens", self.state_dropout_prob > 0.0)
            ),
            layout=str(expert_cfg.get("layout", "robotwin")),
        )
        self.objective = ResWAMFlowObjective(
            expert,
            world_loss_weight=world_loss_weight,
            action_tau_shift=action_tau_shift,
            world_tau_shift=world_tau_shift,
            solver_steps=solver_steps,
        )

    def forward(self, batch: dict[str, Any], *, seed: int) -> ObjectiveOutput:
        prepared = batch.get("qwen_current") or self.vlm.prepare(
            batch["qwen_cameras"], batch["instructions"]
        )
        context, vlm_mask = self.vlm(prepared)
        device = context.device
        with torch.no_grad():
            currents, residuals = [], []
            cached_cur = batch.get("dino_current")
            cached_end = batch.get("dino_endpoint")
            for view in self.world_views:
                if cached_cur is not None and cached_end is not None and view in cached_cur:
                    cur = unview_float32(cached_cur[view]).to(device, non_blocking=True)
                    end = unview_float32(cached_end[view]).to(device, non_blocking=True)
                else:
                    cur = self.dino(batch["current_cameras"][view].to(device, non_blocking=True))
                    end = self.dino(batch["endpoint_cameras"][view].to(device, non_blocking=True))
                currents.append(cur)
                residuals.append(self.objective.residual(cur, end))
            current = currents[0] if len(currents) == 1 else torch.cat(currents, dim=1)
            primary_target = residuals[0] if len(residuals) == 1 else torch.cat(residuals, dim=1)

        state = batch["state"].to(device, dtype=torch.float32)
        if "state_mask" in batch and self.state_dropout_prob <= 0.0:
            state_mask = batch["state_mask"].to(device)
        else:
            base_mask = (
                batch["state_mask"].to(device)
                if "state_mask" in batch
                else torch.ones_like(state, dtype=torch.bool)
            )
            drop_mask = draw_state_keep_mask(
                seed,
                state.shape[0],
                self.state_dim,
                self.state_dropout_prob if self.training else 0.0,
                device=device,
            )
            state_mask = base_mask.bool() & drop_mask

        generator = torch.Generator(device=device).manual_seed(int(seed) % (2**63 - 1))
        compute = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with compute:
            result = self.objective(
                actions=batch["actions"].to(device, dtype=torch.float32),
                action_mask=batch["action_mask"].to(device),
                primary_target=primary_target,
                current_primary=current,
                vlm_context=context,
                vlm_mask=vlm_mask,
                state=state,
                state_mask=state_mask,
                generator=generator,
            )
            if len(self.world_views) > 1:
                for idx, view in enumerate(self.world_views):
                    key = f"loss/world_view_{idx}_raw"
                    if key in result.metrics:
                        result.metrics[f"loss/world_{view}_raw"] = result.metrics.pop(key)
        if result.loss.dtype != torch.float32:
            raise RuntimeError("loss boundary must remain FP32")
        return result

    @torch.inference_mode()
    def sample_actions(
        self,
        *,
        cameras: Mapping[str, Sequence[np.ndarray]],
        instructions: Sequence[str],
        normalized_state: torch.Tensor,
        seeds: Sequence[int],
    ) -> torch.Tensor:
        """10-step synchronous Euler / Jacobi sampling at test time.

        Only requires current observations ``o_t``; predicts and discards the latent
        residual ``Delta_t`` without any pixel decoding.
        Returns normalized actions [B, horizon, action_dim].
        """
        processed = {
            name: torch.stack(
                [
                    preprocess_camera(torch.from_numpy(np.asarray(img, np.uint8)).permute(2, 0, 1))
                    for img in cameras[name]
                ]
            )
            for name in self.views
        }
        images = {
            name: [f.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy() for f in processed[name]]
            for name in self.views
        }
        context, mask = self.vlm(self.vlm.prepare(images, instructions))
        device = context.device
        primaries = [self.dino(processed[v].to(device)) for v in self.world_views]
        primary = primaries[0] if len(primaries) == 1 else torch.cat(primaries, dim=1)
        state = normalized_state.to(device, dtype=torch.float32)
        state_mask = torch.ones_like(state, dtype=torch.bool)
        actions, _, _ = synchronous_sample(
            self.objective.expert,
            current_primary=primary,
            vlm_context=context,
            vlm_mask=mask,
            state=state,
            state_mask=state_mask,
            seeds=seeds,
            action_shift=self.objective.action_tau_shift,
            world_shift=self.objective.world_tau_shift,
            steps=self.objective.solver_steps,
        )
        return actions


def system_config(
    *,
    action_dim: int,
    state_dim: int,
    horizon: int,
    views: Sequence[str],
    world_views: Sequence[str] | None = None,
    world_loss_weight: float = 0.1,
    action_tau_shift: float = 1.0,
    world_tau_shift: float = 5.0,
    solver_steps: int = 10,
    state_dropout_prob: float = 0.0,
    prompt_style: str = "multi_camera",
    rynnbrain_path: str = RYNN_REPO,
    dino_path: str = DINO_REPO,
    expert_layout: str | None = None,
    use_state_null_tokens: bool | None = None,
) -> dict[str, Any]:
    """Build standard system configuration dictionary.

    ``expert_layout`` defaults to ``"libero"`` for the LIBERO CoT prompt and ``"robotwin"``
    otherwise (see :class:`ResWAMExpert`). ``use_state_null_tokens`` defaults to
    ``state_dropout_prob > 0``.
    """
    if expert_layout is None:
        expert_layout = "libero" if prompt_style == "libero_cot" else "robotwin"
    if use_state_null_tokens is None:
        use_state_null_tokens = float(state_dropout_prob) > 0.0
    resolved_wv = list(resolve_world_views(world_views, views))
    return {
        "action_dim": int(action_dim),
        "state_dim": int(state_dim),
        "horizon": int(horizon),
        "views": list(views),
        "world_views": resolved_wv,
        "world_loss_weight": float(world_loss_weight),
        "state_dropout_prob": float(state_dropout_prob),
        "model": {
            "rynnbrain": {
                "path": str(rynnbrain_path),
                "revision": RYNN_REVISION,
                "text_layers": 16,
                "prompt_style": str(prompt_style),
            },
            "dino": {
                "path": str(dino_path),
                "revision": DINO_REVISION,
                "selected_blocks": [8, 10],
                "image_size": 256,
            },
            "expert": {
                "hidden_dim": 1024,
                "heads": 16,
                "ffn_dim": 4096,
                "depth": 12,
                "variant": "bidirectional",
                "use_state_null_tokens": bool(use_state_null_tokens),
                "layout": str(expert_layout),
            },
            "flow": {
                "world_loss_weight": float(world_loss_weight),
                "action_tau_shift": float(action_tau_shift),
                "world_tau_shift": float(world_tau_shift),
                "solver_steps": int(solver_steps),
            },
        },
    }
