"""Numerically constrained dual-stream DiT modules for ResWAM (stable-expert-v1)."""
from __future__ import annotations

from typing import Iterator, Literal

import torch
import torch.nn.functional as F
from torch import nn

STABLE_EXPERT_VERSION = "stable-expert-v1"
AttentionVariant = Literal["bidirectional", "world_to_action"]


def _rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    value = x.float()
    return value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def _zero_(linear: nn.Linear) -> None:
    linear.weight.zero_()
    linear.bias.zero_()


class HeadRMSNorm(nn.Module):
    """Per-head RMS normalization applied to queries and keys."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (_rms_normalize(x, self.eps) * self.weight.float()).to(x.dtype)


class ParameterFreeRMSNorm(nn.Module):
    """Parameter-free RMS normalization for flow timestep embeddings."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_normalize(x, self.eps).to(x.dtype)


class AdaLNZero(nn.Module):
    """Zero-initialized Adaptive LayerNorm modulation for DiT blocks."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        _zero_(self.modulation[-1])

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(condition).chunk(6, dim=-1)
        attention = self.norm(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        feedforward = self.norm(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        return attention, gate_a[:, None], feedforward, gate_f[:, None]


class StableStreamModules(nn.Module):
    """Per-modality (Action or World Residual) projections, QK-Norm, and FFN."""

    def __init__(self, dim: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.adaln = AdaLNZero(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )
        self.q_norm = HeadRMSNorm(dim // heads)
        self.k_norm = HeadRMSNorm(dim // heads)


class StableJointBlock(nn.Module):
    """Dual-stream joint attention block with per-head QK RMSNorm and AdaLN-Zero."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_dim: int,
        variant: AttentionVariant = "bidirectional",
    ) -> None:
        super().__init__()
        if variant not in ("bidirectional", "world_to_action"):
            raise ValueError(f"invalid attention variant: {variant}")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.variant = variant
        self.action = StableStreamModules(dim, heads, ffn_dim)
        self.world = StableStreamModules(dim, heads, ffn_dim)

    def _qkv(self, module: StableStreamModules, tokens: torch.Tensor):
        shape = (
            tokens.shape[0],
            tokens.shape[1],
            self.heads,
            self.dim // self.heads,
        )
        query, key, value = (
            item.reshape(shape).transpose(1, 2)
            for item in module.qkv(tokens).chunk(3, dim=-1)
        )
        return module.q_norm(query), module.k_norm(key), value

    def forward(
        self,
        action: torch.Tensor,
        primary: torch.Tensor,
        wrist: torch.Tensor,
        action_time: torch.Tensor,
        world_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_n, action_gate, action_ffn_n, action_ffn_gate = self.action.adaln(action, action_time)
        primary_n, primary_gate, primary_ffn_n, primary_ffn_gate = self.world.adaln(primary, world_time)
        wrist_n, wrist_gate, wrist_ffn_n, wrist_ffn_gate = self.world.adaln(wrist, world_time)
        world_n = torch.cat((primary_n, wrist_n), dim=1)
        aq, ak, av = self._qkv(self.action, action_n)
        wq, wk, wv = self._qkv(self.world, world_n)
        all_k, all_v = torch.cat((ak, wk), dim=2), torch.cat((av, wv), dim=2)
        action_update = F.scaled_dot_product_attention(aq, all_k, all_v)
        if self.variant == "bidirectional":
            world_update = F.scaled_dot_product_attention(wq, all_k, all_v)
        else:
            world_update = F.scaled_dot_product_attention(wq, wk, wv)
        action_update = action_update.transpose(1, 2).reshape_as(action)
        world_update = world_update.transpose(1, 2).reshape_as(world_n)
        primary_update, wrist_update = world_update.split(
            (primary.shape[1], wrist.shape[1]), dim=1
        )
        action = action + action_gate * self.action.attn_out(action_update)
        primary = primary + primary_gate * self.world.attn_out(primary_update)
        wrist = wrist + wrist_gate * self.world.attn_out(wrist_update)
        action = action + action_ffn_gate * self.action.ffn(action_ffn_n)
        primary = primary + primary_ffn_gate * self.world.ffn(primary_ffn_n)
        wrist = wrist + wrist_ffn_gate * self.world.ffn(wrist_ffn_n)
        return action, primary, wrist


class QKNormCrossContext(nn.Module):
    """Cross-attention conditioning module with per-head QK RMSNorm."""

    def __init__(self, dim: int, heads: int, context_dim: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = HeadRMSNorm(dim // heads)
        self.k_norm = HeadRMSNorm(dim // heads)

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, dim = value.shape
        return value.reshape(batch, length, self.heads, dim // self.heads).transpose(1, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_mask = None
        if context_mask is not None:
            if context_mask.shape != context.shape[:2]:
                raise ValueError("context_mask must be [B, S] matching context")
            visible = context_mask.bool()
            if not bool(visible.any(dim=-1).all()):
                raise ValueError("context_mask hides every context token of at least one sample")
            attn_mask = visible[:, None, None, :]
        normalized_context = self.context_norm(context)
        query = self.q_norm(self._split(self.q_proj(self.query_norm(tokens))))
        key = self.k_norm(self._split(self.k_proj(normalized_context)))
        value = self._split(self.v_proj(normalized_context))
        update = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        return tokens + self.out_proj(update.transpose(1, 2).reshape_as(tokens))


class FinalLayer(nn.Module):
    """Zero-initialized adaptive LayerNorm output head."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, out_dim)
        _zero_(self.modulation[-1])
        _zero_(self.linear)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale[:, None]) + shift[:, None])


def stable_time_mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim), ParameterFreeRMSNorm()
    )


def _zero_initialized(module: nn.Module) -> Iterator[tuple[str, nn.Linear]]:
    for name, child in module.named_modules():
        prefix = f"{name}." if name else ""
        if isinstance(child, (AdaLNZero, FinalLayer)):
            yield f"{prefix}modulation.1", child.modulation[-1]
        if isinstance(child, FinalLayer):
            yield f"{prefix}linear", child.linear


def reapply_zero_init(module: nn.Module) -> int:
    """Re-zero every AdaLNZero and FinalLayer linear projection."""
    targets = list(_zero_initialized(module))
    for _, linear in targets:
        _zero_(linear)
    return len(targets)


def assert_zero_init(module: nn.Module) -> int:
    """Verify all AdaLNZero and FinalLayer linear layers are strictly zero-initialized."""
    targets = list(_zero_initialized(module))
    if not targets:
        raise RuntimeError("no AdaLNZero/FinalLayer modules found")
    nonzero = [
        name
        for name, linear in targets
        if linear.weight.count_nonzero() or linear.bias.count_nonzero()
    ]
    if nonzero:
        raise RuntimeError(f"zero-initialized layers are nonzero: {nonzero}")
    return len(targets)
