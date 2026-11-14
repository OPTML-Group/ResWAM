"""Linear warmup + cosine decay learning-rate scheduler for ResWAM."""
from __future__ import annotations

import math

import torch


def warmup_cosine_value(
    update: int, peak: float, minimum: float, *, warmup: int, total: int
) -> float:
    update = int(update)
    if not 0 <= update <= total:
        raise ValueError(f"optimizer update {update} outside [0, {total}]")
    if update == warmup:
        return peak
    if update < warmup:
        return peak * update / warmup
    progress = (update - warmup) / (total - warmup)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


class WarmupCosine:
    """Linear warmup through ``warmup`` updates, then cosine decay to ``min_lr`` at ``total``."""

    implementation = "reswam_warmup_cosine_v1"

    def __init__(self, optimizer: torch.optim.Optimizer, *, warmup: int, total: int) -> None:
        if not 0 < warmup < total:
            raise ValueError("need 0 < warmup < total")
        self.optimizer = optimizer
        self.warmup = int(warmup)
        self.total = int(total)
        self.step_count = 0
        self._apply(0)

    def value(self, update: int, peak: float, minimum: float) -> float:
        return warmup_cosine_value(update, peak, minimum, warmup=self.warmup, total=self.total)

    def _apply(self, update: int) -> dict[str, float]:
        values = {}
        for group in self.optimizer.param_groups:
            group["lr"] = self.value(update, float(group["peak_lr"]), float(group["min_lr"]))
            values[f"lr/{group['name']}"] = float(group["lr"])
        return values

    def step(self) -> dict[str, float]:
        """Set the learning rate of update ``step_count + 1``, then execute ``optimizer.step()``."""
        update = self.step_count + 1
        values = self._apply(update)
        self.optimizer.step()
        self.step_count = update
        return values

    def state_dict(self) -> dict[str, int | str]:
        return {
            "step_count": self.step_count,
            "warmup": self.warmup,
            "total": self.total,
            "implementation": self.implementation,
        }

    def load_state_dict(self, state: dict[str, int | str]) -> None:
        if int(state["warmup"]) != self.warmup or int(state["total"]) != self.total:
            raise RuntimeError(
                f"scheduler mismatch: checkpoint {state}, run warmup={self.warmup} total={self.total}"
            )
        self.step_count = int(state["step_count"])
        self._apply(self.step_count)


class ConstantFloor:
    """Constant learning rate at each group's ``min_lr`` for updates ``start + 1 .. total``.

    Used for the second RoboDojo stage (200k -> 400k): training resumes from the final
    resume state of a :class:`WarmupCosine` run that ended at ``start`` (the cosine floor),
    and every later update uses ``lr = min_lr`` (VLM 1e-6, expert 1e-5).
    ``load_state_dict`` accepts this scheduler's own state or, once, the state of the source
    :class:`WarmupCosine` run at exactly ``step_count == start`` (scheduler migration).
    """

    implementation = "reswam_constant_floor_v1"

    def __init__(self, optimizer: torch.optim.Optimizer, *, start: int, total: int) -> None:
        if not 0 <= start < total:
            raise ValueError("need 0 <= start < total")
        self.optimizer = optimizer
        self.start = int(start)
        self.total = int(total)
        self.step_count = self.start
        self._apply(self.start)

    def value(self, update: int, peak: float, minimum: float) -> float:
        del peak
        if not self.start <= int(update) <= self.total:
            raise ValueError(f"optimizer update {update} outside [{self.start}, {self.total}]")
        return float(minimum)

    def _apply(self, update: int) -> dict[str, float]:
        values = {}
        for group in self.optimizer.param_groups:
            group["lr"] = self.value(update, float(group["peak_lr"]), float(group["min_lr"]))
            values[f"lr/{group['name']}"] = float(group["lr"])
        return values

    def step(self) -> dict[str, float]:
        update = self.step_count + 1
        values = self._apply(update)
        self.optimizer.step()
        self.step_count = update
        return values

    def state_dict(self) -> dict[str, int | str]:
        return {
            "step_count": self.step_count,
            "start": self.start,
            "total": self.total,
            "implementation": self.implementation,
        }

    def load_state_dict(self, state: dict[str, int | str]) -> None:
        implementation = state.get("implementation")
        if implementation == WarmupCosine.implementation:
            if int(state["step_count"]) != self.start or int(state["total"]) != self.start:
                raise RuntimeError(
                    f"scheduler migration needs the source cosine run's final state at step "
                    f"{self.start} (cosine total {self.start}); got {state}"
                )
        elif implementation == self.implementation:
            if int(state["start"]) != self.start or int(state["total"]) != self.total:
                raise RuntimeError(
                    f"scheduler mismatch: checkpoint {state}, run start={self.start} total={self.total}"
                )
        else:
            raise RuntimeError(f"unsupported scheduler state {implementation!r}")
        self.step_count = int(state["step_count"])
        self._apply(self.step_count)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    kind: str,
    warmup: int,
    total: int,
    start: int = 0,
):
    """``kind`` is ``"warmup_cosine"`` (default) or ``"constant_floor"``."""
    if kind == "warmup_cosine":
        if start:
            raise ValueError("warmup_cosine runs start at update 0")
        return WarmupCosine(optimizer, warmup=warmup, total=total)
    if kind == "constant_floor":
        return ConstantFloor(optimizer, start=start, total=total)
    raise ValueError(f"unknown lr_schedule {kind!r}")
