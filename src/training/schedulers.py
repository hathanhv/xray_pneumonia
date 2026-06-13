from __future__ import annotations

from typing import Any, Mapping

import torch


def build_scheduler(optimizer, config: Mapping[str, Any] | None):
    if not config:
        return None
    name = str(config.get("name", "none")).lower()
    if name in {"none", "disabled"}:
        return None
    if name in {"cosine", "cosine_annealing"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["t_max"]),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name in {"step", "step_lr"}:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config["step_size"]),
            gamma=float(config.get("gamma", 0.1)),
        )
    if name in {"reduce_on_plateau", "reduce_lr_on_plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "min")),
            factor=float(config.get("factor", 0.1)),
            patience=int(config.get("patience", 3)),
            min_lr=float(config.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def step_scheduler(scheduler, metrics, monitor):
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metrics[monitor])
    else:
        scheduler.step()
