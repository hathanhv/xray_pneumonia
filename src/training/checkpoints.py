from __future__ import annotations

from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, best_path, last_path):
        self.best_path = Path(best_path)
        self.last_path = Path(last_path)
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_path.parent.mkdir(parents=True, exist_ok=True)

    def build_state(
        self,
        *,
        epoch,
        model,
        optimizer,
        scheduler,
        early_stopping,
        metrics,
        history,
        metadata=None,
    ):
        return {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "early_stopping_state_dict": early_stopping.state_dict(),
            "metrics": dict(metrics),
            "history": list(history),
            "metadata": dict(metadata or {}),
        }

    def save(self, state, *, is_best):
        torch.save(state, self.last_path)
        if is_best:
            torch.save(state, self.best_path)

    def resume(
        self,
        path,
        *,
        model,
        optimizer=None,
        scheduler=None,
        early_stopping=None,
        device="cpu",
    ):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if early_stopping is not None and checkpoint.get("early_stopping_state_dict"):
            early_stopping.load_state_dict(checkpoint["early_stopping_state_dict"])
        return checkpoint
