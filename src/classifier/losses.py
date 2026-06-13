from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class AlphaFocalLoss(nn.Module):
    """Focal loss with class alpha applied after the focal modulation."""

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


class SoftCrossEntropyLoss(nn.Module):
    def forward(self, logits, targets):
        if targets.ndim == 1:
            targets = F.one_hot(
                targets.long(),
                num_classes=logits.shape[1],
            ).float()
        if logits.shape != targets.shape:
            raise ValueError("Soft labels must have the same shape as logits")
        return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class SoftFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        if targets.ndim == 1:
            targets = F.one_hot(
                targets.long(),
                num_classes=logits.shape[1],
            ).float()
        if logits.shape != targets.shape:
            raise ValueError("Soft labels must have the same shape as logits")
        probabilities = F.softmax(logits, dim=1)
        log_probabilities = F.log_softmax(logits, dim=1)
        loss = -(
            targets * ((1.0 - probabilities) ** self.gamma) * log_probabilities
        )
        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)
        return loss.sum(dim=1).mean()


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(labels), minlength=num_classes).float()
    if torch.any(counts == 0):
        raise ValueError(f"Every class must have samples; counts={counts.tolist()}")
    return counts.sum() / (counts * num_classes)


def resolve_class_weights(
    config: Mapping[str, Any] | None,
    *,
    labels: Sequence[int],
    num_classes: int,
) -> torch.Tensor | None:
    config = dict(config or {})
    strategy = str(config.get("strategy", "none")).lower()
    if strategy in {"none", "disabled"}:
        return None
    if strategy in {"balanced", "inverse_frequency"}:
        return compute_class_weights(labels, num_classes)
    if strategy in {"manual", "explicit"}:
        values = config.get("values")
        if values is None or len(values) != num_classes:
            raise ValueError(
                f"Manual class weights require {num_classes} values"
            )
        weights = torch.as_tensor(values, dtype=torch.float32)
        if torch.any(weights <= 0):
            raise ValueError("Class weights must be positive")
        return weights
    raise ValueError(f"Unknown class-weight strategy: {strategy}")


def build_loss(
    config: Mapping[str, Any] | str,
    *,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    if isinstance(config, str):
        config = {"name": config}
    name = str(config.get("name", "cross_entropy")).lower()
    smoothing = float(config.get("label_smoothing", 0.0))
    if name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    if name in {"weighted_cross_entropy", "weighted_ce"}:
        if class_weights is None:
            raise ValueError("weighted_cross_entropy requires class_weights")
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)
    if name == "focal":
        if str(config.get("alpha_mode", "inside_ce")).lower() == "outside":
            return AlphaFocalLoss(
                gamma=config.get("gamma", 2.0),
                alpha=class_weights,
                label_smoothing=smoothing,
            )
        return FocalLoss(
            gamma=config.get("gamma", 2.0),
            weight=class_weights,
            label_smoothing=smoothing,
        )
    if name in {"soft_cross_entropy", "soft_ce"}:
        return SoftCrossEntropyLoss()
    if name == "soft_focal":
        return SoftFocalLoss(
            gamma=config.get("gamma", 2.0),
            alpha=class_weights,
        )
    raise ValueError(f"Unknown loss: {name}")
