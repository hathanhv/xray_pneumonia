from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        targets = targets.float().flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        *,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight=None,
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice = BinaryDiceLoss(smooth=smooth)
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.as_tensor([pos_weight], dtype=torch.float32),
        )

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
        )
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * bce


def build_segmentation_loss(config: Mapping):
    name = str(config.get("name", "dice_bce")).lower()
    if name in {"dice", "dice_loss"}:
        return BinaryDiceLoss(smooth=float(config.get("smooth", 1.0)))
    if name in {"bce", "bce_with_logits"}:
        pos_weight = config.get("pos_weight")
        tensor = None if pos_weight is None else torch.tensor([float(pos_weight)])
        return nn.BCEWithLogitsLoss(pos_weight=tensor)
    if name in {"dice_bce", "bce_dice", "combined"}:
        return DiceBCELoss(
            dice_weight=float(config.get("dice_weight", 0.5)),
            bce_weight=float(config.get("bce_weight", 0.5)),
            smooth=float(config.get("smooth", 1.0)),
            pos_weight=config.get("pos_weight"),
        )
    raise ValueError(f"Unknown segmentation loss: {name}")
