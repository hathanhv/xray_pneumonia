from __future__ import annotations

from typing import Any, Mapping

from torchvision import transforms


def build_augmentation(
    config: Mapping[str, Any] | str | None,
    *,
    training: bool,
):
    if not training:
        return transforms.Compose([])

    if isinstance(config, str):
        config = {"name": config}
    config = dict(config or {"name": "baseline"})
    name = str(config.get("name", "baseline")).lower()
    horizontal_flip = float(config.get("horizontal_flip", 0.5))

    if name in {"none", "eval"}:
        operations = []
    elif name == "baseline":
        operations = [
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            transforms.RandomRotation(degrees=float(config.get("rotation", 7))),
        ]
    elif name in {"fp_reduction", "false_positive_reduction"}:
        operations = [
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            transforms.RandomAffine(
                degrees=float(config.get("rotation", 5)),
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(brightness=0.08, contrast=0.12),
        ]
    elif name == "strong":
        operations = [
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            transforms.RandomAffine(
                degrees=float(config.get("rotation", 12)),
                translate=(0.08, 0.08),
                scale=(0.85, 1.15),
                shear=5,
            ),
            transforms.ColorJitter(brightness=0.18, contrast=0.22),
            transforms.RandomAutocontrast(p=0.25),
        ]
    elif name in {"light", "light_head_finetuning"}:
        operations = [
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            transforms.RandomRotation(degrees=float(config.get("rotation", 3))),
            transforms.ColorJitter(brightness=0.04, contrast=0.06),
        ]
    else:
        raise ValueError(f"Unknown augmentation strategy: {name}")
    return transforms.Compose(operations)
