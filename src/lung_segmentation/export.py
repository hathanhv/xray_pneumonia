from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import torch


def _load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def export_monai_checkpoint(
    source_path,
    destination_path,
    *,
    backup_existing=True,
):
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    checkpoint = _load(source_path)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing required key: model_state_dict")
    metadata = checkpoint.get("metadata", {})
    exported = {
        "model_state_dict": checkpoint["model_state_dict"],
        "encoder": checkpoint.get("encoder", metadata.get("encoder", "resnet34")),
        "img_size": int(checkpoint.get("img_size", metadata.get("img_size", 256))),
        "best_val_loss": checkpoint.get(
            "best_val_loss",
            metadata.get("best_val_loss"),
        ),
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if destination_path.exists() and backup_existing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = destination_path.with_suffix(
            destination_path.suffix + f".backup_{timestamp}"
        )
        shutil.copy2(destination_path, backup_path)
    torch.save(exported, destination_path)
    return destination_path, backup_path
