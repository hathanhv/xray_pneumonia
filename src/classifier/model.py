from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class MobileNetV2Config:
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.2
    finetune_mode: str = "full"
    unfreeze_blocks: int = 3
    freeze_batchnorm: bool = False


@dataclass(frozen=True)
class CheckpointLoadReport:
    path: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    epoch: int | None
    class_to_idx: dict[str, int] | None


def build_mobilenet_v2(num_classes=2, pretrained=True, dropout=0.2):
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=pretrained)

    in_features = model.classifier[1].in_features
    model.classifier[0] = nn.Dropout(p=float(dropout), inplace=False)
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v2_from_config(config: Mapping[str, Any] | MobileNetV2Config):
    if isinstance(config, Mapping):
        allowed = MobileNetV2Config.__dataclass_fields__
        config = MobileNetV2Config(
            **{key: value for key, value in config.items() if key in allowed}
        )
    model = build_mobilenet_v2(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )
    finetune = configure_finetuning(
        model,
        mode=config.finetune_mode,
        unfreeze_blocks=config.unfreeze_blocks,
        freeze_batchnorm=config.freeze_batchnorm,
    )
    return model, {"model": asdict(config), "finetuning": finetune}


def configure_finetuning(
    model,
    mode="auto",
    train_size=None,
    unfreeze_blocks=3,
    freeze_batchnorm=False,
):
    """
    Configure which MobileNetV2 parameters are trainable.

    Modes:
        full: train the entire network
        head: train only the classifier head
        last_blocks: train classifier and the last N feature blocks
        auto: use last_blocks for small datasets, otherwise full
    """
    valid_modes = {"auto", "full", "head", "last_blocks"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown fine-tune mode: {mode}. Expected one of {sorted(valid_modes)}")

    selected_mode = mode
    if mode == "auto":
        selected_mode = "last_blocks" if train_size is None or train_size < 1000 else "full"

    for parameter in model.parameters():
        parameter.requires_grad = False

    if selected_mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif selected_mode == "head":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        block_count = len(model.features)
        unfreeze_blocks = max(1, min(int(unfreeze_blocks), block_count))

        for block in model.features[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return {
        "requested_mode": mode,
        "selected_mode": selected_mode,
        "unfreeze_blocks": unfreeze_blocks if selected_mode == "last_blocks" else 0,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100.0 * trainable_params / total_params if total_params else 0.0,
        "freeze_batchnorm": bool(freeze_batchnorm),
    }


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint

    raise ValueError("Could not find a model state_dict in checkpoint.")


def _strip_module_prefix(state_dict):
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint_if_available(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"No checkpoint found, start from ImageNet weights: {checkpoint_path}")
        return model

    print(f"Loading classifier checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as error:
        if "Weights only load failed" not in str(error):
            raise

        print(
            "Retry torch.load with weights_only=False because this checkpoint "
            "was saved with Python objects. Only do this for trusted local checkpoints."
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {len(unexpected)}")

    return model


def load_classifier_checkpoint(
    model,
    checkpoint_path,
    *,
    device="cpu",
    strict=True,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    report = CheckpointLoadReport(
        path=str(checkpoint_path),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        epoch=checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        class_to_idx=checkpoint.get("class_to_idx") if isinstance(checkpoint, dict) else None,
    )
    return checkpoint, report


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    class_to_idx,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": "mobilenet_v2",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "metrics": metrics,
            "class_to_idx": class_to_idx,
            "metadata": dict(metadata or {}),
        },
        path,
    )
