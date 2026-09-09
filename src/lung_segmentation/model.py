from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "lung_segmentation"
    / "unet_lung_segmentation.pth"
)


@dataclass(frozen=True)
class LungModelMetadata:
    checkpoint_path: Path
    encoder: str
    img_size: int
    best_val_loss: float | None = None


def _strip_module_prefix(state_dict):
    """Handle checkpoints saved from DataParallel."""
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu"):
    try:
        import torch
    except ImportError as error:
        raise ImportError("Missing dependency: torch is required.") from error

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
    if not isinstance(checkpoint, dict):
        raise TypeError("Lung segmentation checkpoint must be a dictionary.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing required key: model_state_dict")
    return checkpoint


def get_checkpoint_metadata(
    checkpoint,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
):
    return LungModelMetadata(
        checkpoint_path=Path(checkpoint_path),
        encoder=str(checkpoint.get("encoder", "resnet34")),
        img_size=int(checkpoint.get("img_size", 256)),
        best_val_loss=checkpoint.get("best_val_loss"),
    )


def build_lung_segmentation_model(
    encoder="resnet34",
    *,
    encoder_weights=None,
    in_channels=3,
    classes=1,
):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise ImportError(
            "Missing dependency: segmentation_models_pytorch is required."
        ) from error

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=int(in_channels),
        classes=int(classes),
        activation=None,
    )


def build_unet_from_config(config):
    encoder = str(config.get("encoder", "resnet34"))
    model = build_lung_segmentation_model(
        encoder,
        encoder_weights=config.get("encoder_weights"),
        in_channels=int(config.get("in_channels", 3)),
        classes=int(config.get("classes", 1)),
    )
    return model, {
        "architecture": "unet",
        "encoder": encoder,
        "encoder_weights": config.get("encoder_weights"),
        "img_size": int(config.get("image_size", 256)),
        "in_channels": int(config.get("in_channels", 3)),
        "classes": int(config.get("classes", 1)),
    }


def load_lung_segmentation_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=None):
    """
    Load UNet ResNet34 lung segmentation model from checkpoint.

    Returns:
        model: loaded torch model in eval mode
        img_size: input image size saved in checkpoint
        device: torch.device used for inference
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Missing dependency. Please install torch and "
            "segmentation_models_pytorch before loading the lung model."
        ) from error

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    metadata = get_checkpoint_metadata(checkpoint, checkpoint_path)

    print(f"Loading lung segmentation checkpoint: {checkpoint_path}")
    print(f"Encoder: {metadata.encoder}")
    print(f"Image size: {metadata.img_size}")
    print(f"Device: {device}")

    model = build_lung_segmentation_model(metadata.encoder)

    state_dict = _strip_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, metadata.img_size, device


if __name__ == "__main__":
    load_lung_segmentation_model()
