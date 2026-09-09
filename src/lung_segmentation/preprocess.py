from pathlib import Path

import cv2
import numpy as np


def load_image(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {image.shape}: {image_path}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def convert_to_rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def image_to_tensor(image, img_size=256):
    try:
        import torch
    except ImportError as error:
        raise ImportError("Missing dependency: torch is required for preprocessing.") from error

    image_rgb = convert_to_rgb(image)
    resized = cv2.resize(
        image_rgb,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA,
    )
    resized = resized.astype(np.float32) / 255.0
    return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()


def read_image(image_path, img_size=256):
    """
    Read an X-ray image and prepare model input.

    Args:
        image_path: Path to image file.
        img_size: Target square size for the segmentation model.

    Returns:
        tensor: Float tensor with shape [1, 3, img_size, img_size].
        original_image: Original grayscale or BGR image read by cv2.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError("Missing dependency: torch is required for preprocessing.") from error

    original_image = load_image(image_path)
    return image_to_tensor(original_image, img_size=img_size), original_image
