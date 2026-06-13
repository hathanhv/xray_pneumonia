import numpy as np


def crop_by_mask(
    image,
    mask,
    pad_left=90,
    pad_right=90,
    pad_top=60,
    pad_bottom=8,
    max_bottom_ratio=0.75,
):
    """
    Crop image using the bounding box of a binary mask.

    Returns:
        cropped_image, bbox

    bbox is a dict with:
        x1, y1, x2, y2, bbox_w, bbox_h, bottom_ratio
    """
    if image is None:
        raise ValueError("image must not be None")
    if mask is None:
        raise ValueError("mask must not be None")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"image and mask size mismatch: image={image.shape[:2]}, mask={mask.shape[:2]}"
        )
    for name, value in {
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0.0 < max_bottom_ratio <= 1.0:
        raise ValueError("max_bottom_ratio must be in (0, 1]")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    height, width = image.shape[:2]
    max_bottom = int(height * max_bottom_ratio)

    x1 = max(int(xs.min()) - pad_left, 0)
    y1 = max(int(ys.min()) - pad_top, 0)
    x2 = min(int(xs.max()) + pad_right + 1, width)
    y2 = min(int(ys.max()) + pad_bottom + 1, height)

    if y2 > max_bottom:
        y2 = max_bottom

    if x2 <= x1 or y2 <= y1:
        return None, None

    cropped_image = image[y1:y2, x1:x2].copy()
    bbox = {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "bbox_w": int(x2 - x1),
        "bbox_h": int(y2 - y1),
        "bottom_ratio": y2 / height if height > 0 else 0.0,
    }

    return cropped_image, bbox
