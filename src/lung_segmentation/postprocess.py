import cv2
import numpy as np


def clean_mask(mask, keep_components=2, fill_holes=True):
    """
    Clean predicted lung mask.

    Keeps the largest 1 or 2 connected components because a chest X-ray usually
    contains two lung fields.
    """
    if mask is None:
        raise ValueError("mask must not be None")
    if int(keep_components) < 1:
        raise ValueError("keep_components must be at least 1")

    binary_mask = (mask > 0).astype(np.uint8)
    if binary_mask.sum() == 0:
        return binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    components = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        components.append((label, area))

    components = sorted(components, key=lambda item: item[1], reverse=True)
    kept_labels = [label for label, _ in components[: int(keep_components)]]

    cleaned = np.isin(labels, kept_labels).astype(np.uint8)

    if fill_holes:
        cleaned = fill_mask_holes(cleaned)

    return cleaned


def fill_mask_holes(mask):
    """Fill holes inside a binary mask."""
    mask_uint8 = (mask > 0).astype(np.uint8)
    height, width = mask_uint8.shape[:2]

    flood_fill = mask_uint8.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood_fill, flood_mask, (0, 0), 1)

    holes = (flood_fill == 0).astype(np.uint8)
    filled = np.maximum(mask_uint8, holes)

    return filled.astype(np.uint8)
