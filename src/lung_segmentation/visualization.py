import cv2


def create_mask_overlay(image, mask, alpha=0.35, color=(0, 0, 255)):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("image and mask dimensions must match")

    if image.ndim == 2:
        overlay_base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay_base = image.copy()

    mask_color = overlay_base.copy()
    mask_color[mask > 0] = color
    return cv2.addWeighted(mask_color, alpha, overlay_base, 1 - alpha, 0)
