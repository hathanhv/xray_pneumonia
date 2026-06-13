import numpy as np


def _empty_metrics(mask_area_ratio):
    return {
        "mask_area_ratio": mask_area_ratio,
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "bbox_area_ratio": 0.0,
        "bbox_height_ratio": 0.0,
        "bbox_width_ratio": 0.0,
        "bbox_bottom_ratio": 0.0,
        "mask_center_y_ratio": 0.0,
    }


def check_mask_quality(
    mask,
    bbox,
    image_shape,
    min_mask_area_ratio=0.06,
    max_mask_area_ratio=0.65,
    min_bbox_area_ratio=0.12,
    max_bbox_area_ratio=0.75,
    max_bbox_bottom_ratio=0.82,
    max_bbox_height_ratio=0.78,
    warning_mask_center_y_ratio=0.58,
):
    """
    Check lung crop quality using simple heuristic rules.

    Returns:
        qc_status, metrics
    """
    if mask is None:
        raise ValueError("mask must not be None")

    image_height, image_width = image_shape[:2]
    image_area = image_height * image_width
    mask_area = int((mask > 0).sum())
    mask_area_ratio = mask_area / image_area if image_area > 0 else 0.0

    if mask_area == 0 or bbox is None:
        metrics = _empty_metrics(mask_area_ratio)
        return "FAIL_EMPTY", metrics

    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])
    bbox_width = int(bbox.get("bbox_w", max(0, x2 - x1)))
    bbox_height = int(bbox.get("bbox_h", max(0, y2 - y1)))

    bbox_area_ratio = (
        (bbox_width * bbox_height) / image_area if image_area > 0 else 0.0
    )
    bbox_width_ratio = bbox_width / image_width if image_width > 0 else 0.0
    bbox_height_ratio = bbox_height / image_height if image_height > 0 else 0.0
    bbox_bottom_ratio = bbox.get(
        "bottom_ratio",
        y2 / image_height if image_height > 0 else 0.0,
    )

    ys, _ = np.where(mask > 0)
    mask_center_y_ratio = (
        float(ys.mean()) / image_height if len(ys) > 0 and image_height > 0 else 0.0
    )

    metrics = {
        "mask_area_ratio": mask_area_ratio,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_height_ratio": bbox_height_ratio,
        "bbox_width_ratio": bbox_width_ratio,
        "bbox_bottom_ratio": bbox_bottom_ratio,
        "mask_center_y_ratio": mask_center_y_ratio,
    }

    if mask_area_ratio < min_mask_area_ratio:
        qc_status = "FAIL_TOO_SMALL"
    elif mask_area_ratio > max_mask_area_ratio:
        qc_status = "FAIL_TOO_LARGE"
    elif bbox_area_ratio < min_bbox_area_ratio:
        qc_status = "FAIL_BAD_BBOX_SMALL"
    elif bbox_area_ratio > max_bbox_area_ratio:
        qc_status = "FAIL_BAD_BBOX_LARGE"
    elif bbox_bottom_ratio > max_bbox_bottom_ratio:
        qc_status = "FAIL_CROP_TOO_LOW"
    elif bbox_height_ratio > max_bbox_height_ratio:
        qc_status = "FAIL_BBOX_TOO_TALL"
    elif mask_center_y_ratio > warning_mask_center_y_ratio:
        qc_status = "WARNING_MASK_LOW"
    else:
        qc_status = "PASS"

    return qc_status, metrics
