from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

RIGHT_LUNG = 1
LEFT_LUNG = 2
HEART = 3


@dataclass(frozen=True)
class SpatialFusionConfig:
    heatmap_threshold: float = 0.60
    min_lesion_pixels: int = 100
    bilateral_min_share: float = 0.20
    costophrenic_fraction: float = 0.18


class SpatialFusionEngine:
    """Fuse a class-specific CXFormer heatmap with anatomy masks.

    The heatmap-derived mask is an estimated localization mask, not GT.
    Zone determination uses relative lung height as a reproducible heuristic.
    """

    def __init__(self, config: Optional[SpatialFusionConfig] = None):
        self.config = config or SpatialFusionConfig()

    def fuse(
        self,
        finding: str,
        heatmap: np.ndarray,
        anatomy_mask: np.ndarray,
        heatmap_threshold: Optional[float] = None,
    ) -> Dict:
        heatmap = self._normalize_heatmap(heatmap)
        anatomy_mask = self._prepare_anatomy(anatomy_mask, heatmap.shape)

        threshold = float(
            self.config.heatmap_threshold
            if heatmap_threshold is None
            else heatmap_threshold
        )
        lesion_mask = heatmap >= threshold

        right_lung = anatomy_mask == RIGHT_LUNG
        left_lung = anatomy_mask == LEFT_LUNG
        lung_union = right_lung | left_lung
        lesion_in_lungs = lesion_mask & lung_union

        right_overlap = lesion_in_lungs & right_lung
        left_overlap = lesion_in_lungs & left_lung

        right_count = int(right_overlap.sum())
        left_count = int(left_overlap.sum())
        total_count = right_count + left_count

        valid = total_count >= int(self.config.min_lesion_pixels)
        side = self._determine_side(right_count, left_count) if valid else "unknown"
        zone = (
            self._determine_zone(
                finding=finding,
                lesion_mask=lesion_in_lungs,
                anatomy_mask=anatomy_mask,
                side=side,
            )
            if valid
            else "unknown"
        )

        right_pixels = int(right_lung.sum())
        left_pixels = int(left_lung.sum())
        right_burden = 100.0 * right_count / right_pixels if right_pixels else None
        left_burden = 100.0 * left_count / left_pixels if left_pixels else None

        if side == "right":
            reference_region = "right_lung"
            burden = right_burden
        elif side == "left":
            reference_region = "left_lung"
            burden = left_burden
        elif side == "bilateral":
            reference_region = "both_lungs"
            lung_pixels = right_pixels + left_pixels
            burden = 100.0 * total_count / lung_pixels if lung_pixels else None
        else:
            reference_region = None
            burden = None

        return {
            "localization": {
                "type": "cxformer_transformer_gradcam",
                "side": side,
                "zone": zone,
                "mask_semantics": "estimated_localization_mask",
                "heatmap_threshold": threshold,
                "threshold_source": "cxformer_heatmap_config",
                "positive_pixels": total_count,
                "valid": bool(valid),
                "zone_method": "relative_lung_height_heuristic",
            },
            "measurements": {
                "projected_2d_burden_pct": None if burden is None else round(float(burden), 2),
                "right_lung_projected_2d_burden_pct": (
                    None if right_burden is None else round(float(right_burden), 2)
                ),
                "left_lung_projected_2d_burden_pct": (
                    None if left_burden is None else round(float(left_burden), 2)
                ),
                "reference_region": reference_region,
                "method": "cxformer_heatmap_anatomy_overlap",
                "valid": bool(valid),
            },
        }

    def _determine_side(self, right_count: int, left_count: int) -> str:
        total = right_count + left_count
        if total <= 0:
            return "unknown"

        right_share = right_count / total
        left_share = left_count / total
        min_share = float(self.config.bilateral_min_share)

        if right_share >= min_share and left_share >= min_share:
            return "bilateral"
        return "right" if right_count > left_count else "left"

    def _determine_zone(
        self,
        finding: str,
        lesion_mask: np.ndarray,
        anatomy_mask: np.ndarray,
        side: str,
    ) -> str:
        if side == "right":
            reference = anatomy_mask == RIGHT_LUNG
        elif side == "left":
            reference = anatomy_mask == LEFT_LUNG
        else:
            reference = (anatomy_mask == RIGHT_LUNG) | (anatomy_mask == LEFT_LUNG)

        lesion = lesion_mask & reference
        ys = np.where(lesion)[0]
        lung_ys = np.where(reference)[0]
        if ys.size == 0 or lung_ys.size == 0:
            return "unknown"

        y_min = float(lung_ys.min())
        y_max = float(lung_ys.max())
        height = max(1.0, y_max - y_min)
        relative_y = (float(np.median(ys)) - y_min) / height

        lower_name = str(finding).lower()
        if "effusion" in lower_name and relative_y >= 1.0 - self.config.costophrenic_fraction:
            return "costophrenic"
        if relative_y < 1.0 / 3.0:
            return "upper"
        if relative_y < 2.0 / 3.0:
            return "middle"
        return "lower"

    @staticmethod
    def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        if heatmap.ndim != 2:
            heatmap = np.squeeze(heatmap)
        if heatmap.ndim != 2:
            raise ValueError(f"Expected 2-D heatmap, got shape={heatmap.shape}")

        min_value = float(np.nanmin(heatmap))
        max_value = float(np.nanmax(heatmap))
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            raise ValueError("Heatmap contains non-finite values")
        if max_value <= min_value:
            return np.zeros_like(heatmap, dtype=np.float32)
        return (heatmap - min_value) / (max_value - min_value)

    @staticmethod
    def _prepare_anatomy(anatomy_mask: np.ndarray, target_shape) -> np.ndarray:
        anatomy_mask = np.asarray(anatomy_mask)
        anatomy_mask = np.squeeze(anatomy_mask)
        if anatomy_mask.ndim != 2:
            raise ValueError(f"Expected 2-D anatomy mask, got shape={anatomy_mask.shape}")
        if anatomy_mask.shape != tuple(target_shape):
            anatomy_mask = cv2.resize(
                anatomy_mask.astype(np.uint8),
                (int(target_shape[1]), int(target_shape[0])),
                interpolation=cv2.INTER_NEAREST,
            )
        return anatomy_mask.astype(np.uint8)
