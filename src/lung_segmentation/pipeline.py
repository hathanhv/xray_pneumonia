from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import LungSegmentationConfig
from .crop import crop_by_mask
from .model import load_lung_segmentation_model
from .postprocess import clean_mask
from .predict import predict_lung_mask
from .qc import check_mask_quality
from .visualization import create_mask_overlay


@dataclass
class SegmentationResult:
    image_path: Path
    image: np.ndarray
    raw_mask: np.ndarray
    mask: np.ndarray
    crop: np.ndarray | None
    overlay: np.ndarray
    bbox: dict[str, Any] | None
    qc_status: str
    qc_metrics: dict[str, Any]

    def report_row(self) -> dict[str, Any]:
        return {
            "filename": self.image_path.name,
            "qc_status": self.qc_status,
            **self.qc_metrics,
        }


class LungSegmentationPipeline:
    def __init__(
        self,
        config: LungSegmentationConfig,
        *,
        model=None,
        img_size=None,
        device=None,
    ):
        self.config = config
        if model is None:
            model, img_size, device = load_lung_segmentation_model(
                checkpoint_path=config.model.checkpoint_path,
                device=config.model.device,
            )
        self.model = model
        self.img_size = int(img_size)
        self.device = device

    def predict(self, image_path: str | Path) -> SegmentationResult:
        image_path = Path(image_path)
        image, raw_mask = predict_lung_mask(
            self.model,
            image_path,
            self.img_size,
            self.device,
            threshold=self.config.model.threshold,
        )

        postprocess = self.config.postprocess
        mask = clean_mask(
            raw_mask,
            keep_components=postprocess.keep_components,
            fill_holes=postprocess.fill_holes,
        )

        crop_config = asdict(self.config.crop)
        crop, bbox = crop_by_mask(image=image, mask=mask, **crop_config)
        qc_status, qc_metrics = check_mask_quality(
            mask,
            bbox,
            image.shape,
            **asdict(self.config.qc),
        )

        alpha = self.config.output.overlay_alpha if self.config.output else 0.35
        overlay = create_mask_overlay(image, mask, alpha=alpha)
        return SegmentationResult(
            image_path=image_path,
            image=image,
            raw_mask=raw_mask,
            mask=mask,
            crop=crop,
            overlay=overlay,
            bbox=bbox,
            qc_status=qc_status,
            qc_metrics=qc_metrics,
        )

    def save_result(
        self,
        result: SegmentationResult,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        output_config = self.config.output
        if output_dir is None:
            if output_config is None:
                raise ValueError("output_dir is required when output config is absent.")
            output_dir = output_config.output_dir
        output_dir = Path(output_dir)

        save_mask = output_config.save_mask if output_config else True
        save_crop = output_config.save_crop if output_config else True
        save_overlay = output_config.save_overlay if output_config else True

        directories = {
            "mask": output_dir / "masks",
            "crop": output_dir / "cropped",
            "overlay": output_dir / "overlays",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        stem = result.image_path.stem
        paths: dict[str, Path] = {}
        if save_mask:
            path = directories["mask"] / f"{stem}_mask.png"
            self._write_image(path, result.mask * 255)
            paths["mask"] = path
        if save_crop and result.crop is not None:
            path = directories["crop"] / f"{stem}_crop.png"
            self._write_image(path, result.crop)
            paths["crop"] = path
        if save_overlay:
            path = directories["overlay"] / f"{stem}_overlay.png"
            self._write_image(path, result.overlay)
            paths["overlay"] = path
        return paths

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise IOError(f"Could not write image: {path}")
