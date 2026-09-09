from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.core.paths import PROJECT_ROOT, resolve_path


@dataclass(frozen=True)
class ModelConfig:
    checkpoint_path: Path
    device: str | None = None
    threshold: float = 0.5


@dataclass(frozen=True)
class PostprocessConfig:
    keep_components: int = 2
    fill_holes: bool = True


@dataclass(frozen=True)
class CropConfig:
    pad_left: int = 90
    pad_right: int = 90
    pad_top: int = 60
    pad_bottom: int = 8
    max_bottom_ratio: float = 0.75


@dataclass(frozen=True)
class QCConfig:
    min_mask_area_ratio: float = 0.06
    max_mask_area_ratio: float = 0.65
    min_bbox_area_ratio: float = 0.12
    max_bbox_area_ratio: float = 0.75
    max_bbox_bottom_ratio: float = 0.82
    max_bbox_height_ratio: float = 0.78
    warning_mask_center_y_ratio: float = 0.58


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    save_mask: bool = True
    save_crop: bool = True
    save_overlay: bool = True
    overlay_alpha: float = 0.35


@dataclass(frozen=True)
class LungSegmentationConfig:
    model: ModelConfig
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    output: OutputConfig | None = None

    @classmethod
    def from_dict(
        cls,
        config: Mapping[str, Any],
        *,
        project_root: str | Path = PROJECT_ROOT,
    ) -> "LungSegmentationConfig":
        section = config.get("lung_segmentation", config)
        if not isinstance(section, Mapping):
            raise ValueError("'lung_segmentation' config must be a mapping.")

        model_values = dict(section.get("model", {}))
        checkpoint_value = model_values.get(
            "checkpoint_path",
            "checkpoints/lung_segmentation/unet_lung_segmentation.pth",
        )
        model_values["checkpoint_path"] = resolve_path(
            checkpoint_value,
            Path(project_root),
        )
        model = ModelConfig(**model_values)

        postprocess = PostprocessConfig(**dict(section.get("postprocess", {})))
        crop = CropConfig(**dict(section.get("crop", {})))
        qc = QCConfig(**dict(section.get("qc", {})))

        output_values = section.get("output")
        output = None
        if output_values is not None:
            output_values = dict(output_values)
            output_values["output_dir"] = resolve_path(
                output_values["output_dir"],
                Path(project_root),
            )
            output = OutputConfig(**output_values)

        parsed = cls(
            model=model,
            postprocess=postprocess,
            crop=crop,
            qc=qc,
            output=output,
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if not 0.0 <= self.model.threshold <= 1.0:
            raise ValueError("model.threshold must be between 0 and 1.")
        if self.postprocess.keep_components < 1:
            raise ValueError("postprocess.keep_components must be at least 1.")
        for name in ("pad_left", "pad_right", "pad_top", "pad_bottom"):
            if getattr(self.crop, name) < 0:
                raise ValueError(f"crop.{name} must be non-negative.")
        if not 0.0 < self.crop.max_bottom_ratio <= 1.0:
            raise ValueError("crop.max_bottom_ratio must be in (0, 1].")
        if self.output and not 0.0 <= self.output.overlay_alpha <= 1.0:
            raise ValueError("output.overlay_alpha must be between 0 and 1.")
