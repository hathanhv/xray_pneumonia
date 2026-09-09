from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PreprocessResult:
    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)


class PreprocessingStrategy(Protocol):
    def __call__(
        self,
        image: Image.Image,
        metadata: Mapping[str, Any] | None = None,
    ) -> PreprocessResult: ...


class RawPreprocessing:
    def __call__(self, image, metadata=None):
        return PreprocessResult(image.convert("RGB"), {"preprocessing": "raw"})


class ResizePreprocessing:
    def __init__(self, size: int | Sequence[int] = 224):
        self.size = _size_pair(size)

    def __call__(self, image, metadata=None):
        resized = image.convert("RGB").resize(self.size[::-1], Image.Resampling.BILINEAR)
        return PreprocessResult(resized, {"preprocessing": "resize", "size": self.size})


class LegacyClassifierPreprocessing(ResizePreprocessing):
    """Match the notebook order: grayscale to three channels, then resize."""

    def __call__(self, image, metadata=None):
        grayscale = image.convert("L").convert("RGB")
        resized = grayscale.resize(self.size[::-1], Image.Resampling.BILINEAR)
        return PreprocessResult(
            resized,
            {"preprocessing": "legacy_classifier", "size": self.size},
        )


class ResizeWithPaddingPreprocessing:
    def __init__(self, size: int | Sequence[int] = 224, fill: int = 0):
        self.size = _size_pair(size)
        self.fill = int(fill)

    def __call__(self, image, metadata=None):
        image = image.convert("RGB")
        fitted = ImageOps.contain(image, self.size[::-1], Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", self.size[::-1], color=(self.fill,) * 3)
        offset = ((canvas.width - fitted.width) // 2, (canvas.height - fitted.height) // 2)
        canvas.paste(fitted, offset)
        return PreprocessResult(
            canvas,
            {
                "preprocessing": "resize_with_padding",
                "size": self.size,
                "content_size": (fitted.height, fitted.width),
                "offset": offset,
            },
        )


class HistogramMatchingPreprocessing:
    def __init__(self, reference_path: str | Path):
        self.reference_path = Path(reference_path)
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Histogram reference not found: {self.reference_path}")
        self.reference = np.asarray(Image.open(self.reference_path).convert("L"))

    def __call__(self, image, metadata=None):
        source = np.asarray(image.convert("L"))
        matched = _match_histogram(source, self.reference)
        output = Image.fromarray(matched, mode="L").convert("RGB")
        return PreprocessResult(
            output,
            {
                "preprocessing": "histogram_matching",
                "reference_path": str(self.reference_path),
            },
        )


class LungROIPreprocessing:
    def __init__(
        self,
        *,
        mask_key: str = "mask_path",
        fallback: PreprocessingStrategy | None = None,
        padding: int = 0,
    ):
        self.mask_key = mask_key
        self.fallback = fallback or RawPreprocessing()
        self.padding = int(padding)

    def __call__(self, image, metadata=None):
        metadata = dict(metadata or {})
        mask_path = metadata.get(self.mask_key)
        if not mask_path or not Path(mask_path).exists():
            result = self.fallback(image, metadata)
            result.metadata.update(
                {"roi_source": "fallback", "roi_fallback_reason": "mask_missing"}
            )
            return result

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask > 0):
            result = self.fallback(image, metadata)
            result.metadata.update(
                {"roi_source": "fallback", "roi_fallback_reason": "mask_empty"}
            )
            return result

        array = np.asarray(image.convert("RGB"))
        if mask.shape != array.shape[:2]:
            mask = cv2.resize(
                mask,
                (array.shape[1], array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        ys, xs = np.where(mask > 0)
        x1 = max(0, int(xs.min()) - self.padding)
        y1 = max(0, int(ys.min()) - self.padding)
        x2 = min(array.shape[1], int(xs.max()) + 1 + self.padding)
        y2 = min(array.shape[0], int(ys.max()) + 1 + self.padding)
        crop = Image.fromarray(array[y1:y2, x1:x2])
        return PreprocessResult(
            crop,
            {
                "preprocessing": "lung_roi",
                "roi_source": self.mask_key,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            },
        )


class RefinedLungROIPreprocessing(LungROIPreprocessing):
    def __init__(self, **kwargs):
        kwargs.setdefault("mask_key", "refined_mask_path")
        super().__init__(**kwargs)


class CompositePreprocessing:
    def __init__(self, strategies: Sequence[PreprocessingStrategy]):
        if not strategies:
            raise ValueError("Composite preprocessing requires at least one strategy")
        self.strategies = list(strategies)

    def __call__(self, image, metadata=None):
        accumulated = dict(metadata or {})
        current = image
        applied = []
        for strategy in self.strategies:
            result = strategy(current, accumulated)
            current = result.image
            accumulated.update(result.metadata)
            applied.append(result.metadata.get("preprocessing", type(strategy).__name__))
        accumulated["preprocessing_pipeline"] = applied
        return PreprocessResult(current, accumulated)


def build_preprocessing(config: Mapping[str, Any] | None) -> PreprocessingStrategy:
    config = dict(config or {"name": "raw"})
    name = str(config.get("name", "raw")).lower()
    if name == "raw":
        return RawPreprocessing()
    if name == "resize":
        return ResizePreprocessing(config.get("size", 224))
    if name in {"legacy", "legacy_classifier"}:
        return LegacyClassifierPreprocessing(config.get("size", 224))
    if name in {"resize_with_padding", "resize_padding"}:
        return ResizeWithPaddingPreprocessing(
            config.get("size", 224),
            config.get("fill", 0),
        )
    if name == "histogram_matching":
        return HistogramMatchingPreprocessing(config["reference_path"])
    if name == "lung_roi":
        return LungROIPreprocessing(
            mask_key=config.get("mask_key", "mask_path"),
            padding=config.get("padding", 0),
            fallback=build_preprocessing(config.get("fallback")),
        )
    if name == "refined_lung_roi":
        return RefinedLungROIPreprocessing(
            padding=config.get("padding", 0),
            fallback=build_preprocessing(config.get("fallback")),
        )
    if name == "composite":
        return CompositePreprocessing(
            [build_preprocessing(item) for item in config.get("strategies", [])]
        )
    raise ValueError(f"Unknown preprocessing strategy: {name}")


def _size_pair(size):
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError("size must be an integer or [height, width]")
    return (int(size[0]), int(size[1]))


def _match_histogram(source, reference):
    source_values, source_inverse, source_counts = np.unique(
        source.ravel(),
        return_inverse=True,
        return_counts=True,
    )
    reference_values, reference_counts = np.unique(
        reference.ravel(),
        return_counts=True,
    )
    source_quantiles = np.cumsum(source_counts).astype(np.float64)
    source_quantiles /= source_quantiles[-1]
    reference_quantiles = np.cumsum(reference_counts).astype(np.float64)
    reference_quantiles /= reference_quantiles[-1]
    mapped = np.interp(source_quantiles, reference_quantiles, reference_values)
    return mapped[source_inverse].reshape(source.shape).astype(np.uint8)
