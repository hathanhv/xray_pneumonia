from __future__ import annotations

import base64
import csv
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.fusion import SpatialFusionConfig, SpatialFusionEngine


@dataclass(frozen=True)
class CXformerPathologyConfig:
    checkpoint_path: Path
    thresholds_csv: Path
    backbone_name: str = "m42-health/CXformer-small"
    device: Optional[str] = None
    num_threads: int = 2
    top_k: int = 5
    heatmap_threshold: float = 0.60
    min_lesion_pixels: int = 100


class CXformerClassifier(nn.Module):
    """Exact deployment architecture used by cxformer_final_all15000.pt.

    Pooling is stored in the checkpoint metadata:
      0.5 * (CLS + mean_patch), register tokens excluded.
    The checkpoint contains four register tokens and a 14-class Linear head.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_register_tokens: int = 4,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, int(num_labels))
        self.num_register_tokens = int(num_register_tokens)

    def pooled_features(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        cls_feature = last_hidden_state[:, 0]
        patch_start = 1 + self.num_register_tokens
        patch_tokens = last_hidden_state[:, patch_start:]
        if patch_tokens.shape[1] == 0:
            raise RuntimeError("CXFormer output has no patch tokens after special tokens")
        mean_patch = patch_tokens.mean(dim=1)
        return 0.5 * (cls_feature + mean_patch)

    def forward(self, pixel_values, return_tokens: bool = False):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state
        logits = self.classifier(self.pooled_features(last_hidden))
        if return_tokens:
            return logits, last_hidden
        return logits


@dataclass
class CXformerPathologyResult:
    source: str
    model_name: str
    probabilities: Dict[str, float]
    thresholds: Dict[str, float]
    findings: List[Dict]
    elapsed_s: float
    heatmap_method: str
    overlay_base64: Optional[str]

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "model": self.model_name,
            "probabilities": self.probabilities,
            "thresholds": self.thresholds,
            "findings": self.findings,
            "elapsed_s": self.elapsed_s,
            "activation": "sigmoid",
            "heatmap_method": self.heatmap_method,
            "overlay_base64": self.overlay_base64,
        }


class CXformerPathologyService:
    """14-label CXFormer inference + transformer token heatmap + anatomy fusion."""

    def __init__(self, config: CXformerPathologyConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._labels: Optional[List[str]] = None
        self._thresholds: Optional[Dict[str, float]] = None
        self._model_name = config.backbone_name

        self.fusion = SpatialFusionEngine(
            SpatialFusionConfig(
                heatmap_threshold=float(config.heatmap_threshold),
                min_lesion_pixels=int(config.min_lesion_pixels),
            )
        )

    def predict_path(
        self,
        image_path: Path,
        anatomy_mask_path: Optional[Path] = None,
        include_localization: bool = True,
    ) -> CXformerPathologyResult:
        start = time.perf_counter()
        model, processor, device = self._ensure_loaded()

        original = Image.open(image_path).convert("RGB")
        inputs = processor(images=original, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        anatomy_mask = None
        if include_localization and anatomy_mask_path:
            anatomy_mask = self._load_anatomy_mask(Path(anatomy_mask_path))

        with torch.no_grad():
            logits, last_hidden = model(pixel_values, return_tokens=True)
            probabilities_tensor = torch.sigmoid(logits)[0]

        probability_map = {
            label: float(probabilities_tensor[idx].cpu())
            for idx, label in enumerate(self._labels)
        }

        active_indices = [
            idx
            for idx, label in enumerate(self._labels)
            if probability_map[label] >= float(self._thresholds[label])
        ]
        active_indices.sort(key=lambda idx: probability_map[self._labels[idx]], reverse=True)

        heatmap_indices = (
            active_indices[: max(1, int(self.config.top_k))]
            if include_localization
            else []
        )
        heatmaps: Dict[int, np.ndarray] = {}
        for class_idx in heatmap_indices:
            heatmap_model = self._class_specific_patch_attribution(
                model=model,
                last_hidden=last_hidden,
                class_idx=class_idx,
                input_h=int(pixel_values.shape[-2]),
                input_w=int(pixel_values.shape[-1]),
            )
            heatmaps[class_idx] = cv2.resize(
                heatmap_model,
                original.size,
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)

        findings: List[Dict] = []
        for class_idx in active_indices:
            label = self._labels[class_idx]
            item = {
                "finding": label,
                "class_index": int(class_idx),
                "status": "present",
                "probability": probability_map[label],
                "classification_threshold": float(self._thresholds[label]),
                "classification_threshold_source": self.config.thresholds_csv.name,
            }

            if include_localization:
                heatmap = heatmaps.get(class_idx)
                if heatmap is None:
                    item["localization"] = {
                        "type": "cxformer_transformer_token_attribution",
                        "mask_semantics": "estimated_localization_mask",
                        "valid": False,
                        "reason": "heatmap_not_generated_top_k_limit",
                    }
                    item["measurements"] = {
                        "projected_2d_burden_pct": None,
                        "valid": False,
                    }
                elif anatomy_mask is None:
                    item["localization"] = {
                        "type": "cxformer_transformer_token_attribution",
                        "mask_semantics": "estimated_localization_mask",
                        "heatmap_threshold": float(self.config.heatmap_threshold),
                        "valid": False,
                        "reason": "anatomy_mask_not_provided",
                    }
                    item["measurements"] = {
                        "projected_2d_burden_pct": None,
                        "valid": False,
                    }
                else:
                    fused = self.fusion.fuse(
                        finding=label,
                        heatmap=heatmap,
                        anatomy_mask=anatomy_mask,
                    )
                    fused["localization"]["type"] = "cxformer_transformer_token_attribution"
                    item.update(fused)

            findings.append(item)

        overlay_base64 = None
        if include_localization and active_indices and active_indices[0] in heatmaps:
            overlay_base64 = self._overlay_to_base64(original, heatmaps[active_indices[0]])

        return CXformerPathologyResult(
            source=str(image_path),
            model_name=self._model_name,
            probabilities=probability_map,
            thresholds={label: float(self._thresholds[label]) for label in self._labels},
            findings=findings,
            elapsed_s=round(time.perf_counter() - start, 4),
            heatmap_method=(
                "class_specific_transformer_token_attribution"
                if include_localization
                else "disabled_classification_only"
            ),
            overlay_base64=overlay_base64,
        )

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model, self._processor, self._device

        checkpoint_path = Path(self.config.checkpoint_path)
        thresholds_path = Path(self.config.thresholds_csv)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CXFormer checkpoint not found: {checkpoint_path}")
        if not thresholds_path.exists():
            raise FileNotFoundError(f"CXFormer thresholds CSV not found: {thresholds_path}")

        self._configure_threads()
        checkpoint = self._load_checkpoint(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected CXFormer deployment checkpoint to be a dictionary")

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint missing model_state_dict")

        labels = [str(x) for x in checkpoint.get("class_names", [])]
        num_labels = int(checkpoint.get("num_classes", len(labels)))
        if len(labels) != num_labels:
            raise ValueError(
                f"Checkpoint class_names mismatch: labels={len(labels)} num_classes={num_labels}"
            )

        thresholds = self._read_thresholds(thresholds_path)
        missing = [label for label in labels if label not in thresholds]
        if missing:
            raise ValueError("Missing thresholds for: " + ", ".join(missing))

        model_name = str(checkpoint.get("cxformer_source") or self.config.backbone_name)
        num_register_tokens = self._infer_register_token_count(state_dict)
        model = CXformerClassifier(
            model_name=model_name,
            num_labels=num_labels,
            num_register_tokens=num_register_tokens,
        )
        model.load_state_dict(state_dict, strict=True)

        hidden_size = int(checkpoint.get("hidden_size", model.backbone.config.hidden_size))
        if hidden_size != int(model.backbone.config.hidden_size):
            raise ValueError(
                f"Hidden size mismatch: checkpoint={hidden_size}, backbone={model.backbone.config.hidden_size}"
            )

        device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = model.to(device).eval()
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self._model = model
        self._processor = processor
        self._device = device
        self._labels = labels
        self._thresholds = {label: float(thresholds[label]) for label in labels}
        self._model_name = model_name
        return model, processor, device

    @staticmethod
    def _class_specific_patch_attribution(
        model: CXformerClassifier,
        last_hidden: torch.Tensor,
        class_idx: int,
        input_h: int,
        input_w: int,
    ) -> np.ndarray:
        """Exact positive patch contribution to the linear class logit.

        Because deployment pooling is 0.5 * (CLS + mean_patch), each patch token
        contributes linearly through the trained classifier weight. This creates a
        class-specific transformer-token evidence map without adding another model.
        """
        patch_start = 1 + model.num_register_tokens
        patch_tokens = last_hidden[0, patch_start:]
        weight = model.classifier.weight[class_idx]
        scores = 0.5 * torch.einsum("nd,d->n", patch_tokens, weight)
        scores = torch.relu(scores)

        grid_h, grid_w = CXformerPathologyService._infer_patch_grid(
            patch_count=int(scores.numel()),
            input_h=input_h,
            input_w=input_w,
            patch_size=getattr(model.backbone.config, "patch_size", 14),
        )
        expected = grid_h * grid_w
        if expected != int(scores.numel()):
            raise RuntimeError(
                f"Patch-grid mismatch: tokens={scores.numel()} grid={grid_h}x{grid_w}"
            )

        heatmap = scores.reshape(grid_h, grid_w)
        min_v = heatmap.min()
        max_v = heatmap.max()
        heatmap = (heatmap - min_v) / (max_v - min_v + 1e-8)
        heatmap = torch.nn.functional.interpolate(
            heatmap[None, None],
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return heatmap.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _infer_patch_grid(
        patch_count: int,
        input_h: int,
        input_w: int,
        patch_size,
    ) -> Tuple[int, int]:
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = int(patch_size[0]), int(patch_size[-1])
        else:
            patch_h = patch_w = int(patch_size or 14)
        grid_h = input_h // patch_h
        grid_w = input_w // patch_w
        if grid_h * grid_w == patch_count:
            return grid_h, grid_w

        # Fallback preserves the image aspect ratio when processor/model metadata differs.
        ratio = max(1e-6, float(input_h) / float(input_w))
        grid_h = max(1, int(round(math.sqrt(patch_count * ratio))))
        while grid_h > 1 and patch_count % grid_h != 0:
            grid_h -= 1
        grid_w = patch_count // grid_h
        return grid_h, grid_w

    @staticmethod
    def _infer_register_token_count(state_dict: Dict[str, torch.Tensor]) -> int:
        value = state_dict.get("backbone.embeddings.register_tokens")
        if value is None:
            return 0
        if value.ndim != 3:
            raise ValueError(
                "Unexpected register token tensor shape: " + str(tuple(value.shape))
            )
        return int(value.shape[1])

    @staticmethod
    def _read_thresholds(path: Path) -> Dict[str, float]:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [str(x).strip() for x in (reader.fieldnames or [])]
            lower = {name.lower(): name for name in fieldnames}
            label_col = lower.get("class_name") or lower.get("label") or lower.get("finding")
            threshold_col = (
                lower.get("threshold_for_future_model")
                or lower.get("threshold")
                or lower.get("classification_threshold")
            )
            if not label_col or not threshold_col:
                raise ValueError(
                    "Threshold CSV must contain class_name and threshold_for_future_model; "
                    f"found {fieldnames}"
                )
            result = {}
            for row in reader:
                label = str(row.get(label_col, "")).strip()
                if label:
                    result[label] = float(row[threshold_col])
        if not result:
            raise ValueError(f"No thresholds found in {path}")
        return result

    @staticmethod
    def _load_checkpoint(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _load_anatomy_mask(path: Path) -> np.ndarray:
        suffix = "".join(path.suffixes).lower()
        if suffix.endswith(".nrrd") or suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
            try:
                import SimpleITK as sitk
            except ImportError as error:
                raise ImportError("SimpleITK is required to read anatomy NRRD/NIfTI masks") from error
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            arr = np.squeeze(arr).astype(np.uint8)
            # anatomy_infer writes NRRD vertically flipped for Slicer display.
            # Flip back to source-image pixel coordinates before spatial fusion.
            return np.flipud(arr).copy()

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read anatomy mask: {path}")
        return mask.astype(np.uint8)

    @staticmethod
    def _overlay_to_base64(original: Image.Image, heatmap: np.ndarray) -> str:
        base = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
        heatmap = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
        color = np.zeros_like(base)
        color[..., 0] = heatmap
        color[..., 1] = 0.45 * heatmap
        alpha = (0.58 * np.sqrt(heatmap))[..., None]
        blended = base * (1.0 - alpha) + color * alpha
        image = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _configure_threads(self):
        num_threads = max(1, int(self.config.num_threads or 1))
        try:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            torch.set_num_threads(num_threads)
