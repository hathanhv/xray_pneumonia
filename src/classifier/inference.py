import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.classifier.dataset import get_eval_transforms
from src.classifier.model import build_mobilenet_v2
from src.lung_segmentation.crop import crop_by_mask


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClassifierInferenceConfig:
    checkpoint_path: Path
    device: str | None = None
    image_size: int = 224
    include_gradcam: bool = True
    gradcam_alpha: float = 0.4
    pad_left: int = 90
    pad_right: int = 90
    pad_top: int = 60
    pad_bottom: int = 8
    max_bottom_ratio: float = 0.75

    def validate(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {self.checkpoint_path}"
            )
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= self.gradcam_alpha <= 1.0:
            raise ValueError("gradcam_alpha must be in [0, 1]")


@dataclass
class ClassifierInferenceResult:
    prediction: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    roi_source: str
    bbox: dict[str, Any] | None
    model_name: str
    checkpoint_path: str
    epoch: int | None
    overlay_base64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "prediction": self.prediction,
            "predicted_index": self.predicted_index,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "roi_source": self.roi_source,
            "bbox": self.bbox,
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "epoch": self.epoch,
        }
        if self.overlay_base64 is not None:
            result["overlay_base64"] = self.overlay_base64
        return result


class _GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def create(self, tensor, class_index, output_size):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        logits[:, class_index].sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(
            heatmap,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        heatmap -= heatmap.min()
        maximum = float(heatmap.max())
        if maximum > 0:
            heatmap /= maximum
        return heatmap

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


class ClassifierInferenceService:
    def __init__(self, config: ClassifierInferenceConfig):
        config.validate()
        self.config = config
        self.device = torch.device(
            config.device
            if config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint = self._load_checkpoint(config.checkpoint_path)
        self.class_to_idx = self.checkpoint.get(
            "class_to_idx",
            {"NORMAL": 0, "PNEUMONIA": 1},
        )
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}
        self.model = build_mobilenet_v2(
            num_classes=len(self.class_to_idx),
            pretrained=False,
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_eval_transforms(config.image_size)

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise TypeError("Classifier checkpoint must be a dictionary")
        if "model_state_dict" not in checkpoint:
            raise KeyError("Classifier checkpoint missing model_state_dict")
        return checkpoint

    def predict_path(
        self,
        image_path: str | Path,
        mask_path: str | Path | None = None,
        include_gradcam: bool | None = None,
    ) -> ClassifierInferenceResult:
        image = Image.open(image_path).convert("RGB")
        mask = self._read_mask(mask_path) if mask_path else None
        return self.predict_image(
            image,
            mask=mask,
            include_gradcam=include_gradcam,
        )

    def predict_bytes(
        self,
        image_bytes: bytes,
        mask_bytes: bytes | None = None,
        include_gradcam: bool | None = None,
    ) -> ClassifierInferenceResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = None
        if mask_bytes:
            mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
            mask = np.asarray(mask_image)
        return self.predict_image(
            image,
            mask=mask,
            include_gradcam=include_gradcam,
        )

    def predict_image(
        self,
        image: Image.Image,
        mask: np.ndarray | None = None,
        include_gradcam: bool | None = None,
    ) -> ClassifierInferenceResult:
        prepared_image, roi_source, bbox = self._prepare_roi(image, mask)
        tensor = self.transform(prepared_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        prediction = self.idx_to_class[predicted_index]
        probability_map = {
            self.idx_to_class[index]: float(probabilities[index].item())
            for index in sorted(self.idx_to_class)
        }

        should_include_gradcam = (
            self.config.include_gradcam
            if include_gradcam is None
            else bool(include_gradcam)
        )
        overlay_base64 = None
        if should_include_gradcam:
            overlay = self._create_gradcam_overlay(
                prepared_image,
                tensor,
                predicted_index,
            )
            overlay_base64 = self._image_to_base64(overlay)

        return ClassifierInferenceResult(
            prediction=prediction,
            predicted_index=predicted_index,
            confidence=float(probabilities[predicted_index].item()),
            probabilities=probability_map,
            roi_source=roi_source,
            bbox=bbox,
            model_name=str(self.checkpoint.get("model_name", "mobilenet_v2")),
            checkpoint_path=str(self.config.checkpoint_path),
            epoch=self.checkpoint.get("epoch"),
            overlay_base64=overlay_base64,
        )

    def _prepare_roi(self, image, mask):
        if mask is None:
            return image, "input_image", None

        image_array = np.asarray(image.convert("RGB"))
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D lung mask, got shape {mask.shape}")
        if mask.shape != image_array.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_array.shape[1], image_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        crop, bbox = crop_by_mask(
            image_array,
            mask,
            pad_left=self.config.pad_left,
            pad_right=self.config.pad_right,
            pad_top=self.config.pad_top,
            pad_bottom=self.config.pad_bottom,
            max_bottom_ratio=self.config.max_bottom_ratio,
        )
        if crop is None:
            raise ValueError("The supplied lung mask is empty or invalid")
        return Image.fromarray(crop), "edited_lung_mask", bbox

    def _read_mask(self, mask_path):
        mask_path = Path(mask_path)
        suffixes = "".join(mask_path.suffixes).lower()
        if suffixes.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")):
            try:
                import SimpleITK as sitk
            except ImportError as error:
                raise ImportError(
                    "Reading a medical-image labelmap requires SimpleITK"
                ) from error
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
            while mask.ndim > 2:
                mask = mask[mask.shape[0] // 2]
            return mask

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read lung mask: {mask_path}")
        return mask

    def _create_gradcam_overlay(self, image, tensor, predicted_index):
        gradcam = _GradCAM(self.model, self.model.features[-1])
        try:
            heatmap = gradcam.create(
                tensor,
                predicted_index,
                (self.config.image_size, self.config.image_size),
            )
        finally:
            gradcam.close()

        heatmap = cv2.resize(
            heatmap,
            image.size,
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap_rgb = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET,
        )
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
        original = np.asarray(image.convert("RGB"))
        overlay = self._blend_heatmap(
            original,
            heatmap_rgb,
            heatmap,
            self.config.gradcam_alpha,
        )
        return Image.fromarray(overlay)

    @staticmethod
    def _blend_heatmap(original, heatmap_rgb, heatmap, maximum_alpha):
        """
        Blend only activated Grad-CAM regions.

        A fixed alpha colors even zero-valued heatmap pixels dark blue, which
        creates a visible rectangular ROI boundary. Intensity-weighted alpha
        leaves inactive pixels unchanged while preserving the trained model's
        original square resize preprocessing.
        """
        activation = np.clip((heatmap.astype(np.float32) - 0.08) / 0.92, 0.0, 1.0)
        alpha = (activation * float(maximum_alpha))[:, :, np.newaxis]
        blended = (
            original.astype(np.float32) * (1.0 - alpha)
            + heatmap_rgb.astype(np.float32) * alpha
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _image_to_base64(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
