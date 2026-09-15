import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


PATHOLOGY_NAMES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class ScalePatchNet(nn.Module):
    """MedicalPatchNet ScalePatchNet architecture compatible with official weights."""

    def __init__(self, patch_size: int, out_features: int) -> None:
        super().__init__()
        import torchvision

        self.baseBackbone = torchvision.models.efficientnet_v2_s(weights=None)
        self.baseBackbone.features[0][0] = nn.Conv2d(
            1,
            24,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.baseBackbone.classifier[1] = nn.Linear(
            in_features=1280,
            out_features=out_features,
            bias=True,
        )
        self.patchSize = patch_size

    def forward(self, x):
        patch_logits = self.forwardRawPatches(x)
        return torch.mean(patch_logits, dim=1)

    def forwardRawPatches(self, x):
        batch_size = x.size()[0]
        x = x.unfold(2, self.patchSize, self.patchSize).unfold(
            3,
            self.patchSize,
            self.patchSize,
        )
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(
            -1,
            1,
            self.patchSize,
            self.patchSize,
        )
        x = self.baseBackbone(x)
        patch_count = int(x.size()[0] / batch_size)
        return torch.stack(torch.split(x, patch_count))

    def forwardScaledPatches(self, x):
        raw_patch_logits = self.forwardRawPatches(x)
        global_logits = torch.mean(raw_patch_logits, dim=1)
        global_prob = torch.sigmoid(global_logits).unsqueeze(1).unsqueeze(2)
        return raw_patch_logits * global_prob


@dataclass(frozen=True)
class MedicalPatchNetConfig:
    checkpoint_path: Optional[Path] = None
    hf_repo_id: str = "patrick-w/MedicalPatchNet"
    hf_filename: str = "MedicalPatchNet_weights.pt"
    device: Optional[str] = None
    num_threads: int = 2
    image_size: int = 512
    patch_size: int = 64
    shift_pixels: int = 64
    shift_batch_size: int = 4
    use_scaled_patch_maps: bool = True
    probability_threshold: float = 0.50
    mask_logit_threshold: float = 1.0
    top_k: int = 5
    include_overlay: bool = True
    flip_display_vertical: bool = False


@dataclass(frozen=True)
class MedicalPatchNetResult:
    source: str
    original_size: Tuple[int, int]
    crop_box: Tuple[int, int, int, int]
    probabilities: Dict[str, float]
    findings: List[Dict]
    elapsed_s: float
    shift_pixels: int
    forward_grid_count: int
    probability_threshold: float
    roi_source: str = "input_image"
    lung_label_source: Optional[str] = None
    overlay_base64: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "original_size": list(self.original_size),
            "crop_box": {
                "left": self.crop_box[0],
                "top": self.crop_box[1],
                "right": self.crop_box[2],
                "bottom": self.crop_box[3],
            },
            "probabilities": self.probabilities,
            "findings": self.findings,
            "elapsed_s": self.elapsed_s,
            "shift_pixels": self.shift_pixels,
            "forward_grid_count": self.forward_grid_count,
            "probability_threshold": self.probability_threshold,
            "roi_source": self.roi_source,
            "lung_label_source": self.lung_label_source,
            "overlay_base64": self.overlay_base64,
            "model": "patrick-w/MedicalPatchNet",
            "localization_method": "patch_based_self_explainable_map",
        }


class MedicalPatchNetService:
    def __init__(self, config: MedicalPatchNetConfig):
        self.config = config
        self._validate_config()
        self._model = None
        self._device = None

    def predict_path(
        self,
        image_path: Path,
        mask_path: Optional[Path] = None,
        mask_array: Optional[np.ndarray] = None,
    ) -> MedicalPatchNetResult:
        image_path = Path(image_path)
        mask_path = Path(mask_path) if mask_path else None
        start = time.perf_counter()
        model, device = self._get_model()
        image_orig, crop_box, input_tensor, roi_source = self._load_and_preprocess(
            image_path,
            mask_path=mask_path,
            mask_array=mask_array,
        )
        input_tensor = input_tensor.to(device)

        with torch.inference_mode():
            global_logits = model(input_tensor)
            probabilities = torch.sigmoid(global_logits)[0].detach().cpu().numpy()
            patch_maps = self._shifted_patch_logit_maps(model, input_tensor)

        maps_np = patch_maps.detach().cpu().numpy()
        findings = self._build_findings(probabilities, maps_np, image_orig.size, crop_box)
        overlay_base64 = None
        # Keep the original MedicalPatchNet Yellow-panel behavior: always render
        # the top-k patch-evidence maps when overlay display is enabled, even if
        # no class crosses the reporting threshold. The findings list still uses
        # probability_threshold for PRESENT/ABSENT semantics; this overlay is
        # visualization only.
        if self.config.include_overlay:
            overlay_base64 = self._paper_style_overlay_to_base64(
                image_orig,
                probabilities,
                maps_np,
                crop_box,
            )

        elapsed_s = round(time.perf_counter() - start, 4)
        return MedicalPatchNetResult(
            source=str(image_path),
            original_size=image_orig.size,
            crop_box=crop_box,
            probabilities={
                name: float(probabilities[idx])
                for idx, name in enumerate(PATHOLOGY_NAMES)
            },
            findings=findings,
            elapsed_s=elapsed_s,
            shift_pixels=self.config.shift_pixels,
            forward_grid_count=(self.config.patch_size // self.config.shift_pixels) ** 2,
            probability_threshold=float(self.config.probability_threshold),
            roi_source=roi_source,
            lung_label_source=(
                str(mask_path)
                if mask_path
                else "auto_lung_segmentation"
                if mask_array is not None
                else None
            ),
            overlay_base64=overlay_base64,
        )

    def _get_model(self):
        if self._model is not None:
            return self._model, self._device

        self._configure_threads()
        device = torch.device(
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint_path = self._resolve_checkpoint()
        model = ScalePatchNet(
            patch_size=self.config.patch_size,
            out_features=len(PATHOLOGY_NAMES),
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            key.replace("_orig_mod.", "").replace("module.", ""): value
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "MedicalPatchNet checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self._model = model.to(device).eval()
        self._device = device
        return self._model, self._device

    def _validate_config(self):
        if self.config.image_size <= 0 or self.config.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.config.shift_pixels <= 0:
            raise ValueError("shift_pixels must be positive")
        if self.config.image_size % self.config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.config.patch_size % self.config.shift_pixels != 0:
            raise ValueError("patch_size must be divisible by shift_pixels")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")

    def _resolve_checkpoint(self) -> Path:
        if self.config.checkpoint_path and self.config.checkpoint_path.exists():
            return self.config.checkpoint_path
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ImportError(
                "MedicalPatchNet requires huggingface_hub or a local checkpoint_path."
            ) from error
        return Path(
            hf_hub_download(
                repo_id=self.config.hf_repo_id,
                filename=self.config.hf_filename,
            )
        )

    def _configure_threads(self):
        num_threads = max(1, int(self.config.num_threads or 1))
        try:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            torch.set_num_threads(num_threads)

    def _load_and_preprocess(
        self,
        image_path: Path,
        mask_path: Optional[Path] = None,
        mask_array: Optional[np.ndarray] = None,
    ):
        import torchvision.transforms.functional as TF

        image_orig = Image.open(image_path).convert("L")
        roi_source = "input_image"
        if mask_array is not None:
            image_orig = self._apply_lung_mask(image_orig, mask_array)
            roi_source = "lung_segmented_image"
        elif mask_path:
            mask = self._read_lung_mask(mask_path)
            image_orig = self._apply_lung_mask(image_orig, mask)
            roi_source = "lung_segmented_image"

        image_crop, crop_box = self._center_square_crop(image_orig)
        tensor = TF.to_tensor(image_crop)
        tensor = TF.resize(
            tensor,
            [self.config.image_size, self.config.image_size],
            antialias=True,
        )
        return image_orig, crop_box, tensor.unsqueeze(0), roi_source

    @staticmethod
    def _read_lung_mask(mask_path: Path) -> np.ndarray:
        suffixes = "".join(mask_path.suffixes).lower()
        if suffixes.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")):
            try:
                import SimpleITK as sitk
            except ImportError as error:
                raise ImportError(
                    "Reading a MONAI Label lung mask requires SimpleITK"
                ) from error
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
            mask = np.squeeze(mask)
            while mask.ndim > 2:
                mask = mask[mask.shape[0] // 2]
            # lung_infer writes 2-D NRRD labelmaps flipped vertically for Slicer
            # display. Convert them back to source-image pixel coordinates before
            # masking the MedicalPatchNet input.
            return np.flipud(mask).astype(np.uint8)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read lung mask: {mask_path}")
        return mask.astype(np.uint8)

    @staticmethod
    def _apply_lung_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
        image_array = np.asarray(image.convert("L"), dtype=np.uint8)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D lung mask, got shape {mask.shape}")
        if mask.shape != image_array.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_array.shape[1], image_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        lung = mask > 0
        if not lung.any():
            raise ValueError("The supplied lung mask is empty or invalid")

        masked = image_array.copy()
        masked[~lung] = 0
        return Image.fromarray(masked, mode="L")

    @staticmethod
    def _center_square_crop(img: Image.Image):
        width, height = img.size
        crop_len = min(width, height)
        left = (width - crop_len) // 2
        top = (height - crop_len) // 2
        return img.crop((left, top, left + crop_len, top + crop_len)), (
            left,
            top,
            left + crop_len,
            top + crop_len,
        )

    def _shifted_patch_logit_maps(self, model, img):
        shifts = [
            (dx, dy)
            for dy in range(0, self.config.patch_size, self.config.shift_pixels)
            for dx in range(0, self.config.patch_size, self.config.shift_pixels)
        ]
        total = None
        counts = None
        fill_value = float(img.min().detach().cpu())

        for start in range(0, len(shifts), self.config.shift_batch_size):
            batch_shifts = shifts[start : start + self.config.shift_batch_size]
            shifted_imgs = torch.cat(
                [
                    self._shift_tensor_2d(img, dx, dy, fill=fill_value)
                    for dx, dy in batch_shifts
                ],
                dim=0,
            )
            if self.config.use_scaled_patch_maps:
                patch_features = model.forwardScaledPatches(shifted_imgs)
            else:
                patch_features = model.forwardRawPatches(shifted_imgs)
            batch_maps = self._patch_features_to_grid_maps(patch_features)

            for idx, (dx, dy) in enumerate(batch_shifts):
                unshifted = self._shift_tensor_2d(batch_maps[idx : idx + 1], -dx, -dy)[0]
                valid = self._shift_tensor_2d(
                    torch.ones_like(img[:, :1]),
                    dx,
                    dy,
                )
                valid = self._shift_tensor_2d(valid, -dx, -dy)[0, 0]
                if total is None:
                    total = torch.zeros_like(unshifted)
                    counts = torch.zeros_like(valid)
                total += unshifted * valid.unsqueeze(0)
                counts += valid
        return total / counts.clamp_min(1).unsqueeze(0)

    @staticmethod
    def _shift_tensor_2d(x, dx, dy, fill=0.0):
        batch, channels, height, width = x.shape
        out = x.new_full((batch, channels, height, width), fill)
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx) if dx >= 0 else width
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy) if dy >= 0 else height
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        if src_x1 > src_x0 and src_y1 > src_y0:
            out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[
                :,
                :,
                src_y0:src_y1,
                src_x0:src_x1,
            ]
        return out

    def _patch_features_to_grid_maps(self, patch_features):
        grid = self.config.image_size // self.config.patch_size
        maps = patch_features.reshape(patch_features.shape[0], grid, grid, -1)
        maps = maps.permute(0, 3, 1, 2)
        return F.interpolate(
            maps,
            size=(self.config.image_size, self.config.image_size),
            mode="nearest",
        )

    def _build_findings(self, probabilities, maps_np, original_size, crop_box):
        ranked = np.argsort(probabilities)[::-1]
        findings = []
        for class_idx in ranked:
            probability = float(probabilities[class_idx])
            pathology = PATHOLOGY_NAMES[int(class_idx)]
            if probability < self.config.probability_threshold:
                break
            if pathology == "No Finding":
                continue
            signed_map = maps_np[int(class_idx)]
            mask_crop = signed_map > self.config.mask_logit_threshold
            findings.append(
                {
                    "finding": pathology,
                    "class_index": int(class_idx),
                    "status": "present",
                    "probability": probability,
                    "localization": {
                        "type": "patch_based_localization",
                        "mask_semantics": "estimated_localization_mask",
                        "threshold": float(self.config.mask_logit_threshold),
                        "positive_pixels_512": int(mask_crop.sum()),
                    },
                }
            )
        return findings

    @staticmethod
    def _paste_crop_map_to_original(map_crop, original_size, crop_box):
        orig_w, orig_h = original_size
        left, top, right, bottom = crop_box
        resized = cv2.resize(
            map_crop.astype(np.float32),
            (right - left, bottom - top),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        canvas[top:bottom, left:right] = resized
        return canvas

    @staticmethod
    def _overlay_to_base64(gray_img_pil, signed_map, clip_value=20.0, alpha=0.58):
        gray = np.array(gray_img_pil.convert("L"))
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        signed = np.clip(signed_map / clip_value, -1.0, 1.0)
        positive = np.clip(signed, 0.0, 1.0)
        negative = np.clip(-signed, 0.0, 1.0)
        color = np.zeros_like(base)
        color[..., 0] = positive
        color[..., 1] = 0.15 * (positive + negative)
        color[..., 2] = negative
        strength = np.clip(np.abs(signed), 0.0, 1.0)
        local_alpha = alpha * np.sqrt(strength)[..., None]
        blended = base * (1.0 - local_alpha) + color * local_alpha
        image = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _paper_style_overlay_to_base64(
        self,
        gray_img_pil,
        probabilities,
        maps_np,
        crop_box,
        clip_value=20.0,
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        # The montage is a visualization, while probabilities and patch maps were
        # already computed at the model's native 512 px resolution.  Building five
        # RGBA overlays at the original DICOM resolution (often 3k x 3k) can require
        # several hundred MB per panel.  Render the montage on a bounded canvas and
        # keep the original model outputs unchanged.
        source_w, source_h = gray_img_pil.size
        display_image = gray_img_pil.convert("L")
        display_crop_box = tuple(int(value) for value in crop_box)
        overlay_max_side = 768
        if max(source_w, source_h) > overlay_max_side:
            scale = float(overlay_max_side) / float(max(source_w, source_h))
            display_w = max(1, int(round(source_w * scale)))
            display_h = max(1, int(round(source_h * scale)))
            display_image = display_image.resize(
                (display_w, display_h),
                resample=Image.Resampling.BILINEAR,
            )
            left, top, right, bottom = crop_box
            display_crop_box = (
                max(0, min(display_w - 1, int(round(left * scale)))),
                max(0, min(display_h - 1, int(round(top * scale)))),
                max(1, min(display_w, int(round(right * scale)))),
                max(1, min(display_h, int(round(bottom * scale)))),
            )

        top_indices = np.argsort(probabilities)[::-1][: self.config.top_k]
        valid_mask = self._paste_crop_map_to_original(
            np.ones((self.config.image_size, self.config.image_size), dtype=np.float32),
            display_image.size,
            display_crop_box,
        ) > 0

        fig, axes = plt.subplots(
            1,
            len(top_indices),
            figsize=(4.0 * len(top_indices), 2.9),
            constrained_layout=True,
        )
        if len(top_indices) == 1:
            axes = [axes]

        for ax, idx in zip(axes, top_indices):
            pathology = PATHOLOGY_NAMES[int(idx)]
            signed_original = self._paste_crop_map_to_original(
                maps_np[int(idx)],
                display_image.size,
                display_crop_box,
            )
            overlay = self._signed_overlay_array(
                display_image,
                signed_original,
                clip_value=clip_value,
                valid_mask=valid_mask,
            )
            mask, contour_threshold = self._positive_evidence_mask(
                signed_original,
                valid_mask,
            )
            if self.config.flip_display_vertical:
                overlay = np.flipud(overlay)
                mask = np.flipud(mask)
            ax.imshow(overlay)
            if mask.any():
                ax.contour(
                    mask.astype(np.uint8),
                    levels=[0.5],
                    colors=["#ffe66d"],
                    linewidths=1.2,
                )
            ax.axis("off")
            ax.set_title(
                f"{pathology}\np={probabilities[int(idx)]:.3f} | t={contour_threshold:.2f}",
                fontsize=10,
                pad=5,
            )

        norm = mcolors.TwoSlopeNorm(vmin=-clip_value, vcenter=0, vmax=clip_value)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.024, pad=0.012)
        cbar.set_label("Patch evidence logit")
        cbar.set_ticks([-clip_value, 0, clip_value])
        cbar.set_ticklabels(["against", "0", "supports"])

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=150,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _positive_evidence_mask(self, signed_map, valid_mask):
        threshold = float(self.config.mask_logit_threshold)
        mask = (signed_map >= threshold) & valid_mask
        if mask.any():
            return mask, threshold

        valid_values = signed_map[valid_mask]
        positive_values = valid_values[valid_values > 0]
        if positive_values.size == 0:
            return mask, threshold

        threshold = float(np.percentile(positive_values, 75))
        mask = (signed_map >= threshold) & valid_mask
        return mask, threshold

    @staticmethod
    def _signed_overlay_array(
        gray_img_pil,
        signed_map,
        clip_value=20.0,
        alpha=0.58,
        valid_mask=None,
    ):
        gray = np.asarray(gray_img_pil.convert("L"), dtype=np.uint8)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        signed = np.asarray(signed_map, dtype=np.float32) / np.float32(clip_value)
        signed = np.clip(signed, -1.0, 1.0)
        try:
            import matplotlib.pyplot as plt

            # Build only a tiny 256-color LUT.  Calling the colormap directly on
            # a 3k image creates a float64 RGBA array (~288 MiB at 3072 x 3072).
            lut = plt.get_cmap("RdBu_r")(
                np.linspace(0.0, 1.0, 256, dtype=np.float32),
                bytes=True,
            )[:, :3]
            color_indices = np.rint((signed + 1.0) * 127.5).astype(np.uint8)
            color = lut[color_indices].astype(np.float32) / 255.0
        except Exception:
            positive = np.clip(signed, 0.0, 1.0)
            negative = np.clip(-signed, 0.0, 1.0)
            color = np.zeros_like(base)
            color[..., 0] = positive
            color[..., 1] = 0.15 * (positive + negative)
            color[..., 2] = negative
        strength = np.clip(np.abs(signed), 0.0, 1.0)
        local_alpha = alpha * np.sqrt(strength)[..., None]
        if valid_mask is not None:
            local_alpha *= np.asarray(valid_mask, dtype=np.float32)[..., None]
        blended = base * (1.0 - local_alpha) + color * local_alpha
        # Matplotlib retains every panel until savefig(); uint8 uses one quarter
        # of the memory of float32 for those retained images.
        return np.rint(np.clip(blended, 0, 1) * 255.0).astype(np.uint8)
