from pathlib import Path
import tempfile
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np


LABELS = {"lung": 1}


class LungSegmentationInfer:
    """
    Minimal MONAI Label inference task for 2D chest X-ray lung segmentation.

    This task does not train or update the model. It only runs inference and
    returns a binary lung mask to the viewer.
    """

    def __init__(self, model_dir, studies=None, threshold=0.5):
        try:
            import torch
            import segmentation_models_pytorch as smp
            from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
        except ImportError as error:
            raise ImportError(
                "Missing dependency for MONAI Label inference. Install: "
                "monailabel torch segmentation_models_pytorch opencv-python"
            ) from error

        class _InferTask(InferTask):
            def __init__(self, outer):
                super().__init__(
                    type=InferType.SEGMENTATION,
                    labels=LABELS,
                    dimension=2,
                    description="2D chest X-ray lung segmentation",
                    config={"device": ["cuda", "cpu"], "threshold": threshold},
                )
                self.outer = outer

            def is_valid(self):
                return self.outer.checkpoint_path.exists()

            def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
                return self.outer.infer(request)

        self.torch = torch
        self.smp = smp
        self.model_dir = Path(model_dir)
        self.studies = Path(studies) if studies else None
        self.threshold = threshold
        self.checkpoint_path = self.model_dir / "unet_lung_segmentation.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.img_size = self._load_model()
        self.task = _InferTask(self)

    def __getattr__(self, name):
        return getattr(self.task, name)

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = self.torch.load(self.checkpoint_path, map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing required key: model_state_dict")

        encoder = checkpoint.get("encoder", "resnet34")
        img_size = int(checkpoint.get("img_size", 256))

        model = self.smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        state_dict = checkpoint["model_state_dict"]
        state_dict = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        print(f"[lung_monai_app] Loaded model: {self.checkpoint_path}")
        print(f"[lung_monai_app] encoder={encoder}, img_size={img_size}, device={self.device}")
        return model, img_size

    def infer(self, request):
        requested_device = request.get("device")
        if requested_device and str(requested_device).startswith("cpu"):
            self.device = self.torch.device("cpu")
            self.model.to(self.device)

        image_path = self._resolve_image_path(request)
        print(f"[lung_monai_app] Inference image: {image_path}")

        image_array, reference_info = self._read_image(image_path)
        mask_array = self._predict_array(image_array)
        output_path = self._write_mask(mask_array, image_path, reference_info)

        params = {
            "label_names": LABELS,
            "foreground": "lung",
            "background": 0,
            "threshold": self.threshold,
            "source": str(image_path),
        }
        return str(output_path), params

    def _resolve_image_path(self, request):
        image_value = request.get("image_path") or request.get("image")
        if not image_value:
            raise ValueError("Inference request missing image/image_path")

        image_path = Path(str(image_value))
        if image_path.exists():
            return image_path

        if self.studies:
            candidate = self.studies / str(image_value)
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not resolve image path from request: {image_value}")

    def _read_image(self, image_path):
        suffixes = "".join(image_path.suffixes).lower()
        if suffixes.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")):
            return self._read_volume(image_path)

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image, {"kind": "image2d"}

    def _read_volume(self, image_path):
        try:
            import SimpleITK as sitk
        except ImportError as error:
            raise ImportError(
                "Reading NIfTI/NRRD requires SimpleITK. Install: pip install SimpleITK"
            ) from error

        itk_image = sitk.ReadImage(str(image_path))
        array = sitk.GetArrayFromImage(itk_image)
        return array, {"kind": "volume", "itk_image": itk_image}

    def _predict_array(self, image_array):
        if image_array.ndim == 2 or (image_array.ndim == 3 and image_array.shape[-1] in (3, 4)):
            return self._predict_one_slice(image_array)

        if image_array.ndim == 3:
            masks = [self._predict_one_slice(image_array[z]) for z in range(image_array.shape[0])]
            return np.stack(masks, axis=0).astype(np.uint8)

        if image_array.ndim == 4 and image_array.shape[-1] in (3, 4):
            masks = [
                self._predict_one_slice(image_array[z])
                for z in range(image_array.shape[0])
            ]
            return np.stack(masks, axis=0).astype(np.uint8)

        raise ValueError(f"Unsupported image array shape: {image_array.shape}")

    def _predict_one_slice(self, image):
        original_height, original_width = image.shape[:2]

        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        resized = resized.astype(np.float32) / 255.0

        tensor = (
            self.torch.from_numpy(resized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = self.torch.sigmoid(logits)
            mask = (probabilities >= self.threshold).float()

        mask_np = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
        mask_np = cv2.resize(
            mask_np,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return self._clean_mask(mask_np)

    def _clean_mask(self, mask):
        binary = (mask > 0).astype(np.uint8)
        if binary.sum() == 0:
            return binary

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            components.append((label, area))

        components = sorted(components, key=lambda item: item[1], reverse=True)
        keep_labels = [label for label, _ in components[:2]]
        cleaned = np.isin(labels, keep_labels).astype(np.uint8)
        return self._fill_holes(cleaned)

    def _fill_holes(self, mask):
        mask = (mask > 0).astype(np.uint8)
        height, width = mask.shape[:2]
        flood_fill = mask.copy()
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood_fill, flood_mask, (0, 0), 1)
        holes = (flood_fill == 0).astype(np.uint8)
        return np.maximum(mask, holes).astype(np.uint8)

    def _write_mask(self, mask_array, image_path, reference_info):
        output_dir = Path(tempfile.mkdtemp(prefix="lung_monai_label_"))

        if reference_info.get("kind") == "volume":
            return self._write_volume_mask(mask_array, output_dir, reference_info)

        output_path = output_dir / f"{image_path.stem}_lung_mask.nrrd"
        # Slicer displays directly loaded JPG/PNG images with a different
        # vertical axis convention than the NumPy/OpenCV array. Flip only the
        # returned 2D labelmap so the mask overlays the source image correctly.
        mask_array = np.flipud(mask_array)
        mask_volume = (mask_array > 0).astype(np.uint8)[np.newaxis, :, :]

        try:
            import SimpleITK as sitk
        except ImportError as error:
            raise ImportError(
                "Writing MONAI Label masks as NRRD requires SimpleITK. "
                "Install: pip install SimpleITK"
            ) from error

        mask_image = sitk.GetImageFromArray(mask_volume)
        mask_image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(mask_image, str(output_path))
        return output_path

    def _write_volume_mask(self, mask_array, output_dir, reference_info):
        import SimpleITK as sitk

        mask_array = (mask_array > 0).astype(np.uint8)
        if mask_array.ndim == 2:
            mask_array = mask_array[np.newaxis, :, :]

        mask_image = sitk.GetImageFromArray(mask_array)
        reference = reference_info.get("itk_image")
        if reference and list(reference.GetSize()) == list(mask_image.GetSize()):
            mask_image.CopyInformation(reference)

        output_path = output_dir / "lung_mask.nii.gz"
        sitk.WriteImage(mask_image, str(output_path))
        return output_path


def create_lung_segmentation_infer(model_dir, studies=None, threshold=0.5):
    return LungSegmentationInfer(model_dir=model_dir, studies=studies, threshold=threshold).task
