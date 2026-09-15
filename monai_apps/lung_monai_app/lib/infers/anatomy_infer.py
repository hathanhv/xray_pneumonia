"""
MONAI Label inference adapter for chest X-ray anatomy segmentation.

Uses ianpan/chest-x-ray-basic (HuggingFace) to segment:
  label 1 = right_lung
  label 2 = left_lung
  label 3 = heart

Returns a multi-label NRRD mask + JSON params with per-label confidence and CTR.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)
DEBUG_LOG_PATH = Path(__file__).resolve().parents[2] / "anatomy_infer_debug.log"

ANATOMY_LABELS = {
    "right_lung": 1,
    "left_lung": 2,
    "heart": 3,
}

# ianpan/chest-x-ray-basic label indices (as returned by the model)
_MODEL_RIGHT_LUNG = 1
_MODEL_LEFT_LUNG = 2
_MODEL_HEART = 3


class AnatomySegmentationInfer:
    """
    MONAI Label inference task for chest X-ray anatomy segmentation.

    Loads ianpan/chest-x-ray-basic on first call (lazy, cached on instance).
    Input:  JPG / PNG chest X-ray (2-D, any size).
    Output: Multi-label NRRD mask (uint8, shape 1×H×W) + JSON params.
    """

    MODEL_ID = "ianpan/chest-x-ray-basic"

    def __init__(self, studies=None, device=None, num_threads=2):
        try:
            from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
        except ImportError as error:
            raise ImportError(
                "Missing monailabel dependency for anatomy segmentation."
            ) from error

        self.studies = Path(studies) if studies else None
        self._device_override = device
        self._num_threads = self._parse_num_threads(num_threads)
        self._model = None  # lazy-loaded

        outer = self

        class _InferTask(InferTask):
            def __init__(self):
                super().__init__(
                    type=InferType.SEGMENTATION,
                    labels=ANATOMY_LABELS,
                    dimension=2,
                    description=(
                        "Chest X-ray anatomy segmentation: "
                        "left lung, right lung, heart"
                    ),
                    config={
                        "device": ["cuda", "cpu"],
                        "num_threads": outer._num_threads,
                    },
                )
                self.outer = outer

            def is_valid(self):
                # Model is downloaded on demand from HuggingFace — always valid.
                return True

            def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
                return self.outer.infer(request)

        self.task = _InferTask()

    def __getattr__(self, name):
        return getattr(self.task, name)

    # ------------------------------------------------------------------
    # Public inference entry-point
    # ------------------------------------------------------------------

    def infer(self, request):
        image_path = self._resolve_image_path(request)
        self._debug_log("image", path=image_path)
        request_params = self._parse_request_params(request)
        self._debug_log("request params", params=request_params)

        image_bgr = self._read_image_bgr(image_path)
        mask, softmax = self._predict(image_bgr)
        confidence = self._per_label_confidence(softmax)
        ctr = self._estimate_ctr(mask, image_bgr.shape[:2])

        params = {
            "label_names": ANATOMY_LABELS,
            "source": str(image_path),
            "confidence": confidence,
            "ctr": ctr,
            "ctr_valid": ctr is not None,
            "input_shape": list(image_bgr.shape[:2]),
        }
        params.update(request_params)
        output_path = self._write_mask(mask, image_path, params)
        return str(output_path), params

    @staticmethod
    def _parse_request_params(request):
        value = request.get("params")
        if not value:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _debug_log(message, **fields):
        field_text = " | ".join(
            f"{key}={value}"
            for key, value in fields.items()
        )
        line = f"[anatomy_infer] {message}"
        if field_text:
            line = f"{line} | {field_text}"
        logger.info(line)
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_tied_weights_keys():
        """
        Compatibility shim for ianpan/chest-x-ray-basic with transformers 5.x.

        The model's custom code references `all_tied_weights_keys` (old API).
        In transformers 5.x the framework calls `.keys()` on this attribute,
        so it must be a dict, not a list. We expose an empty dict which
        satisfies both usages:
          - `self.all_tied_weights_keys.keys()` (set-like iteration)
          - `missing_keys - self.all_tied_weights_keys.keys()`
        """
        try:
            from transformers import modeling_utils

            if not hasattr(
                modeling_utils.PreTrainedModel, "all_tied_weights_keys"
            ):
                # transformers 5.x assigns to ``self.all_tied_weights_keys`` during
                # post_init().  A read-only property breaks that assignment with
                # ``AttributeError: can't set attribute``.  Provide a writable
                # compatibility property instead, while keeping the dict-style
                # interface expected by the ianpan remote model code.
                def _get_all_tied_weights_keys(instance):
                    return instance.__dict__.get(
                        "_chest_analyzer_all_tied_weights_keys", {}
                    )

                def _set_all_tied_weights_keys(instance, value):
                    instance.__dict__[
                        "_chest_analyzer_all_tied_weights_keys"
                    ] = value

                modeling_utils.PreTrainedModel.all_tied_weights_keys = property(
                    _get_all_tied_weights_keys,
                    _set_all_tied_weights_keys,
                )
                logger.debug(
                    "[anatomy_infer] Applied writable all_tied_weights_keys patch"
                )
        except Exception as patch_err:
            logger.warning(
                "[anatomy_infer] Could not apply tied-weights patch: %s",
                patch_err,
            )


    def _get_model(self):
        if self._model is not None:
            return self._model

        import torch
        from transformers import AutoModel

        # Must patch before from_pretrained loads the custom modeling code
        self._patch_tied_weights_keys()
        self._configure_torch_threads(torch)

        logger.info("[anatomy_infer] Loading model %s …", self.MODEL_ID)
        model = AutoModel.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        device = self._resolve_device()
        model = model.eval().to(device)
        self._model = (model, device)
        logger.info("[anatomy_infer] Model loaded on %s", device)
        return self._model

    def _resolve_device(self):
        import torch

        if self._device_override:
            return torch.device(self._device_override)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _parse_num_threads(value):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 2
        return max(1, parsed)

    def _configure_torch_threads(self, torch):
        if self._num_threads <= 0:
            return
        try:
            torch.set_num_threads(self._num_threads)
            torch.set_num_interop_threads(1)
            self._debug_log("torch cpu threads limited", num_threads=self._num_threads)
        except RuntimeError:
            torch.set_num_threads(self._num_threads)
            self._debug_log("torch cpu threads limited", num_threads=self._num_threads)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _predict(self, image_bgr):
        """
        Run anatomy segmentation on a BGR uint8 image.

        ianpan/chest-x-ray-basic expects single-channel (grayscale) input —
        its first conv layer has weight shape [24, 1, 3, 3].

        Returns
        -------
        mask    : np.ndarray  shape (H, W)  uint8  values in {0,1,2,3}
        softmax : np.ndarray  shape (C, H, W)  float32
        """
        import torch

        model, device = self._get_model()
        h, w = image_bgr.shape[:2]
        self._debug_log("input image shape", h=h, w=w)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if not hasattr(model, "preprocess"):
            raise RuntimeError(
                "Anatomy model does not expose preprocess(); "
                "cannot prepare ianpan/chest-x-ray-basic input safely."
            )

        pixel_values = model.preprocess(gray)
        self._debug_log(
            "preprocessed shape",
            shape=getattr(pixel_values, "shape", None),
        )
        pixel_values = (
            torch.from_numpy(pixel_values)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        with torch.inference_mode():
            output = model(pixel_values)

        if not isinstance(output, dict) or "mask" not in output:
            raise RuntimeError(
                "Unexpected anatomy model output; expected a dict with key 'mask'."
            )
        logits = output["mask"]

        logger.debug(
            "[anatomy_infer] logits shape: %s dtype: %s",
            tuple(logits.shape),
            logits.dtype,
        )
        self._debug_log(
            "raw logits shape",
            shape=tuple(logits.shape),
            dtype=logits.dtype,
        )

        # Convert (1, H', W', C) to (1, C, H', W') if the remote model ever
        # returns channels-last. The common model output is already channels-first.
        if (
            logits.ndim == 4
            and logits.shape[1] not in (3, 4)
            and logits.shape[-1] in (3, 4)
        ):
            logits = logits.permute(0, 3, 1, 2)
            self._debug_log(
                "permuted logits to channels-first",
                shape=tuple(logits.shape),
            )

        softmax = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
        self._debug_log("softmax shape", shape=softmax.shape)

        # Keep confidence computation at model resolution. Upsampling the full
        # CxHxW softmax to a large Slicer image can allocate many GB of RAM.
        mask = np.argmax(softmax, axis=0).astype(np.uint8)
        if softmax.shape[0] == 3 and mask.max() <= 2:
            mask = mask + 1
        self._debug_log("model mask labels/counts", counts=self._label_counts(mask))

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        self._debug_log(
            "output mask shape labels/counts",
            shape=mask.shape,
            counts=self._label_counts(mask),
        )

        return mask, softmax


    def _per_label_confidence(self, softmax):
        """
        Average softmax score inside each predicted region, per label.
        Returns dict {label_name: float}.
        """
        confidence = {}
        mask = np.argmax(softmax, axis=0)
        if softmax.shape[0] == 3 and mask.max() <= 2:
            mask = mask + 1
        for name, idx in ANATOMY_LABELS.items():
            region = mask == idx
            channel_idx = idx if softmax.shape[0] > 3 else idx - 1
            if region.any() and 0 <= channel_idx < softmax.shape[0]:
                confidence[name] = float(softmax[channel_idx][region].mean())
            else:
                confidence[name] = 0.0
        self._debug_log("confidence", confidence=confidence)
        return confidence

    @staticmethod
    def _label_counts(mask):
        values, counts = np.unique(mask, return_counts=True)
        return {
            int(value): int(count)
            for value, count in zip(values.tolist(), counts.tolist())
        }

    def _estimate_ctr(self, mask, shape):
        """
        Cardiothoracic ratio from anatomy mask.
        Returns float or None if anatomy is incomplete.
        """
        h, w = shape
        right_lung = mask == _MODEL_RIGHT_LUNG
        left_lung = mask == _MODEL_LEFT_LUNG
        heart = mask == _MODEL_HEART

        if not right_lung.any() or not left_lung.any() or not heart.any():
            return None

        # Thoracic width: rightmost pixel of right lung → leftmost pixel of left lung
        rl_cols = np.where(right_lung.any(axis=0))[0]
        ll_cols = np.where(left_lung.any(axis=0))[0]
        thoracic_width = float(ll_cols.max() - rl_cols.min())

        if thoracic_width <= 0:
            return None

        # Heart width: leftmost → rightmost heart pixel
        heart_cols = np.where(heart.any(axis=0))[0]
        heart_width = float(heart_cols.max() - heart_cols.min())

        ctr = heart_width / thoracic_width
        logger.debug(
            "[anatomy_infer] CTR=%.3f  heart_w=%.1f  thoracic_w=%.1f",
            ctr,
            heart_width,
            thoracic_width,
        )
        return round(ctr, 4)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _resolve_image_path(self, request):
        value = request.get("image_path") or request.get("image")
        if not value:
            raise ValueError("Anatomy inference request missing image/image_path")

        path = Path(str(value))
        if path.exists():
            return path

        if self.studies:
            candidate = self.studies / str(value)
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not resolve image path: {value}")

    def _read_image_bgr(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image

    def _write_mask(self, mask, image_path, params=None):
        """
        Write multi-label mask as NRRD (uint8, shape 1×H×W).
        Flip vertically to match Slicer's display convention for 2-D images.
        """
        try:
            import SimpleITK as sitk
        except ImportError as error:
            raise ImportError(
                "Writing anatomy masks requires SimpleITK: pip install SimpleITK"
            ) from error

        output_dir = Path(tempfile.mkdtemp(prefix="anatomy_monai_"))
        output_path = output_dir / f"{image_path.stem}_anatomy_mask.nrrd"
        raw_mask_path = output_dir / f"{image_path.stem}_anatomy_mask_raw.png"
        cv2.imwrite(str(raw_mask_path), mask.astype(np.uint8))

        mask_flipped = np.flipud(mask).astype(np.uint8)
        mask_volume = mask_flipped[np.newaxis, :, :]  # 1 × H × W

        itk_mask = sitk.GetImageFromArray(mask_volume)
        itk_mask.SetSpacing((1.0, 1.0, 1.0))
        if params:
            itk_mask.SetMetaData(
                "ChestAnalyzer.confidence",
                json.dumps(params.get("confidence", {})),
            )
            ctr = params.get("ctr")
            if ctr is not None:
                itk_mask.SetMetaData("ChestAnalyzer.ctr", str(ctr))
            for key in (
                "analysis_scope",
                "roi_source",
                "input_shape",
                "source_image_shape",
                "bbox_in_source_image",
                "lung_mask_source",
            ):
                value = params.get(key)
                if value is not None:
                    itk_mask.SetMetaData(f"ChestAnalyzer.{key}", json.dumps(value))
        itk_mask.SetMetaData("ChestAnalyzer.raw_mask_png", str(raw_mask_path))
        sitk.WriteImage(itk_mask, str(output_path))

        self._debug_log("mask written", path=output_path, raw_mask_path=raw_mask_path)
        return output_path


def create_anatomy_segmentation_infer(studies=None, device=None, num_threads=2):
    return AnatomySegmentationInfer(
        studies=studies,
        device=device,
        num_threads=num_threads,
    ).task
