import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lesion import MedicalPatchNetConfig, MedicalPatchNetService  # noqa: E402


class LesionLocalizationInfer:
    """MONAI Label adapter for MedicalPatchNet lesion localization."""

    def __init__(
        self,
        studies=None,
        checkpoint_path=None,
        device=None,
        num_threads=2,
        shift_pixels=64,
        shift_batch_size=4,
        probability_threshold=0.50,
        mask_logit_threshold=5.0,
        top_k=5,
        include_overlay=True,
        flip_display_vertical=True,
    ):
        try:
            from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
        except ImportError as error:
            raise ImportError(
                "Missing dependency for MONAI Label lesion localization: monailabel"
            ) from error

        self.studies = Path(studies) if studies else None
        checkpoint = Path(checkpoint_path) if checkpoint_path else None
        self.service = MedicalPatchNetService(
            MedicalPatchNetConfig(
                checkpoint_path=checkpoint,
                device=device,
                num_threads=int(num_threads),
                shift_pixels=int(shift_pixels),
                shift_batch_size=int(shift_batch_size),
                probability_threshold=float(probability_threshold),
                mask_logit_threshold=float(mask_logit_threshold),
                top_k=int(top_k),
                include_overlay=self._as_bool(include_overlay),
                flip_display_vertical=self._as_bool(flip_display_vertical),
            )
        )

        class _InferTask(InferTask):
            def __init__(self, outer):
                super().__init__(
                    type=InferType.CLASSIFICATION,
                    labels={name: idx for idx, name in enumerate(self.pathologies())},
                    dimension=2,
                    description=(
                        "MedicalPatchNet 14-finding classification with "
                        "patch-based lesion localization overlay"
                    ),
                    config={
                        "device": ["cuda", "cpu"],
                        "shift_pixels": int(shift_pixels),
                        "mask_logit_threshold": float(mask_logit_threshold),
                    },
                )
                self.outer = outer

            @staticmethod
            def pathologies():
                from src.lesion.medical_patchnet import PATHOLOGY_NAMES

                return PATHOLOGY_NAMES

            def is_valid(self):
                return True

            def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
                return self.outer.infer(request)

        self.task = _InferTask(self)

    def __getattr__(self, name):
        return getattr(self.task, name)

    def infer(self, request):
        image_path = self._resolve_path(
            request.get("image_path") or request.get("image"),
            name="image",
        )
        result = self.service.predict_path(image_path)
        return None, result.to_dict()

    def _resolve_path(self, value, name):
        if not value:
            raise ValueError(f"Lesion localization request missing {name}")

        path = Path(str(value))
        if path.exists():
            return path

        if self.studies:
            candidate = self.studies / str(value)
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not resolve {name} path: {value}")

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}


def create_lesion_localization_infer(**kwargs):
    return LesionLocalizationInfer(**kwargs).task
