import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import (  # noqa: E402
    ClassifierInferenceConfig,
    ClassifierInferenceService,
)


class ClassifierInfer:
    """MONAI Label adapter for MobileNetV2 pneumonia classification."""

    def __init__(
        self,
        checkpoint_path,
        studies=None,
        device=None,
        include_gradcam=True,
    ):
        try:
            from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
        except ImportError as error:
            raise ImportError(
                "Missing dependency for MONAI Label classification: monailabel"
            ) from error

        self.studies = Path(studies) if studies else None
        self.include_gradcam = bool(include_gradcam)
        self.service = ClassifierInferenceService(
            ClassifierInferenceConfig(
                checkpoint_path=Path(checkpoint_path),
                device=device,
                include_gradcam=self.include_gradcam,
            )
        )

        class _InferTask(InferTask):
            def __init__(self, outer):
                super().__init__(
                    type=InferType.CLASSIFICATION,
                    labels=outer.service.class_to_idx,
                    dimension=2,
                    description=(
                        "MobileNetV2 NORMAL/PNEUMONIA classification with "
                        "optional edited lung labelmap"
                    ),
                    config={
                        "device": ["cuda", "cpu"],
                        "include_gradcam": outer.include_gradcam,
                        "accepts_lung_label": True,
                    },
                )
                self.outer = outer

            def is_valid(self):
                return self.outer.service.config.checkpoint_path.exists()

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
        label_value = request.get("label_path") or request.get("label")
        label_path = (
            self._resolve_path(label_value, name="label", use_studies=False)
            if label_value
            else None
        )
        include_gradcam = self._as_bool(
            request.get("include_gradcam", self.include_gradcam)
        )

        result = self.service.predict_path(
            image_path,
            mask_path=label_path,
            include_gradcam=include_gradcam,
        )
        params = result.to_dict()
        params["source"] = str(image_path)
        params["lung_label_source"] = str(label_path) if label_path else None
        return None, params

    def _resolve_path(self, value, name, use_studies=True):
        if not value:
            raise ValueError(f"Classification request missing {name}")

        path = Path(str(value))
        if path.exists():
            return path

        if use_studies and self.studies:
            candidate = self.studies / str(value)
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not resolve {name} path: {value}")

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}


def create_classifier_infer(
    checkpoint_path,
    studies=None,
    device=None,
    include_gradcam=True,
):
    return ClassifierInfer(
        checkpoint_path=checkpoint_path,
        studies=studies,
        device=device,
        include_gradcam=include_gradcam,
    ).task
