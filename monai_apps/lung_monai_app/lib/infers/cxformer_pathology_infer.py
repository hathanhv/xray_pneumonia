import sys
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cxformer import CXformerPathologyConfig, CXformerPathologyService  # noqa: E402


logger = logging.getLogger(__name__)


class CXformerPathologyInfer:
    """MONAI Label adapter for CXFormer 14-label classification.

    Localization remains optional for backward compatibility; ChestAnalyzer now
    requests classification-only output so MedicalPatchNet owns the Yellow lesion panel.
    """

    def __init__(
        self,
        studies=None,
        checkpoint_path=None,
        thresholds_csv=None,
        backbone_name="m42-health/CXformer-small",
        device=None,
        num_threads=2,
        top_k=5,
        heatmap_threshold=0.60,
        min_lesion_pixels=100,
    ):
        try:
            from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
        except ImportError as error:
            raise ImportError(
                "Missing dependency for MONAI Label CXFormer pathology inference: monailabel"
            ) from error

        self.studies = Path(studies) if studies else None
        self.service = CXformerPathologyService(
            CXformerPathologyConfig(
                checkpoint_path=Path(checkpoint_path),
                thresholds_csv=Path(thresholds_csv),
                backbone_name=str(backbone_name),
                device=device,
                num_threads=int(num_threads),
                top_k=int(top_k),
                heatmap_threshold=float(heatmap_threshold),
                min_lesion_pixels=int(min_lesion_pixels),
            )
        )

        class _InferTask(InferTask):
            def __init__(self, outer):
                super().__init__(
                    type=InferType.CLASSIFICATION,
                    labels={},
                    dimension=2,
                    description=(
                        "CXFormer 14-label classification with optional transformer "
                        "localization for backward compatibility"
                    ),
                    config={
                        "device": ["cuda", "cpu"],
                        "heatmap_threshold": float(heatmap_threshold),
                    },
                )
                self.outer = outer

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
            required=True,
        )
        include_localization = self._as_bool(
            request.get("include_localization", True)
        )
        anatomy_path = self._resolve_path(
            request.get("label") or request.get("anatomy_mask"),
            name="anatomy mask",
            required=False,
        ) if include_localization else None
        try:
            result = self.service.predict_path(
                image_path=image_path,
                anatomy_mask_path=anatomy_path,
                include_localization=include_localization,
            )
        except Exception:
            logger.exception(
                "CXFormer inference failed | image=%s | include_localization=%s",
                image_path,
                include_localization,
            )
            raise
        return None, result.to_dict()


    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_path(self, value, name, required=True):
        if not value:
            if required:
                raise ValueError(f"CXFormer request missing {name}")
            return None

        path = Path(str(value))
        if path.exists():
            return path

        if self.studies:
            candidate = self.studies / str(value)
            if candidate.exists():
                return candidate

        if required:
            raise FileNotFoundError(f"Could not resolve {name} path: {value}")
        return None


def create_cxformer_pathology_infer(**kwargs):
    return CXformerPathologyInfer(**kwargs).task
