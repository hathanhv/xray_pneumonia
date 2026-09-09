from pathlib import Path

from lib.infers.classifier_infer import create_classifier_infer


class ClassifierConfig:
    name = "classifier"
    labels = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(self, app_dir, studies=None, conf=None):
        self.app_dir = Path(app_dir)
        self.studies = Path(studies) if studies else None
        self.conf = conf or {}
        self.project_root = self.app_dir.parents[1]

    def infer(self):
        checkpoint_value = self.conf.get(
            "classifier_checkpoint",
            "checkpoints/pneumonia_classifier/"
            "mobilenet_2025_lung_crop_corrected.pth",
        )
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path

        return create_classifier_infer(
            checkpoint_path=checkpoint_path,
            studies=self.studies,
            device=self.conf.get("classifier_device"),
            include_gradcam=self._as_bool(
                self.conf.get("classifier_gradcam", True)
            ),
        )

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
