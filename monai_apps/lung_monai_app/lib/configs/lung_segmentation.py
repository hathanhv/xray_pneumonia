from pathlib import Path

from lib.infers.lung_infer import create_lung_segmentation_infer


class LungSegmentationConfig:
    """
    Minimal config for the only model in this MONAI Label app.
    """

    name = "lung_segmentation"
    labels = {"lung": 1}

    def __init__(self, app_dir, studies=None, conf=None):
        self.app_dir = Path(app_dir)
        self.studies = Path(studies) if studies else None
        self.conf = conf or {}
        self.model_dir = self.app_dir / "model"

    def infer(self):
        threshold = float(self.conf.get("threshold", 0.5))
        return create_lung_segmentation_infer(
            model_dir=self.model_dir,
            studies=self.studies,
            threshold=threshold,
        )
