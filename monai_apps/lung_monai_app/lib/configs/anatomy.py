from pathlib import Path

from lib.infers.anatomy_infer import create_anatomy_segmentation_infer


class AnatomyConfig:
    """
    MONAI Label config for chest X-ray anatomy segmentation.

    Uses ianpan/chest-x-ray-basic (HuggingFace) to produce:
      label 1 = right_lung
      label 2 = left_lung
      label 3 = heart

    The model is downloaded on first inference call and cached by
    HuggingFace's model hub mechanism.
    """

    name = "anatomy_segmentation"
    labels = {"right_lung": 1, "left_lung": 2, "heart": 3}

    def __init__(self, app_dir, studies=None, conf=None):
        self.app_dir = Path(app_dir)
        self.studies = Path(studies) if studies else None
        self.conf = conf or {}

    def infer(self):
        device = self.conf.get("anatomy_device") or "cpu"
        num_threads = self.conf.get("anatomy_num_threads", 2)
        return create_anatomy_segmentation_infer(
            studies=self.studies,
            device=device,
            num_threads=num_threads,
        )
