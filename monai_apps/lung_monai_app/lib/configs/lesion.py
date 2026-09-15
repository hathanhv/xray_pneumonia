from pathlib import Path

from lib.infers.lesion_infer import create_lesion_localization_infer


class LesionConfig:
    """
    MONAI Label config for MedicalPatchNet lesion localization.

    Defaults are CPU-friendly. Set lesion_shift_pixels=16 for smoother maps
    after confirming latency is acceptable on the target machine.
    """

    name = "lesion_localization"

    def __init__(self, app_dir, studies=None, conf=None):
        self.app_dir = Path(app_dir)
        self.studies = Path(studies) if studies else None
        self.conf = conf or {}
        self.project_root = self.app_dir.parents[1]

    def infer(self):
        checkpoint_value = self.conf.get("lesion_checkpoint")
        checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
        if checkpoint_path and not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path

        return create_lesion_localization_infer(
            studies=self.studies,
            checkpoint_path=checkpoint_path,
            device=self.conf.get("lesion_device", "cpu"),
            num_threads=self.conf.get("lesion_num_threads", 2),
            shift_pixels=self.conf.get("lesion_shift_pixels", 64),
            shift_batch_size=self.conf.get("lesion_shift_batch_size", 4),
            probability_threshold=self.conf.get("lesion_probability_threshold", 0.50),
            mask_logit_threshold=self.conf.get("lesion_mask_logit_threshold", 1.0),
            top_k=self.conf.get("lesion_top_k", 5),
            include_overlay=self.conf.get("lesion_overlay", True),
            flip_display_vertical=self.conf.get("lesion_flip_display_vertical", False),
            lung_model_dir=self.app_dir / "model",
            lung_threshold=self.conf.get("threshold", 0.5),
            auto_lung_segmentation=self.conf.get("lesion_auto_lung_segmentation", True),
        )
