from pathlib import Path

from lib.infers.cxformer_pathology_infer import create_cxformer_pathology_infer


class CXformerPathologyConfig:
    """MONAI Label config for CXFormer pathology + transformer heatmap + fusion."""

    name = "cxformer_pathology"

    def __init__(self, app_dir, studies=None, conf=None):
        self.app_dir = Path(app_dir)
        self.studies = Path(studies) if studies else None
        self.conf = conf or {}
        self.project_root = self.app_dir.parents[1]

    def infer(self):
        checkpoint_value = self.conf.get("cxformer_checkpoint")
        thresholds_value = self.conf.get("cxformer_thresholds")

        checkpoint_path = (
            Path(checkpoint_value)
            if checkpoint_value
            else self.project_root / "checkpoints" / "cxformer" / "cxformer_final_all15000.pt"
        )
        thresholds_path = (
            Path(thresholds_value)
            if thresholds_value
            else self.project_root / "checkpoints" / "cxformer" / "thresholds_for_future_final_model.csv"
        )

        if not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path
        if not thresholds_path.is_absolute():
            thresholds_path = self.project_root / thresholds_path

        return create_cxformer_pathology_infer(
            studies=self.studies,
            checkpoint_path=checkpoint_path,
            thresholds_csv=thresholds_path,
            backbone_name=self.conf.get("cxformer_backbone", "m42-health/CXformer-small"),
            device=self.conf.get("cxformer_device", "cpu"),
            num_threads=self.conf.get("cxformer_num_threads", 2),
            top_k=self.conf.get("cxformer_top_k", 5),
            heatmap_threshold=self.conf.get("cxformer_heatmap_threshold", 0.60),
            min_lesion_pixels=self.conf.get("cxformer_min_lesion_pixels", 100),
        )
