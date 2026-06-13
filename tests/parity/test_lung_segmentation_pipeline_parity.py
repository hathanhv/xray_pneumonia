import csv
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.core.config import load_config
from src.lung_segmentation import LungSegmentationConfig, LungSegmentationPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "outputs" / "legacy_baselines" / "segmentation"


def dice_score(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual > 0
    expected = expected > 0
    denominator = int(actual.sum() + expected.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(actual, expected).sum()) / denominator


class LungSegmentationPipelineParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "lung_segmentation"
            / "unet_lung_segmentation.pth"
        )
        summary_path = BASELINE_DIR / "parity_summary.json"
        if not checkpoint_path.exists() or not summary_path.exists():
            raise unittest.SkipTest(
                "Segmentation checkpoint and legacy baseline are external artifacts"
            )
        raw_config = load_config(
            PROJECT_ROOT / "configs" / "pipelines" / "lung_segmentation_2025.yaml"
        )
        cls.pipeline = LungSegmentationPipeline(
            LungSegmentationConfig.from_dict(
                raw_config,
                project_root=PROJECT_ROOT,
            )
        )

        with summary_path.open(encoding="utf-8") as handle:
            cls.expected_metrics = json.load(handle)

    def test_pipeline_preserves_legacy_segmentation_outputs(self) -> None:
        dice_scores = []
        qc_matches = 0
        crop_matches = 0

        with (BASELINE_DIR / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        for row in rows:
            image_path = PROJECT_ROOT / row["prepared_image_path"]
            result = self.pipeline.predict(image_path)
            expected_mask = cv2.imread(
                str(BASELINE_DIR / "masks" / f"{image_path.stem}_mask.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            expected_crop = cv2.imread(
                str(BASELINE_DIR / "crops" / f"{image_path.stem}_crop.png"),
                cv2.IMREAD_UNCHANGED,
            )

            self.assertIsNotNone(expected_mask)
            self.assertIsNotNone(expected_crop)
            dice_scores.append(dice_score(result.mask, expected_mask))
            qc_matches += int(result.qc_status == row["expected_qc_status"])
            crop_matches += int(np.array_equal(result.crop, expected_crop))

        self.assertEqual(len(rows), self.expected_metrics["sample_count"])
        self.assertGreaterEqual(
            min(dice_scores),
            self.expected_metrics["minimum_mask_dice"],
        )
        self.assertEqual(qc_matches, self.expected_metrics["qc_matches"])
        self.assertEqual(crop_matches, self.expected_metrics["crop_exact_matches"])


if __name__ == "__main__":
    unittest.main()
