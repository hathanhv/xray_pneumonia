import csv
import unittest
from pathlib import Path

from src.classifier import ClassifierInferenceConfig, ClassifierInferenceService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "legacy_baselines"
    / "classification_2025"
    / "predictions.csv"
)


class ClassifierInferenceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "pneumonia_classifier"
            / "mobilenet_2025_lung_crop_corrected.pth"
        )
        if not checkpoint_path.exists() or not BASELINE_PATH.exists():
            raise unittest.SkipTest(
                "Classifier checkpoint and legacy baseline are external artifacts"
            )
        cls.service = ClassifierInferenceService(
            ClassifierInferenceConfig(
                checkpoint_path=checkpoint_path,
                device="cpu",
                include_gradcam=False,
            )
        )

    def test_predictions_and_probabilities_match_task_zero_baseline(self):
        with BASELINE_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 20)
        for row in rows:
            result = self.service.predict_path(PROJECT_ROOT / row["image_path"])
            self.assertEqual(result.predicted_index, int(row["prediction"]))
            self.assertAlmostEqual(
                result.probabilities["NORMAL"],
                float(row["p_normal"]),
                places=7,
            )
            self.assertAlmostEqual(
                result.probabilities["PNEUMONIA"],
                float(row["p_pneumonia"]),
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
