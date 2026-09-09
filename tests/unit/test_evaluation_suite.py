import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.classifier.evaluation_suite import (
    curve_metrics,
    export_errors,
    probability_metrics,
)


class EvaluationSuiteTests(unittest.TestCase):
    def test_probability_and_curve_metrics(self):
        targets = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        calibration = probability_metrics(targets, probabilities, bins=5)
        curves = curve_metrics(targets, probabilities)
        self.assertAlmostEqual(curves["roc_auc"], 1.0)
        self.assertAlmostEqual(curves["pr_auc"], 1.0)
        self.assertLess(calibration["brier_score"], 0.05)
        self.assertGreaterEqual(calibration["ece"], 0.0)

    def test_error_export_separates_false_positives_and_negatives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fp_image = root / "fp.png"
            fn_image = root / "fn.png"
            Image.new("L", (8, 8), color=100).save(fp_image)
            Image.new("L", (8, 8), color=120).save(fn_image)
            result = export_errors(
                [
                    {
                        "sample_id": "fp",
                        "image_path": str(fp_image),
                        "target": 0,
                        "prediction": 1,
                        "p_normal": 0.1,
                        "p_pneumonia": 0.9,
                    },
                    {
                        "sample_id": "fn",
                        "image_path": str(fn_image),
                        "target": 1,
                        "prediction": 0,
                        "p_normal": 0.8,
                        "p_pneumonia": 0.2,
                    },
                ],
                root / "errors",
            )
            self.assertEqual(result["false_positives"], 1)
            self.assertEqual(result["false_negatives"], 1)
            self.assertTrue((root / "errors/false_positives/fp.png").exists())
            self.assertTrue((root / "errors/false_negatives/fn.png").exists())


if __name__ == "__main__":
    unittest.main()
