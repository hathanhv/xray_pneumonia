from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from src.lung_segmentation.config import LungSegmentationConfig
from src.lung_segmentation.batch import QC_REPORT_FIELDS, run_manifest_inference
from src.lung_segmentation.pipeline import LungSegmentationPipeline


class ConstantMaskModel(torch.nn.Module):
    def forward(self, tensor):
        batch, _, height, width = tensor.shape
        logits = torch.full((batch, 1, height, width), -10.0)
        logits[:, :, height // 4 : height * 3 // 4, width // 4 : width * 3 // 4] = 10.0
        return logits


class LungSegmentationPipelineTests(unittest.TestCase):
    def test_predicts_and_saves_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            cv2.imwrite(str(image_path), np.full((100, 120), 127, dtype=np.uint8))

            config = LungSegmentationConfig.from_dict(
                {
                    "model": {"checkpoint_path": "unused.pth"},
                    "crop": {
                        "pad_left": 0,
                        "pad_right": 0,
                        "pad_top": 0,
                        "pad_bottom": 0,
                        "max_bottom_ratio": 1.0,
                    },
                    "output": {"output_dir": str(root / "outputs")},
                },
                project_root=root,
            )
            pipeline = LungSegmentationPipeline(
                config,
                model=ConstantMaskModel(),
                img_size=32,
                device=torch.device("cpu"),
            )
            result = pipeline.predict(image_path)
            paths = pipeline.save_result(result)

            self.assertEqual(result.mask.shape, (100, 120))
            self.assertIsNotNone(result.crop)
            self.assertTrue(paths["mask"].exists())
            self.assertTrue(paths["crop"].exists())
            self.assertTrue(paths["overlay"].exists())

    def test_manifest_runner_preserves_legacy_report_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_dir = root / "images"
            images_dir.mkdir()
            image_path = images_dir / "sample.png"
            cv2.imwrite(str(image_path), np.full((100, 120), 127, dtype=np.uint8))

            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["new_filename", "original_path", "split", "class"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "new_filename": "sample.png",
                        "original_path": "source/sample.png",
                        "split": "test",
                        "class": "NORMAL",
                    }
                )

            config = LungSegmentationConfig.from_dict(
                {
                    "model": {"checkpoint_path": "unused.pth"},
                    "crop": {"max_bottom_ratio": 1.0},
                    "output": {"output_dir": str(root / "outputs")},
                },
                project_root=root,
            )
            pipeline = LungSegmentationPipeline(
                config,
                model=ConstantMaskModel(),
                img_size=32,
                device=torch.device("cpu"),
            )
            report_path = root / "qc_report.csv"
            rows, summary = run_manifest_inference(
                pipeline,
                manifest_path=manifest_path,
                images_dir=images_dir,
                output_dir=config.output.output_dir,
                report_path=report_path,
            )

            with report_path.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                report_rows = list(reader)

            self.assertEqual(reader.fieldnames, QC_REPORT_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(sum(summary.values()), 1)
            self.assertEqual(report_rows[0]["filename"], "sample.png")


if __name__ == "__main__":
    unittest.main()
