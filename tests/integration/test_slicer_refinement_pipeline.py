import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.pipelines.slicer_refinement import SlicerRefinementPipeline


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SlicerRefinementPipelineTests(unittest.TestCase):
    def test_direct_image_refinement_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "input/images"
            predicted_masks = root / "output/masks"
            predicted_crops = root / "output/cropped"
            overlays = root / "output/overlays"
            for directory in (images, predicted_masks, predicted_crops, overlays):
                directory.mkdir(parents=True)

            metadata_rows = []
            qc_rows = []
            for index, status in enumerate(("PASS", "FAIL_TOO_SMALL"), start=1):
                filename = f"sample_{index}.png"
                stem = Path(filename).stem
                image = np.full((32, 32), 100 + index, dtype=np.uint8)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[6:26, 7:25] = 255
                cv2.imwrite(str(images / filename), image)
                cv2.imwrite(str(predicted_masks / f"{stem}_mask.png"), mask)
                cv2.imwrite(str(predicted_crops / f"{stem}_crop.png"), image)
                cv2.imwrite(str(overlays / f"{stem}_overlay.png"), image)
                metadata_rows.append(
                    {
                        "new_filename": filename,
                        "original_path": str(images / filename),
                        "split": "train",
                        "class": "NORMAL",
                    }
                )
                qc_rows.append({"filename": filename, "qc_status": status})

            metadata_path = root / "input/metadata.csv"
            qc_path = root / "output/qc.csv"
            write_rows(metadata_path, metadata_rows)
            write_rows(qc_path, qc_rows)
            config = {
                "input": {
                    "images_dir": str(images),
                    "manifest_path": str(metadata_path),
                    "filename_column": "new_filename",
                    "original_path_column": "original_path",
                    "split_column": "split",
                    "class_column": "class",
                },
                "report": {"qc_report_path": str(qc_path)},
                "slicer_refinement": {
                    "predicted_masks_dir": str(predicted_masks),
                    "predicted_crops_dir": str(predicted_crops),
                    "overlays_dir": str(overlays),
                    "studies_dir": str(root / "qc/fail/images"),
                    "study_predicted_masks_dir": str(root / "qc/fail/masks"),
                    "study_crops_dir": str(root / "qc/fail/crops"),
                    "study_overlays_dir": str(root / "qc/fail/overlays"),
                    "studies_manifest_path": str(root / "qc/fail/manifest.csv"),
                    "labels_dir": str(root / "qc/fail/images/labels/final"),
                    "corrected_images_dir": str(root / "qc/corrected/images"),
                    "corrected_masks_dir": str(root / "qc/corrected/masks"),
                    "pass_crops_dir": str(root / "qc/corrected/crop"),
                    "final_masks_dir": str(root / "output/final_masks"),
                    "merge_report_path": str(root / "output/merge.csv"),
                    "final_dataset_dir": str(root / "final"),
                    "final_dataset_report_path": str(root / "final/report.csv"),
                    "failed_crops_dir": str(root / "final/failed"),
                    "status_path": str(root / "qc/status.json"),
                    "pass_status": "PASS",
                    "flip_vertical": False,
                    "crop": {
                        "pad_left": 0,
                        "pad_right": 0,
                        "pad_top": 0,
                        "pad_bottom": 0,
                        "max_bottom_ratio": 1.0,
                    },
                },
            }
            pipeline = SlicerRefinementPipeline(config)

            prepared = pipeline.prepare_studies()
            self.assertEqual(prepared["studies"], 1)
            label_path = root / "qc/fail/images/labels/final/sample_2.nii.gz"
            label_path.touch()
            label = np.zeros((1, 32, 32), dtype=np.uint8)
            label[0, 5:27, 6:26] = 1
            imported = pipeline.import_labels(
                label_reader=lambda _path: label,
                flip_vertical=False,
            )
            self.assertEqual(imported["converted"], 1)

            merged = pipeline.merge_masks()
            self.assertEqual(merged["predicted"], 1)
            self.assertEqual(merged["corrected"], 1)
            cropped = pipeline.create_final_dataset()
            self.assertEqual(cropped["saved"], 2)
            status = pipeline.status()
            self.assertTrue(status["complete"])

    def test_empty_slicer_label_is_not_used_as_correction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            studies = root / "studies"
            labels = studies / "labels/final"
            corrected = root / "corrected/masks"
            labels.mkdir(parents=True)
            corrected.mkdir(parents=True)
            cv2.imwrite(
                str(studies / "sample.png"),
                np.full((16, 16), 100, dtype=np.uint8),
            )
            label_path = labels / "sample.nii.gz"
            label_path.touch()
            stale_mask = corrected / "sample_mask.png"
            cv2.imwrite(
                str(stale_mask),
                np.ones((16, 16), dtype=np.uint8) * 255,
            )
            config = {
                "input": {
                    "images_dir": str(root / "unused"),
                    "manifest_path": str(root / "unused.csv"),
                },
                "report": {"qc_report_path": str(root / "unused_qc.csv")},
                "slicer_refinement": {
                    "studies_dir": str(studies),
                    "labels_dir": str(labels),
                    "corrected_images_dir": str(root / "corrected/images"),
                    "corrected_masks_dir": str(corrected),
                },
            }

            result = SlicerRefinementPipeline(config).import_labels(
                label_reader=lambda _path: np.zeros((1, 16, 16)),
                flip_vertical=False,
            )

            self.assertEqual(result["empty_label"], 1)
            self.assertFalse(stale_mask.exists())


if __name__ == "__main__":
    unittest.main()
