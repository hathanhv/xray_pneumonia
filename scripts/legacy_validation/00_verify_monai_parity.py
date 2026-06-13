from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "monai_apps" / "lung_monai_app"
sys.path.insert(0, str(APP_ROOT))

from lib.infers.lung_infer import LungSegmentationInfer


MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "manifests" / "monai.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "legacy_baselines" / "monai"
REPORT_PATH = OUTPUT_ROOT / "parity_report.csv"
SUMMARY_PATH = OUTPUT_ROOT / "parity_summary.json"


def dice_score(left: np.ndarray, right: np.ndarray) -> float:
    left = left > 0
    right = right > 0
    denominator = int(left.sum()) + int(right.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int(np.logical_and(left, right).sum()) / denominator


def read_png_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def main() -> None:
    infer = LungSegmentationInfer(
        model_dir=APP_ROOT / "model",
        studies=PROJECT_ROOT / "data" / "qc" / "fail_qc" / "images",
        threshold=0.5,
    )

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))

    rows = []
    for item in manifest:
        image_path = PROJECT_ROOT / item["image_path"]
        expected_mask = read_png_mask(
            PROJECT_ROOT / item["legacy_predicted_mask_path"]
        )

        image_array, reference_info = infer._read_image(image_path)
        predicted_mask = infer._predict_array(image_array)
        nrrd_path = infer._write_mask(predicted_mask, image_path, reference_info)

        nrrd_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(nrrd_path))).squeeze()
        nrrd_mask = (nrrd_mask > 0).astype(np.uint8)
        restored_nrrd_mask = np.flipud(nrrd_mask)

        rows.append(
            {
                "image_path": item["image_path"],
                "prediction_exact": np.array_equal(predicted_mask, expected_mask),
                "prediction_dice": dice_score(predicted_mask, expected_mask),
                "nrrd_orientation_exact": np.array_equal(
                    restored_nrrd_mask, predicted_mask
                ),
                "nrrd_values": ",".join(
                    str(int(value)) for value in np.unique(nrrd_mask)
                ),
            }
        )

    with REPORT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_count": len(rows),
        "prediction_exact_matches": sum(row["prediction_exact"] for row in rows),
        "minimum_prediction_dice": min(row["prediction_dice"] for row in rows),
        "nrrd_orientation_matches": sum(
            row["nrrd_orientation_exact"] for row in rows
        ),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
