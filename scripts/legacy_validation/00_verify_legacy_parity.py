from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lung_segmentation.crop import crop_by_mask
from src.lung_segmentation.model import load_lung_segmentation_model
from src.lung_segmentation.postprocess import clean_mask
from src.lung_segmentation.predict import predict_lung_mask
from src.lung_segmentation.qc import check_mask_quality


MANIFEST_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "manifests" / "segmentation.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "legacy_baselines"
    / "segmentation"
    / "parity_report.csv"
)
SUMMARY_PATH = OUTPUT_PATH.with_name("parity_summary.json")


def dice_score(left: np.ndarray, right: np.ndarray) -> float:
    left = left > 0
    right = right > 0
    denominator = int(left.sum()) + int(right.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int(np.logical_and(left, right).sum()) / denominator


def read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def main() -> None:
    model, img_size, device = load_lung_segmentation_model()

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))

    rows = []
    for item in manifest:
        image_path = PROJECT_ROOT / item["prepared_image_path"]
        expected_mask = read_binary_mask(PROJECT_ROOT / item["expected_mask_path"])
        expected_crop = cv2.imread(
            str(PROJECT_ROOT / item["expected_crop_path"]),
            cv2.IMREAD_UNCHANGED,
        )

        image, predicted_mask = predict_lung_mask(
            model=model,
            image_path=image_path,
            img_size=img_size,
            device=device,
            threshold=0.5,
        )
        cleaned_mask = clean_mask(predicted_mask, keep_components=2, fill_holes=True)
        predicted_crop, bbox = crop_by_mask(image, cleaned_mask)
        qc_status, metrics = check_mask_quality(cleaned_mask, bbox, image.shape)

        crop_equal = (
            predicted_crop is not None
            and expected_crop is not None
            and predicted_crop.shape == expected_crop.shape
            and np.array_equal(predicted_crop, expected_crop)
        )
        mask_equal = np.array_equal(cleaned_mask, expected_mask)

        rows.append(
            {
                "prepared_image_path": item["prepared_image_path"],
                "expected_qc_status": item["expected_qc_status"],
                "actual_qc_status": qc_status,
                "qc_match": qc_status == item["expected_qc_status"],
                "mask_equal": mask_equal,
                "mask_dice": dice_score(cleaned_mask, expected_mask),
                "crop_equal": crop_equal,
                "bbox_x1": metrics["bbox_x1"],
                "bbox_y1": metrics["bbox_y1"],
                "bbox_x2": metrics["bbox_x2"],
                "bbox_y2": metrics["bbox_y2"],
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_count": len(rows),
        "mask_exact_matches": sum(row["mask_equal"] for row in rows),
        "minimum_mask_dice": min(row["mask_dice"] for row in rows),
        "qc_matches": sum(row["qc_match"] for row in rows),
        "crop_exact_matches": sum(row["crop_equal"] for row in rows),
        "device": str(device),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
