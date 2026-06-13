from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.dataset import (
    ClassificationRecord,
    save_manifest,
    sklearn_stratified_split_records,
)
from src.pipelines.slicer_refinement import read_csv


METADATA_PATH = PROJECT_ROOT / "data/lung_seg_input/2025_all/metadata.csv"
PREDICTED_MASKS_DIR = PROJECT_ROOT / "data/lung_seg_outputs/2025_all/masks"
FINAL_MASKS_DIR = PROJECT_ROOT / "data/lung_seg_outputs/2025_all/final_masks"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/classification_2025_task10.csv"
REFERENCE_PATH = PROJECT_ROOT / "data/references/histogram_reference.png"
REFERENCE_SOURCE_DIR = PROJECT_ROOT / "data/raw/chest_xray_2018/train"


def create_manifest(val_fraction=0.15, seed=42):
    rows = read_csv(METADATA_PATH)
    records = []
    for row in rows:
        filename = row["new_filename"]
        stem = Path(filename).stem
        image_path = PROJECT_ROOT / "data/lung_seg_input/2025_all/images" / filename
        records.append(
            ClassificationRecord(
                image_path=image_path,
                label=0 if row["class"] == "NORMAL" else 1,
                class_name=row["class"],
                split=row["split"].lower(),
                sample_id=stem,
                patient_id=stem,
                mask_path=PREDICTED_MASKS_DIR / f"{stem}_mask.png",
                refined_mask_path=FINAL_MASKS_DIR / f"{stem}_mask.png",
            )
        )
    splits = sklearn_stratified_split_records(
        records,
        val_fraction=val_fraction,
        seed=seed,
        preserve_existing_test=True,
    )
    output = [record for name in ("train", "val", "test") for record in splits[name]]
    save_manifest(output, MANIFEST_PATH)
    return {name: len(split) for name, split in splits.items()}


def create_histogram_reference(max_images=100):
    image_paths = sorted(
        path
        for path in REFERENCE_SOURCE_DIR.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )[:max_images]
    if not image_paths:
        raise FileNotFoundError(
            f"No 2018 reference images found under: {REFERENCE_SOURCE_DIR}"
        )
    arrays = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            arrays.append(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA))
    if not arrays:
        raise RuntimeError("No readable images were available for histogram reference")
    reference = np.median(np.stack(arrays), axis=0).astype(np.uint8)
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(REFERENCE_PATH), reference)
    return len(arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-reference-images", type=int, default=100)
    args = parser.parse_args()
    split_counts = create_manifest(args.val_fraction, args.seed)
    reference_count = create_histogram_reference(args.max_reference_images)
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Splits: {split_counts}")
    print(f"Histogram reference: {REFERENCE_PATH} ({reference_count} image(s))")


if __name__ == "__main__":
    main()
