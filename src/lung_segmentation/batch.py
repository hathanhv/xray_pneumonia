from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .pipeline import LungSegmentationPipeline


QC_REPORT_FIELDS = [
    "filename",
    "original_path",
    "split",
    "class",
    "qc_status",
    "mask_area_ratio",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_area_ratio",
    "bbox_height_ratio",
    "bbox_width_ratio",
    "bbox_bottom_ratio",
    "mask_center_y_ratio",
]


def read_csv_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def validate_manifest_columns(
    rows: Iterable[Mapping[str, Any]],
    required_columns: set[str],
) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Manifest is empty.")
    missing = required_columns - set(rows[0])
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")


def write_qc_report(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=QC_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_manifest_inference(
    pipeline: LungSegmentationPipeline,
    *,
    manifest_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    filename_column: str = "new_filename",
    original_path_column: str = "original_path",
    split_column: str = "split",
    class_column: str = "class",
    progress_callback=None,
) -> tuple[list[dict[str, Any]], Counter]:
    rows = read_csv_manifest(manifest_path)
    required = {
        filename_column,
        original_path_column,
        split_column,
        class_column,
    }
    validate_manifest_columns(rows, required)

    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Input images folder not found: {images_dir}")

    report_rows = []
    for index, row in enumerate(rows, start=1):
        filename = row[filename_column]
        image_path = images_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in manifest not found: {image_path}")

        if progress_callback:
            progress_callback(index, len(rows), filename)

        result = pipeline.predict(image_path)
        pipeline.save_result(result, output_dir=output_dir)
        report_rows.append(
            {
                "filename": filename,
                "original_path": row[original_path_column],
                "split": row[split_column],
                "class": row[class_column],
                "qc_status": result.qc_status,
                **result.qc_metrics,
            }
        )

    write_qc_report(report_rows, report_path)
    summary = Counter(row["qc_status"] for row in report_rows)
    return report_rows, summary
