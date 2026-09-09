from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from src.lung_segmentation.crop import crop_by_mask


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LABEL_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".nhdr")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def require_columns(
    rows: list[Mapping[str, Any]],
    columns: set[str],
    *,
    source: str,
) -> None:
    if not rows:
        raise ValueError(f"{source} is empty")
    missing = columns - set(rows[0])
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")


def strip_label_extension(path: Path) -> str:
    name = path.name
    for suffix in LABEL_EXTENSIONS:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def normalize_label_stem(stem: str) -> str:
    lowered = stem.lower()
    for suffix in ("-label", "_label", ".seg", "_seg", "-seg"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def select_2d_mask(label_array: np.ndarray) -> np.ndarray:
    label_array = np.asarray(label_array)
    if label_array.ndim == 2:
        return label_array
    if label_array.ndim == 3:
        areas = [(label_array[z] > 0).sum() for z in range(label_array.shape[0])]
        return label_array[int(np.argmax(areas))]
    if label_array.ndim == 4:
        return select_2d_mask(label_array.max(axis=-1))
    raise ValueError(f"Unsupported label shape: {label_array.shape}")


def read_label_volume(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "SimpleITK is required to import Slicer labels. "
            "Run this command inside the lung_app environment."
        ) from error
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


class SlicerRefinementPipeline:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.input = self.config["input"]
        self.refinement = self.config["slicer_refinement"]
        self.qc_report_path = Path(self.config["report"]["qc_report_path"])

    def _path(self, key: str) -> Path:
        return Path(self.refinement[key])

    def _metadata_rows(self) -> list[dict[str, str]]:
        rows = read_csv(self.input["manifest_path"])
        require_columns(
            rows,
            {
                self.input.get("filename_column", "new_filename"),
                self.input.get("original_path_column", "original_path"),
                self.input.get("split_column", "split"),
                self.input.get("class_column", "class"),
            },
            source="metadata",
        )
        return rows

    def _qc_rows(self) -> list[dict[str, str]]:
        rows = read_csv(self.qc_report_path)
        require_columns(rows, {"filename", "qc_status"}, source="QC report")
        return rows

    def prepare_studies(self) -> dict[str, int]:
        images_dir = Path(self.input["images_dir"])
        masks_dir = self._path("predicted_masks_dir")
        crops_dir = self._path("predicted_crops_dir")
        overlays_dir = self._path("overlays_dir")
        studies_dir = self._path("studies_dir")
        study_masks_dir = self._path("study_predicted_masks_dir")
        study_crops_dir = self._path("study_crops_dir")
        study_overlays_dir = self._path("study_overlays_dir")
        pass_crops_dir = self._path("pass_crops_dir")
        for directory in (
            studies_dir,
            study_masks_dir,
            study_crops_dir,
            study_overlays_dir,
            pass_crops_dir,
            self._path("labels_dir"),
        ):
            directory.mkdir(parents=True, exist_ok=True)

        pass_status = str(self.refinement.get("pass_status", "PASS"))
        manifest_rows = []
        counts = Counter()
        for row in self._qc_rows():
            filename = row["filename"]
            stem = Path(filename).stem
            status = row["qc_status"]
            crop_name = f"{stem}_crop.png"
            if status == pass_status:
                source = crops_dir / crop_name
                if source.exists():
                    shutil.copy2(source, pass_crops_dir / crop_name)
                    counts["pass_crops"] += 1
                else:
                    counts["missing_pass_crops"] += 1
                continue

            assets = {
                "image": (images_dir / filename, studies_dir / filename),
                "predicted_mask": (
                    masks_dir / f"{stem}_mask.png",
                    study_masks_dir / f"{stem}_mask.png",
                ),
                "crop": (crops_dir / crop_name, study_crops_dir / crop_name),
                "overlay": (
                    overlays_dir / f"{stem}_overlay.png",
                    study_overlays_dir / f"{stem}_overlay.png",
                ),
            }
            copied = {}
            for name, (source, destination) in assets.items():
                if source.exists():
                    shutil.copy2(source, destination)
                    copied[name] = str(destination.resolve())
                    counts[f"copied_{name}"] += 1
                else:
                    copied[name] = ""
                    counts[f"missing_{name}"] += 1
            manifest_rows.append(
                {
                    "filename": filename,
                    "stem": stem,
                    "qc_status": status,
                    "study_image_path": copied["image"],
                    "predicted_mask_path": copied["predicted_mask"],
                    "overlay_path": copied["overlay"],
                    "expected_label_filename": f"{stem}.nii.gz",
                }
            )
        write_csv(self._path("studies_manifest_path"), manifest_rows)
        counts["studies"] = len(manifest_rows)
        return dict(counts)

    def _find_source_image(self, stem: str) -> Path | None:
        studies_dir = self._path("studies_dir")
        for suffix in IMAGE_EXTENSIONS:
            candidate = studies_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def import_labels(
        self,
        *,
        label_reader: Callable[[Path], np.ndarray] = read_label_volume,
        flip_vertical: bool | None = None,
    ) -> dict[str, int]:
        labels_dir = self._path("labels_dir")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Slicer labels folder not found: {labels_dir}")
        corrected_images_dir = self._path("corrected_images_dir")
        corrected_masks_dir = self._path("corrected_masks_dir")
        corrected_images_dir.mkdir(parents=True, exist_ok=True)
        corrected_masks_dir.mkdir(parents=True, exist_ok=True)

        label_paths = sorted(
            [
                path
                for path in labels_dir.iterdir()
                if path.is_file()
                and any(path.name.lower().endswith(ext) for ext in LABEL_EXTENSIONS)
            ],
            key=lambda path: path.name.lower(),
        )
        if not label_paths:
            raise RuntimeError(f"No Slicer label files found in: {labels_dir}")

        if flip_vertical is None:
            flip_vertical = bool(self.refinement.get("flip_vertical", True))
        counts = Counter()
        rows = []
        for label_path in label_paths:
            stem = normalize_label_stem(strip_label_extension(label_path))
            source_image = self._find_source_image(stem)
            if source_image is None:
                counts["missing_source_image"] += 1
                rows.append(
                    {
                        "label_path": str(label_path.resolve()),
                        "stem": stem,
                        "status": "MISSING_SOURCE_IMAGE",
                    }
                )
                continue

            image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
            if image is None:
                counts["unreadable_source_image"] += 1
                continue
            mask = (select_2d_mask(label_reader(label_path)) > 0).astype(np.uint8)
            output_mask = corrected_masks_dir / f"{stem}_mask.png"
            if not np.any(mask):
                output_mask.unlink(missing_ok=True)
                counts["empty_label"] += 1
                rows.append(
                    {
                        "label_path": str(label_path.resolve()),
                        "stem": stem,
                        "status": "EMPTY_LABEL",
                        "corrected_mask_path": "",
                    }
                )
                continue
            if flip_vertical:
                mask = np.flipud(mask)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                counts["resized"] += 1
            if not cv2.imwrite(str(output_mask), mask * 255):
                raise RuntimeError(f"Failed to write corrected mask: {output_mask}")
            shutil.copy2(source_image, corrected_images_dir / source_image.name)
            counts["converted"] += 1
            rows.append(
                {
                    "label_path": str(label_path.resolve()),
                    "stem": stem,
                    "status": "OK",
                    "corrected_mask_path": str(output_mask.resolve()),
                }
            )
        write_csv(corrected_masks_dir.parent / "label_import_report.csv", rows)
        return dict(counts)

    def merge_masks(self) -> dict[str, int]:
        metadata = self._metadata_rows()
        filename_column = self.input.get("filename_column", "new_filename")
        original_column = self.input.get("original_path_column", "original_path")
        split_column = self.input.get("split_column", "split")
        class_column = self.input.get("class_column", "class")
        qc_by_filename = {
            row["filename"]: row["qc_status"] for row in self._qc_rows()
        }
        predicted_masks_dir = self._path("predicted_masks_dir")
        corrected_masks_dir = self._path("corrected_masks_dir")
        final_masks_dir = self._path("final_masks_dir")
        final_masks_dir.mkdir(parents=True, exist_ok=True)

        counts = Counter()
        rows = []
        pass_status = str(self.refinement.get("pass_status", "PASS"))
        for row in metadata:
            filename = row[filename_column]
            mask_name = f"{Path(filename).stem}_mask.png"
            corrected = corrected_masks_dir / mask_name
            predicted = predicted_masks_dir / mask_name
            final = final_masks_dir / mask_name
            qc_status = qc_by_filename.get(filename, "")
            source = corrected if corrected.exists() else predicted
            mask_source = (
                "corrected"
                if corrected.exists()
                else "predicted"
                if predicted.exists()
                else "missing"
            )
            if source.exists():
                shutil.copy2(source, final)
                merge_status = "OK"
                counts[mask_source] += 1
                if qc_status and qc_status != pass_status and mask_source == "predicted":
                    merge_status = "WARNING_FAIL_USED_PREDICTED"
                    counts["fail_without_correction"] += 1
            else:
                merge_status = "MISSING_MASK"
                counts["missing"] += 1
            rows.append(
                {
                    "filename": filename,
                    "original_path": row[original_column],
                    "split": row[split_column],
                    "class": row[class_column],
                    "qc_status": qc_status,
                    "mask_source": mask_source,
                    "merge_status": merge_status,
                    "source_mask_path": str(source.resolve()) if source.exists() else "",
                    "final_mask_path": str(final.resolve()) if final.exists() else "",
                }
            )
        write_csv(self._path("merge_report_path"), rows)
        counts["total"] = len(rows)
        return dict(counts)

    def create_final_dataset(self) -> dict[str, int]:
        merge_rows = read_csv(self._path("merge_report_path"))
        require_columns(
            merge_rows,
            {"filename", "split", "class", "mask_source", "merge_status"},
            source="merge report",
        )
        images_dir = Path(self.input["images_dir"])
        final_masks_dir = self._path("final_masks_dir")
        final_dataset_dir = self._path("final_dataset_dir")
        failed_crops_dir = self._path("failed_crops_dir")
        failed_crops_dir.mkdir(parents=True, exist_ok=True)
        crop_config = self.refinement.get("crop", {})

        counts = Counter()
        rows = []
        for row in merge_rows:
            filename = row["filename"]
            stem = Path(filename).stem
            image_path = images_dir / filename
            mask_path = final_masks_dir / f"{stem}_mask.png"
            output_path = final_dataset_dir / row["split"] / row["class"] / f"{stem}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            bbox = None
            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
                if mask is None:
                    raise ValueError(f"Could not read mask: {mask_path}")
                mask = (mask > 0).astype(np.uint8)
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                cropped, bbox = crop_by_mask(image, mask, **crop_config)
                if cropped is None:
                    raise RuntimeError("Mask is empty or invalid")
                if not cv2.imwrite(str(output_path), cropped):
                    raise RuntimeError(f"Failed to write crop: {output_path}")
                crop_status = "OK"
                counts["saved"] += 1
            except Exception as error:
                crop_status = f"FAIL: {error}"
                counts["failed"] += 1
                if image_path.exists():
                    shutil.copy2(image_path, failed_crops_dir / filename)
            report = {
                "filename": filename,
                "split": row["split"],
                "class": row["class"],
                "mask_source": row["mask_source"],
                "merge_status": row["merge_status"],
                "crop_status": crop_status,
                "output_path": str(output_path.resolve()) if output_path.exists() else "",
            }
            bbox_columns = {
                "bbox_x1": "x1",
                "bbox_y1": "y1",
                "bbox_x2": "x2",
                "bbox_y2": "y2",
                "bbox_w": "bbox_w",
                "bbox_h": "bbox_h",
                "bottom_ratio": "bottom_ratio",
            }
            for column, key in bbox_columns.items():
                report[column] = bbox[key] if bbox else ""
            rows.append(report)
        write_csv(self._path("final_dataset_report_path"), rows)
        counts["total"] = len(rows)
        return dict(counts)

    def status(self) -> dict[str, Any]:
        qc_rows = self._qc_rows()
        pass_status = str(self.refinement.get("pass_status", "PASS"))
        review_rows = [row for row in qc_rows if row["qc_status"] != pass_status]
        label_stems = {
            normalize_label_stem(strip_label_extension(path))
            for path in self._path("labels_dir").glob("*")
            if path.is_file()
            and any(path.name.lower().endswith(ext) for ext in LABEL_EXTENSIONS)
        }
        corrected_stems = {
            path.name[: -len("_mask.png")]
            for path in self._path("corrected_masks_dir").glob("*_mask.png")
        }
        import_report_path = self._path("corrected_masks_dir").parent / (
            "label_import_report.csv"
        )
        import_rows = (
            read_csv(import_report_path) if import_report_path.exists() else []
        )
        invalid_labels = sorted(
            row["stem"]
            for row in import_rows
            if row.get("status") not in {"", "OK"}
        )
        expected_stems = {Path(row["filename"]).stem for row in review_rows}
        final_report_path = self._path("final_dataset_report_path")
        final_rows = read_csv(final_report_path) if final_report_path.exists() else []
        status = {
            "total_images": len(qc_rows),
            "pass_images": len(qc_rows) - len(review_rows),
            "review_required": len(review_rows),
            "slicer_labels": len(label_stems),
            "corrected_masks": len(corrected_stems),
            "missing_labels": sorted(expected_stems - label_stems),
            "missing_corrected_masks": sorted(expected_stems - corrected_stems),
            "invalid_labels": invalid_labels,
            "final_crops": sum(
                1 for row in final_rows if row.get("crop_status") == "OK"
            ),
        }
        status["ready_for_merge"] = not status["missing_corrected_masks"]
        status["complete"] = (
            status["ready_for_merge"]
            and status["final_crops"] == status["total_images"]
        )
        status_path = self._path("status_path")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status
