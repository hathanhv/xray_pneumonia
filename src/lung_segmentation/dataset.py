from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.reproducibility import create_torch_generator, seed_worker


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MANIFEST_FIELDS = ("sample_id", "image_path", "mask_path", "split")


@dataclass(frozen=True)
class SegmentationRecord:
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str


def _image_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _mask_key(path: Path, suffix: str) -> str:
    stem = path.stem
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def _split_names(count: int, val_fraction: float, test_fraction: float, seed: int):
    if count < 3:
        raise ValueError("At least 3 image-mask pairs are required for train/val/test.")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be > 0 and sum to < 1.")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    val_count = max(1, round(count * val_fraction))
    test_count = max(1, round(count * test_fraction))
    while val_count + test_count >= count:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Not enough samples to create non-empty splits.")
    labels = ["train"] * count
    for index in indices[:val_count]:
        labels[index] = "val"
    for index in indices[val_count : val_count + test_count]:
        labels[index] = "test"
    return labels


def build_segmentation_manifest(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_path: str | Path,
    *,
    mask_suffix: str = "_mask",
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> list[SegmentationRecord]:
    images = _image_files(image_dir)
    masks = _image_files(mask_dir)
    mask_lookup = {}
    for mask_path in masks:
        key = _mask_key(mask_path, mask_suffix)
        if key in mask_lookup:
            raise ValueError(f"Duplicate mask key '{key}': {mask_path}")
        mask_lookup[key] = mask_path.resolve()

    pairs = []
    missing_masks = []
    for image_path in images:
        mask_path = mask_lookup.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(image_path.name)
        else:
            pairs.append((image_path.resolve(), mask_path))
    if missing_masks:
        preview = ", ".join(missing_masks[:5])
        raise ValueError(f"Missing masks for {len(missing_masks)} images: {preview}")
    if not pairs:
        raise ValueError("No matching image-mask pairs found.")

    split_names = _split_names(len(pairs), val_fraction, test_fraction, seed)
    records = [
        SegmentationRecord(image_path.stem, image_path, mask_path, split)
        for (image_path, mask_path), split in zip(pairs, split_names)
    ]
    write_segmentation_manifest(records, output_path)
    return records


def write_segmentation_manifest(
    records: Iterable[SegmentationRecord],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "image_path": str(record.image_path),
                    "mask_path": str(record.mask_path),
                    "split": record.split,
                }
            )
    return output_path


def load_segmentation_manifest(
    manifest_path: str | Path,
    *,
    split: str | None = None,
) -> list[SegmentationRecord]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Segmentation manifest not found: {manifest_path}")
    records = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            row_split = row["split"].strip().lower()
            if row_split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split '{row_split}' in {manifest_path}")
            if split is not None and row_split != split:
                continue
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            if not mask_path.is_absolute():
                mask_path = manifest_path.parent / mask_path
            if not image_path.exists() or not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing pair for sample {row['sample_id']}: "
                    f"{image_path}, {mask_path}"
                )
            records.append(
                SegmentationRecord(
                    row["sample_id"],
                    image_path.resolve(),
                    mask_path.resolve(),
                    row_split,
                )
            )
    if split is not None and not records:
        raise ValueError(f"Manifest has no samples for split '{split}'.")
    return records


class LungSegmentationDataset(Dataset):
    def __init__(
        self,
        records: Iterable[SegmentationRecord],
        *,
        image_size: int = 256,
        mask_threshold: int = 127,
        augmentation: Mapping | None = None,
    ):
        self.records = list(records)
        self.image_size = int(image_size)
        self.mask_threshold = int(mask_threshold)
        self.augmentation = dict(augmentation or {})

    def __len__(self):
        return len(self.records)

    def _augment(self, image, mask):
        if random.random() < float(self.augmentation.get("horizontal_flip", 0.0)):
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))
        max_rotation = float(self.augmentation.get("rotation_degrees", 0.0))
        if max_rotation > 0:
            angle = random.uniform(-max_rotation, max_rotation)
            center = (self.image_size / 2.0, self.image_size / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (self.image_size, self.image_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (self.image_size, self.image_size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return image, mask

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {record.image_path}")
        if mask is None:
            raise ValueError(f"Cannot read mask: {record.mask_path}")
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        image, mask = self._augment(image, mask)
        image = image.astype(np.float32) / 255.0
        image = np.stack([image, image, image], axis=0)
        mask = (mask > self.mask_threshold).astype(np.float32)[None, ...]
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
        }


def create_segmentation_loaders(config: Mapping):
    dataset_config = config["dataset"]
    training_config = config["training"]
    manifest_path = Path(dataset_config["manifest_path"])
    image_size = int(config["model"].get("image_size", 256))
    seed = int(config.get("seed", config.get("reproducibility", {}).get("seed", 42)))
    datasets = {}
    loaders = {}
    for split in ("train", "val", "test"):
        augmentation = (
            dataset_config.get("augmentation", {}) if split == "train" else {}
        )
        dataset = LungSegmentationDataset(
            load_segmentation_manifest(manifest_path, split=split),
            image_size=image_size,
            mask_threshold=int(dataset_config.get("mask_threshold", 127)),
            augmentation=augmentation,
        )
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training_config.get("batch_size", 8)),
            shuffle=split == "train",
            num_workers=int(training_config.get("num_workers", 0)),
            pin_memory=bool(training_config.get("pin_memory", True)),
            worker_init_fn=seed_worker,
            generator=create_torch_generator(seed),
        )
    return loaders, datasets
