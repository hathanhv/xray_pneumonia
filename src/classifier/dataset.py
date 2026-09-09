from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.classifier.augmentations import build_augmentation
from src.classifier.preprocessing import RawPreprocessing, build_preprocessing
from src.classifier.samplers import build_sampler
from src.core.reproducibility import create_torch_generator, seed_worker


CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ClassificationRecord:
    image_path: Path
    label: int
    class_name: str
    split: str = ""
    sample_id: str = ""
    patient_id: str = ""
    mask_path: Path | None = None
    refined_mask_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        result = dict(self.metadata)
        result.update(
            {
                "image_path": str(self.image_path),
                "label": self.label,
                "class_name": self.class_name,
                "split": self.split,
                "sample_id": self.sample_id or self.image_path.stem,
                "patient_id": self.patient_id or self.sample_id or self.image_path.stem,
            }
        )
        if self.mask_path:
            result["mask_path"] = str(self.mask_path)
        if self.refined_mask_path:
            result["refined_mask_path"] = str(self.refined_mask_path)
        return result


class ManifestClassificationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ClassificationRecord],
        *,
        preprocessing=None,
        augmentation=None,
        image_transform=None,
        return_metadata: bool = True,
        validate: bool = True,
    ):
        self.records = list(records)
        if validate:
            validate_records(self.records)
        self.preprocessing = preprocessing or RawPreprocessing()
        self.augmentation = augmentation or transforms.Compose([])
        self.image_transform = image_transform or build_tensor_transform()
        self.return_metadata = return_metadata
        self.targets = [record.label for record in self.records]
        self.class_to_idx = dict(CLASS_TO_IDX)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        result = self.preprocessing(image, record.as_metadata())
        image = self.augmentation(result.image)
        tensor = self.image_transform(image)
        if not self.return_metadata:
            return tensor, record.label
        metadata = record.as_metadata()
        metadata.update(result.metadata)
        return {"image": tensor, "label": record.label, "metadata": metadata}


def get_train_transforms(img_size=224):
    """Legacy notebook transform. Keep unchanged for checkpoint parity."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(img_size=224):
    """Legacy notebook transform. Keep unchanged for checkpoint parity."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_tensor_transform(*, grayscale=True, normalize=True):
    operations = []
    if grayscale:
        operations.append(transforms.Grayscale(num_output_channels=3))
    operations.append(transforms.ToTensor())
    if normalize:
        operations.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(operations)


def create_imagefolder(root_dir, transform):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root_dir}")
    dataset = datasets.ImageFolder(root=str(root_dir), transform=transform)
    if dataset.class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            f"Unexpected class mapping: {dataset.class_to_idx}. Expected: {CLASS_TO_IDX}"
        )
    return dataset


def load_manifest(
    manifest_path: str | Path,
    *,
    root_dir: str | Path | None = None,
    class_to_idx: Mapping[str, int] = CLASS_TO_IDX,
) -> list[ClassificationRecord]:
    manifest_path = Path(manifest_path)
    root = Path(root_dir) if root_dir else manifest_path.parent
    records = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=2):
            raw_path = _first(row, "image_path", "path", "output_path", "filepath")
            if not raw_path:
                raise ValueError(f"Missing image path at {manifest_path}:{row_index}")
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = root / image_path

            class_value = _first(row, "class_name", "class", "label")
            if class_value is None:
                raise ValueError(f"Missing label at {manifest_path}:{row_index}")
            if str(class_value).strip().lstrip("-").isdigit():
                label = int(class_value)
                idx_to_class = {value: key for key, value in class_to_idx.items()}
                if label not in idx_to_class:
                    raise ValueError(f"Unknown numeric label {label} at row {row_index}")
                class_name = idx_to_class[label]
            else:
                class_name = str(class_value).strip().upper()
                if class_name not in class_to_idx:
                    raise ValueError(f"Unknown class {class_name!r} at row {row_index}")
                label = class_to_idx[class_name]

            records.append(
                ClassificationRecord(
                    image_path=image_path,
                    label=label,
                    class_name=class_name,
                    split=str(row.get("split", "")).strip().lower(),
                    sample_id=str(row.get("sample_id") or image_path.stem),
                    patient_id=str(row.get("patient_id") or row.get("subject_id") or ""),
                    mask_path=_optional_path(row.get("mask_path"), root),
                    refined_mask_path=_optional_path(row.get("refined_mask_path"), root),
                    metadata={key: value for key, value in row.items() if value != ""},
                )
            )
    return records


def records_from_imagefolder(
    data_dir: str | Path,
    *,
    class_to_idx: Mapping[str, int] = CLASS_TO_IDX,
) -> list[ClassificationRecord]:
    data_dir = Path(data_dir)
    records = []
    split_dirs = [path for path in data_dir.iterdir() if path.is_dir()]
    has_split_dirs = any(path.name.lower() in {"train", "val", "test"} for path in split_dirs)
    roots = split_dirs if has_split_dirs else [data_dir]
    for split_root in sorted(roots):
        split = split_root.name.lower() if has_split_dirs else ""
        for class_name, label in class_to_idx.items():
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    records.append(
                        ClassificationRecord(
                            image_path=image_path,
                            label=label,
                            class_name=class_name,
                            split=split,
                            sample_id=image_path.stem,
                        )
                    )
    if not records:
        raise ValueError(f"No classification images found in {data_dir}")
    return records


def validate_records(
    records: Sequence[ClassificationRecord],
    *,
    check_images: bool = True,
) -> None:
    if not records:
        raise ValueError("Dataset has no records")
    errors = []
    for record in records:
        if record.class_name not in CLASS_TO_IDX:
            errors.append(f"unknown class {record.class_name}: {record.image_path}")
        elif CLASS_TO_IDX[record.class_name] != record.label:
            errors.append(f"class mapping mismatch: {record.image_path}")
        if not record.image_path.exists():
            errors.append(f"missing image: {record.image_path}")
        elif check_images:
            try:
                with Image.open(record.image_path) as handle:
                    handle.verify()
            except Exception as error:
                errors.append(f"corrupt image: {record.image_path} ({error})")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Dataset validation failed ({len(errors)} error(s)):\n{preview}")


def split_records(
    records: Sequence[ClassificationRecord],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    patient_level: bool = True,
    preserve_existing_test: bool = True,
) -> dict[str, list[ClassificationRecord]]:
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    existing_test = [record for record in records if record.split == "test"]
    candidates = [
        record
        for record in records
        if not (preserve_existing_test and record.split == "test")
    ]
    grouped: dict[tuple[int, str], list[ClassificationRecord]] = defaultdict(list)
    for record in candidates:
        group_id = (
            record.patient_id or record.sample_id or record.image_path.stem
            if patient_level
            else record.sample_id or record.image_path.stem
        )
        grouped[(record.label, group_id)].append(record)

    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": list(existing_test)}
    by_class: dict[int, list[list[ClassificationRecord]]] = defaultdict(list)
    for (label, _), group_records in grouped.items():
        by_class[label].append(group_records)

    for groups in by_class.values():
        rng.shuffle(groups)
        test_count = 0 if preserve_existing_test and existing_test else round(
            len(groups) * test_fraction
        )
        val_count = round(len(groups) * val_fraction)
        if len(groups) >= 3 and val_fraction > 0:
            val_count = max(1, val_count)
        for index, group in enumerate(groups):
            split = "test" if index < test_count else (
                "val" if index < test_count + val_count else "train"
            )
            result[split].extend(replace(record, split=split) for record in group)

    result["test"] = [
        replace(record, split="test") for record in result["test"]
    ]
    assert_no_patient_leakage(result)
    if not result["val"]:
        raise ValueError("Validation split is empty; increase data or val_fraction")
    return result


def sklearn_stratified_split_records(
    records: Sequence[ClassificationRecord],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    preserve_existing_test: bool = True,
):
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for the NB04-compatible split"
        ) from error

    existing_test = [record for record in records if record.split == "test"]
    candidates = [
        record
        for record in records
        if not (preserve_existing_test and record.split == "test")
    ]
    indices = np.arange(len(candidates))
    labels = [record.label for record in candidates]
    class_counts = {
        label: labels.count(label) for label in sorted(set(labels))
    }
    if any(count < 2 for count in class_counts.values()):
        raise ValueError(
            "Stratified train/validation split requires at least 2 samples "
            f"per class; counts={class_counts}"
        )
    minimum_val_size = len(class_counts)
    requested_val_size = max(
        minimum_val_size,
        int(math.ceil(len(candidates) * float(val_fraction))),
    )
    maximum_val_size = len(candidates) - len(class_counts)
    if requested_val_size > maximum_val_size:
        raise ValueError(
            "Dataset is too small for a stratified train/validation split "
            f"with every class in both splits; counts={class_counts}"
        )
    train_indices, val_indices = train_test_split(
        indices,
        test_size=requested_val_size,
        stratify=labels,
        random_state=seed,
    )
    return {
        "train": [replace(candidates[index], split="train") for index in train_indices],
        "val": [replace(candidates[index], split="val") for index in val_indices],
        "test": [replace(record, split="test") for record in existing_test],
    }


def assert_no_patient_leakage(splits: Mapping[str, Sequence[ClassificationRecord]]):
    ownership = {}
    for split, records in splits.items():
        for record in records:
            patient_id = record.patient_id
            if not patient_id:
                continue
            previous = ownership.setdefault(patient_id, split)
            if previous != split:
                raise ValueError(
                    f"Patient leakage detected: {patient_id} in {previous} and {split}"
                )


def save_manifest(
    records: Iterable[ClassificationRecord],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "label",
        "class_name",
        "split",
        "sample_id",
        "patient_id",
        "mask_path",
        "refined_mask_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": record.image_path,
                    "label": record.label,
                    "class_name": record.class_name,
                    "split": record.split,
                    "sample_id": record.sample_id,
                    "patient_id": record.patient_id,
                    "mask_path": record.mask_path or "",
                    "refined_mask_path": record.refined_mask_path or "",
                }
            )
    return output_path


def create_datasets_from_config(config: Mapping[str, Any]):
    dataset_config = config["dataset"]
    split_config = dataset_config.get("split", {})
    fixed_manifest = split_config.get("manifest_output")
    if fixed_manifest and Path(fixed_manifest).exists():
        records = load_manifest(fixed_manifest)
    elif dataset_config.get("manifest_path"):
        records = load_manifest(
            dataset_config["manifest_path"],
            root_dir=dataset_config.get("root_dir"),
        )
    else:
        records = records_from_imagefolder(dataset_config["data_dir"])

    available = {record.split for record in records}
    if not {"train", "val", "test"}.issubset(available):
        if split_config.get("method") == "sklearn_stratified":
            splits = sklearn_stratified_split_records(
                records,
                val_fraction=split_config.get("val_fraction", 0.2),
                seed=split_config.get("seed", config.get("seed", 42)),
                preserve_existing_test=split_config.get(
                    "preserve_existing_test",
                    True,
                ),
            )
        else:
            splits = split_records(
                records,
                val_fraction=split_config.get("val_fraction", 0.15),
                test_fraction=split_config.get("test_fraction", 0.15),
                seed=split_config.get("seed", config.get("seed", 42)),
                patient_level=split_config.get("patient_level", True),
                preserve_existing_test=split_config.get(
                    "preserve_existing_test",
                    True,
                ),
            )
        manifest_output = split_config.get("manifest_output")
        if manifest_output:
            save_manifest(
                [record for split in splits.values() for record in split],
                manifest_output,
            )
    else:
        splits = {
            name: [record for record in records if record.split == name]
            for name in ("train", "val", "test")
        }
        assert_no_patient_leakage(splits)

    few_shot = dataset_config.get("few_shot")
    if few_shot and few_shot.get("enabled", True):
        splits["train"] = select_few_shot_records(
            splits["train"],
            shots_per_class=int(few_shot.get("shots_per_class", 5)),
            seed=int(few_shot.get("seed", config.get("seed", 42))),
        )

    preprocessing = build_preprocessing(config.get("preprocessing"))
    augmentation_config = config.get("augmentation", {"name": "baseline"})
    tensor_transform = build_tensor_transform(
        grayscale=config.get("input", {}).get("grayscale", True),
        normalize=config.get("input", {}).get("normalize", True),
    )
    return {
        split: ManifestClassificationDataset(
            split_records_,
            preprocessing=preprocessing,
            augmentation=build_augmentation(
                augmentation_config,
                training=split == "train",
            ),
            image_transform=tensor_transform,
            return_metadata=config.get("return_metadata", True),
        )
        for split, split_records_ in splits.items()
    }


def create_loaders_from_config(config: Mapping[str, Any]):
    datasets_ = create_datasets_from_config(config)
    loader_config = config.get("dataloader", {})
    generator = create_torch_generator(config.get("seed", 42))
    loaders = {}
    for split, dataset in datasets_.items():
        sampler = None
        if split == "train":
            sampler = build_sampler(
                dataset.targets,
                config.get("sampler", {"name": "random"}),
                generator=generator,
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=loader_config.get("batch_size", 16),
            shuffle=False,
            sampler=sampler,
            num_workers=loader_config.get("num_workers", 0),
            pin_memory=loader_config.get("pin_memory", True),
            worker_init_fn=seed_worker,
            generator=generator,
            collate_fn=classification_collate,
        )
    return loaders, datasets_


def select_few_shot_records(
    records: Sequence[ClassificationRecord],
    *,
    shots_per_class: int,
    seed: int = 42,
) -> list[ClassificationRecord]:
    if shots_per_class < 1:
        raise ValueError("shots_per_class must be at least 1")
    by_class: dict[int, list[ClassificationRecord]] = defaultdict(list)
    for record in records:
        by_class[record.label].append(record)
    missing = set(CLASS_TO_IDX.values()) - set(by_class)
    if missing:
        raise ValueError(f"Few-shot training is missing class labels: {sorted(missing)}")
    rng = random.Random(seed)
    selected = []
    for label in sorted(by_class):
        candidates = sorted(
            by_class[label],
            key=lambda record: (record.sample_id, str(record.image_path)),
        )
        rng.shuffle(candidates)
        selected.extend(candidates[: min(shots_per_class, len(candidates))])
    return sorted(selected, key=lambda record: (record.label, record.sample_id))


def create_dataloaders(
    data_dir,
    img_size=224,
    batch_size=16,
    num_workers=0,
    seed=42,
):
    """Backward-compatible legacy loader. Prefer create_loaders_from_config."""
    data_dir = Path(data_dir)
    generator = create_torch_generator(seed)
    train_dataset = create_imagefolder(
        data_dir / "train",
        transform=get_train_transforms(img_size),
    )
    test_dataset = create_imagefolder(
        data_dir / "test",
        transform=get_eval_transforms(img_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, test_loader, train_dataset, test_dataset


def unpack_batch(batch):
    if isinstance(batch, Mapping):
        return batch["image"], batch["label"], batch.get("metadata")
    images, labels = batch
    return images, labels, None


def classification_collate(samples):
    if not samples:
        return {"image": torch.empty(0), "label": torch.empty(0), "metadata": []}
    if not isinstance(samples[0], Mapping):
        return torch.utils.data.default_collate(samples)
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "label": torch.as_tensor([sample["label"] for sample in samples]),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }


def _first(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _optional_path(value, root):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path
