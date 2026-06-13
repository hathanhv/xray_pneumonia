from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image


class BoundaryDataset(Dataset):
    """Boundary images with either oracle soft labels or NORMAL hard labels."""

    def __init__(
        self,
        image_dir,
        metadata_path,
        *,
        transform=None,
        max_confusion_distance=0.15,
        max_images=None,
        label_strategy="soft",
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_strategy = str(label_strategy).lower()
        if self.label_strategy not in {"soft", "hard_normal"}:
            raise ValueError(
                "label_strategy must be 'soft' or 'hard_normal'"
            )
        rows = _read_metadata(metadata_path)
        rows = [
            row
            for row in rows
            if float(row["confusion_distance"])
            < float(max_confusion_distance)
            and (self.image_dir / row["filename"]).exists()
        ]
        rows.sort(key=lambda row: float(row["confusion_distance"]))
        if max_images is not None:
            rows = rows[: int(max_images)]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.image_dir / row["filename"]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if self.transform:
            image = self.transform(image)
        probability = float(row["p_pneumonia"])
        if self.label_strategy == "soft":
            target = torch.tensor(
                [1.0 - probability, probability],
                dtype=torch.float32,
            )
        else:
            target = torch.tensor([1.0, 0.0], dtype=torch.float32)
        return image, target


class OneHotDatasetWrapper(Dataset):
    def __init__(self, dataset, num_classes=2):
        self.dataset = dataset
        self.num_classes = int(num_classes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if isinstance(sample, dict):
            image = sample["image"]
            label = int(sample["label"])
        else:
            image, label = sample[:2]
            label = int(label)
        target = F.one_hot(
            torch.tensor(label),
            num_classes=self.num_classes,
        ).float()
        return image, target


@torch.no_grad()
def generate_boundary_images(
    generator,
    oracle,
    *,
    output_dir,
    count,
    latent_dim,
    device,
    ambiguity_threshold=0.20,
    batch_size=64,
    max_attempt_multiplier=30,
    generator_checkpoint="",
    oracle_checkpoint="",
):
    output_dir = Path(output_dir)
    image_dir = output_dir / "NORMAL"
    image_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    oracle.eval()
    rows = []
    attempted = 0
    maximum_attempts = int(count) * int(max_attempt_multiplier)
    while len(rows) < int(count) and attempted < maximum_attempts:
        current_batch = min(int(batch_size), maximum_attempts - attempted)
        noise = torch.randn(current_batch, int(latent_dim), device=device)
        images = generator(noise)
        probabilities = oracle(images, temperature=1.0)
        for batch_index, (image, probability) in enumerate(
            zip(images.cpu(), probabilities.cpu())
        ):
            if len(rows) >= int(count):
                break
            probability = float(probability.item())
            distance = abs(probability - 0.5)
            if distance >= float(ambiguity_threshold):
                continue
            sample_index = len(rows)
            filename = f"boundary_{sample_index:06d}.png"
            save_image(
                image,
                image_dir / filename,
                normalize=True,
                value_range=(-1, 1),
            )
            rows.append(
                {
                    "filename": filename,
                    "p_pneumonia": probability,
                    "confusion_distance": distance,
                    "attempt_index": attempted + batch_index,
                    "generator_checkpoint": str(generator_checkpoint),
                    "oracle_checkpoint": str(oracle_checkpoint),
                }
            )
        attempted += current_batch

    metadata_path = output_dir / "metadata.csv"
    _write_metadata(rows, metadata_path)
    return {
        "saved_count": len(rows),
        "attempt_count": attempted,
        "acceptance_rate": len(rows) / max(attempted, 1),
        "image_dir": str(image_dir),
        "metadata_path": str(metadata_path),
        "rows": rows,
    }


def _read_metadata(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "confusion_distance" not in row and "confusion_dist" in row:
            row["confusion_distance"] = row["confusion_dist"]
    required = {"filename", "p_pneumonia", "confusion_distance"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(
            f"Boundary metadata requires columns: {sorted(required)}"
        )
    return rows


def _write_metadata(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "p_pneumonia",
        "confusion_distance",
        "attempt_index",
        "generator_checkpoint",
        "oracle_checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
