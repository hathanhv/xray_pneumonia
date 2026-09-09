from __future__ import annotations

from typing import Any, Mapping

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.classifier.dataset import classification_collate
from src.classifier.prediction import collect_logits


class IdentityView:
    def __call__(self, image):
        return image


class HorizontalFlipView:
    def __call__(self, image):
        return TF.hflip(image)


class RotateView:
    def __init__(self, angle):
        self.angle = float(angle)

    def __call__(self, image):
        return TF.rotate(image, self.angle)


class BrightnessView:
    def __init__(self, factor):
        self.factor = float(factor)

    def __call__(self, image):
        return ImageEnhance.Brightness(image).enhance(self.factor)


class _TTAViewDataset(Dataset):
    def __init__(self, dataset, view):
        self.dataset = dataset
        self.view = view

    def __len__(self):
        return len(self.dataset.records)

    def __getitem__(self, index):
        record = self.dataset.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        result = self.dataset.preprocessing(image, record.as_metadata())
        image = self.view(result.image)
        tensor = self.dataset.image_transform(image)
        return {
            "image": tensor,
            "label": record.label,
            "metadata": record.as_metadata(),
        }


def build_tta_views(
    config: Mapping[str, Any] | None = None,
    *,
    image_size=224,
):
    config = dict(config or {})
    rotation = float(config.get("rotation", 5.0))
    bright = float(config.get("brightness_delta", 0.1))
    scale = float(config.get("scale", 1.1))
    scaled_size = max(int(round(image_size * scale)), image_size)
    return [
        IdentityView(),
        HorizontalFlipView(),
        RotateView(rotation),
        RotateView(-rotation),
        BrightnessView(1.0 + bright),
        BrightnessView(1.0 - bright),
        transforms.Compose(
            [
                transforms.Resize((scaled_size, scaled_size)),
                transforms.CenterCrop(image_size),
            ]
        ),
        transforms.Compose(
            [
                HorizontalFlipView(),
                RotateView(rotation),
            ]
        ),
    ]


def predict_tta(
    model,
    dataset,
    *,
    device,
    batch_size=32,
    num_workers=0,
    views=None,
    image_size=224,
):
    views = views or build_tta_views(image_size=image_size)
    probability_sum = None
    targets_reference = None
    metadata_reference = None
    for view in views:
        loader = DataLoader(
            _TTAViewDataset(dataset, view),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=classification_collate,
        )
        logits, targets, metadata = collect_logits(model, loader, device)
        probabilities = torch.softmax(logits, dim=1)
        probability_sum = (
            probabilities
            if probability_sum is None
            else probability_sum + probabilities
        )
        if targets_reference is None:
            targets_reference = targets
            metadata_reference = metadata
        elif not torch.equal(targets_reference, targets):
            raise ValueError("TTA views changed sample order")
    return (
        probability_sum / len(views),
        targets_reference,
        metadata_reference,
    )
