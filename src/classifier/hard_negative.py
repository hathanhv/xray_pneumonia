from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.classifier.dataset import (
    ManifestClassificationDataset,
    build_tensor_transform,
    classification_collate,
)
from src.classifier.prediction import collect_logits, positive_probabilities


@dataclass(frozen=True)
class HardNegativeResult:
    indices: tuple[int, ...]
    probabilities: tuple[float, ...]


def create_evaluation_dataset(dataset):
    return ManifestClassificationDataset(
        dataset.records,
        preprocessing=dataset.preprocessing,
        image_transform=dataset.image_transform,
        return_metadata=True,
        validate=False,
    )


def mine_hard_negatives(
    model,
    dataset,
    *,
    device,
    threshold=0.3,
    negative_class=0,
    positive_class=1,
    batch_size=32,
    num_workers=0,
):
    evaluation_dataset = create_evaluation_dataset(dataset)
    loader = DataLoader(
        evaluation_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=classification_collate,
    )
    logits, targets, _metadata = collect_logits(model, loader, device)
    probabilities = positive_probabilities(logits, positive_class)
    mask = (targets == int(negative_class)) & (
        probabilities > float(threshold)
    )
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    return HardNegativeResult(
        indices=tuple(int(index) for index in indices.tolist()),
        probabilities=tuple(
            float(probabilities[index].item()) for index in indices
        ),
    )


def hard_negative_sampler_config(
    result,
    *,
    dataset_size,
    oversample_factor=3,
):
    return {
        "name": "hard_negative",
        "hard_negative_indices": list(result.indices),
        "multiplier": float(oversample_factor) + 1.0,
        "num_samples": int(dataset_size)
        + int(len(result.indices) * oversample_factor),
    }
