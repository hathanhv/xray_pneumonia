from __future__ import annotations

import torch

from src.classifier.dataset import unpack_batch


def collect_logits(model, dataloader, device):
    model.eval()
    logits_all = []
    targets_all = []
    metadata_all = []
    with torch.no_grad():
        for batch in dataloader:
            images, targets, metadata = unpack_batch(batch)
            logits_all.append(model(images.to(device)).detach().cpu())
            targets_all.append(targets.detach().cpu())
            metadata_all.extend(metadata or [{} for _ in range(len(targets))])
    logits = (
        torch.cat(logits_all)
        if logits_all
        else torch.empty((0, 2), dtype=torch.float32)
    )
    targets = (
        torch.cat(targets_all)
        if targets_all
        else torch.empty(0, dtype=torch.long)
    )
    return logits, targets, metadata_all


def positive_probabilities(logits, positive_class=1):
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape [samples, classes]")
    return torch.softmax(logits, dim=1)[:, int(positive_class)]
