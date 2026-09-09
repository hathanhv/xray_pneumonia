from __future__ import annotations

import torch


@torch.no_grad()
def compute_hubris(model, dataloader, device):
    """Return E[|P(PNEUMONIA)-0.5|] and the per-sample probabilities."""
    model.eval()
    probabilities = []
    for batch in dataloader:
        if isinstance(batch, dict):
            images = batch["image"]
        else:
            images = batch[0]
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    probabilities = (
        torch.cat(probabilities)
        if probabilities
        else torch.empty(0, dtype=torch.float32)
    )
    score = (
        float(torch.abs(probabilities - 0.5).mean().item())
        if len(probabilities)
        else 0.0
    )
    return score, probabilities
