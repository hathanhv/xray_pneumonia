from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler


def build_sampler(
    labels: Sequence[int],
    config: Mapping[str, Any] | str | None,
    *,
    generator: torch.Generator | None = None,
):
    if isinstance(config, str):
        config = {"name": config}
    config = dict(config or {"name": "random"})
    name = str(config.get("name", "random")).lower()
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)

    if name == "sequential":
        return SequentialSampler(labels)
    if name == "random":
        return RandomSampler(labels, generator=generator)
    if name in {"weighted", "class_balanced"}:
        configured_weights = config.get("class_weights")
        if configured_weights is not None:
            class_weights = torch.as_tensor(
                configured_weights,
                dtype=torch.double,
            )
            if labels_tensor.numel() and int(labels_tensor.max()) >= len(class_weights):
                raise ValueError("Sampler class_weights do not cover every label")
            sample_weights = class_weights[labels_tensor]
        else:
            counts = torch.bincount(labels_tensor)
            if torch.any(counts == 0):
                raise ValueError(
                    f"Cannot balance empty class; counts={counts.tolist()}"
                )
            sample_weights = (1.0 / counts.double())[labels_tensor]
        return WeightedRandomSampler(
            sample_weights,
            num_samples=int(config.get("num_samples", len(labels))),
            replacement=bool(config.get("replacement", True)),
            generator=generator,
        )
    if name == "hard_negative":
        weights = torch.ones(len(labels), dtype=torch.double)
        multiplier = float(config.get("multiplier", 2.0))
        for index in config.get("hard_negative_indices", []):
            weights[int(index)] = multiplier
        return WeightedRandomSampler(
            weights,
            num_samples=int(config.get("num_samples", len(labels))),
            replacement=True,
            generator=generator,
        )
    raise ValueError(f"Unknown sampler: {name}")
