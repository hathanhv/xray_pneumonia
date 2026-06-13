from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.classifier.evaluate import compute_metrics


@dataclass(frozen=True)
class ThresholdSearchResult:
    threshold: float
    metrics: dict[str, Any]
    constraint_satisfied: bool
    fallback_used: bool


def metrics_at_threshold(targets, probabilities, threshold):
    targets = torch.as_tensor(targets, dtype=torch.long)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    predictions = (probabilities >= float(threshold)).long()
    return compute_metrics(targets, predictions)


def _score(metrics, objective):
    objective = objective.lower()
    if objective == "min_fp":
        return (-metrics["fp"], metrics["specificity"], metrics["f1"])
    if objective in {"max_specificity", "max_normal_recall"}:
        return (metrics["specificity"], metrics["f1"], -metrics["fp"])
    if objective == "max_f1":
        return (metrics["f1"], metrics["specificity"], metrics["accuracy"])
    if objective == "max_accuracy":
        return (metrics["accuracy"], metrics["f1"], metrics["specificity"])
    raise ValueError(f"Unknown threshold objective: {objective}")


def tune_threshold(
    targets,
    probabilities,
    config: Mapping[str, Any] | None = None,
):
    config = dict(config or {})
    start = float(config.get("start", 0.10))
    stop = float(config.get("stop", 0.95))
    step = float(config.get("step", 0.01))
    min_recall = float(config.get("min_recall", 0.95))
    fallback_recall = config.get("fallback_recall")
    objective = str(config.get("objective", "min_fp"))
    if step <= 0 or stop < start:
        raise ValueError("Invalid threshold search range")

    count = int(round((stop - start) / step)) + 1
    thresholds = [start + index * step for index in range(count)]
    candidates = [
        (threshold, metrics_at_threshold(targets, probabilities, threshold))
        for threshold in thresholds
    ]

    eligible = [
        item for item in candidates if item[1]["recall"] >= min_recall
    ]
    fallback_used = False
    if not eligible and fallback_recall is not None:
        eligible = [
            item
            for item in candidates
            if item[1]["recall"] >= float(fallback_recall)
        ]
        fallback_used = bool(eligible)
    pool = eligible or candidates
    threshold, metrics = max(
        pool,
        key=lambda item: (_score(item[1], objective), item[0]),
    )
    return ThresholdSearchResult(
        threshold=float(threshold),
        metrics=metrics,
        constraint_satisfied=metrics["recall"] >= min_recall,
        fallback_used=fallback_used or not eligible,
    )
