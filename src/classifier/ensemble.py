from __future__ import annotations

from dataclasses import dataclass

import torch

from src.classifier.thresholding import metrics_at_threshold


@dataclass(frozen=True)
class EnsembleSearchResult:
    weight_first: float
    probabilities: torch.Tensor
    metrics: dict
    constraint_satisfied: bool


def blend_probabilities(first, second, weight_first):
    first = torch.as_tensor(first, dtype=torch.float32)
    second = torch.as_tensor(second, dtype=torch.float32)
    if first.shape != second.shape:
        raise ValueError("Ensemble probability tensors must have the same shape")
    weight_first = float(weight_first)
    if not 0.0 <= weight_first <= 1.0:
        raise ValueError("Ensemble weight must be in [0, 1]")
    return weight_first * first + (1.0 - weight_first) * second


def search_ensemble_weight(
    targets,
    first,
    second,
    *,
    weights=(0.2, 0.3, 0.4, 0.5),
    threshold=0.5,
    min_recall=0.99,
):
    candidates = []
    for weight in weights:
        probabilities = blend_probabilities(first, second, weight)
        metrics = metrics_at_threshold(targets, probabilities, threshold)
        candidates.append((float(weight), probabilities, metrics))
    eligible = [item for item in candidates if item[2]["recall"] >= min_recall]
    pool = eligible or candidates
    weight, probabilities, metrics = max(
        pool,
        key=lambda item: (
            item[2]["specificity"],
            item[2]["accuracy"],
            item[2]["f1"],
        ),
    )
    return EnsembleSearchResult(
        weight_first=weight,
        probabilities=probabilities,
        metrics=metrics,
        constraint_satisfied=metrics["recall"] >= min_recall,
    )


def select_recall_constrained_method(
    methods,
    *,
    min_recall=0.99,
    min_specificity=0.85,
    min_accuracy=0.92,
):
    if not methods:
        raise ValueError("No candidate methods supplied")
    fully_eligible = [
        item
        for item in methods
        if item["metrics"]["recall"] >= min_recall
        and item["metrics"]["specificity"] >= min_specificity
        and item["metrics"]["accuracy"] >= min_accuracy
    ]
    recall_eligible = [
        item for item in methods if item["metrics"]["recall"] >= min_recall
    ]
    pool = fully_eligible or recall_eligible or list(methods)
    return max(
        pool,
        key=lambda item: (
            item["metrics"]["specificity"],
            item["metrics"]["accuracy"],
            item["metrics"]["f1"],
            item["metrics"]["recall"],
        ),
    )
