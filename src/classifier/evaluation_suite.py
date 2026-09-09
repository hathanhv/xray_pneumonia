from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.classifier.calibration import TemperatureScaler
from src.classifier.evaluate import (
    build_classification_report,
    compute_metrics,
    format_classification_report,
    save_confusion_matrix_figure,
)


def probability_metrics(targets, probabilities, *, bins=10):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predictions = (probabilities >= 0.5).astype(np.int64)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = (predictions == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    reliability = []
    ece = 0.0
    for index in range(len(edges) - 1):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == len(edges) - 2 else confidence < upper
        )
        count = int(selected.sum())
        accuracy = float(correctness[selected].mean()) if count else 0.0
        mean_confidence = float(confidence[selected].mean()) if count else 0.0
        if count:
            ece += count / len(targets) * abs(accuracy - mean_confidence)
        reliability.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    brier = float(np.mean((probabilities - targets) ** 2))
    nll = float(
        -np.mean(
            targets * np.log(probabilities)
            + (1 - targets) * np.log(1 - probabilities)
        )
    )
    return {
        "ece": float(ece),
        "brier_score": brier,
        "negative_log_likelihood": nll,
        "reliability": reliability,
    }


def curve_metrics(targets, probabilities):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        targets, probabilities
    )
    precision, recall, pr_thresholds = precision_recall_curve(
        targets, probabilities
    )
    return {
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "roc": {
            "false_positive_rate": false_positive_rate.tolist(),
            "true_positive_rate": true_positive_rate.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def fit_temperature(val_logits, val_targets):
    scaler = TemperatureScaler(initial_temperature=1.0)
    temperature = scaler.fit(
        torch.as_tensor(val_logits, dtype=torch.float32),
        torch.as_tensor(val_targets, dtype=torch.long),
        max_iter=100,
    )
    return scaler, temperature


def save_curve_figure(curves, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        curves["roc"]["false_positive_rate"],
        curves["roc"]["true_positive_rate"],
        label=f"ROC-AUC={curves['roc_auc']:.3f}",
    )
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        curves["pr"]["recall"],
        curves["pr"]["precision"],
        label=f"PR-AUC={curves['pr_auc']:.3f}",
    )
    axes[1].set(xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_reliability_figure(before, after, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for name, report in (("Before", before), ("After", after)):
        rows = [row for row in report["reliability"] if row["count"]]
        ax.plot(
            [row["confidence"] for row in rows],
            [row["accuracy"] for row in rows],
            marker="o",
            label=f"{name} (ECE={report['ece']:.3f})",
        )
    ax.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def export_errors(rows, output_dir):
    output_dir = Path(output_dir)
    fp_dir = output_dir / "false_positives"
    fn_dir = output_dir / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for row in rows:
        target = int(row["target"])
        prediction = int(row["prediction"])
        if target == prediction:
            continue
        error_type = "FP" if target == 0 else "FN"
        destination_dir = fp_dir if error_type == "FP" else fn_dir
        source = Path(row["image_path"])
        destination = destination_dir / source.name
        if source.exists():
            shutil.copy2(source, destination)
        exported.append(
            {
                **row,
                "error_type": error_type,
                "exported_path": str(destination.resolve()) if destination.exists() else "",
            }
        )
    report_path = output_dir / "errors.csv"
    fieldnames = list(exported[0]) if exported else [
        "sample_id",
        "image_path",
        "target",
        "prediction",
        "p_normal",
        "p_pneumonia",
        "error_type",
        "exported_path",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exported)
    return {
        "false_positives": sum(row["error_type"] == "FP" for row in exported),
        "false_negatives": sum(row["error_type"] == "FN" for row in exported),
        "report_path": str(report_path),
    }


def evaluate_logits(
    *,
    strategy,
    test_logits,
    test_targets,
    test_metadata,
    val_logits,
    val_targets,
    output_dir,
    run_type="full",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_logits = torch.as_tensor(test_logits, dtype=torch.float32)
    test_targets = torch.as_tensor(test_targets, dtype=torch.long)
    val_logits = torch.as_tensor(val_logits, dtype=torch.float32)
    val_targets = torch.as_tensor(val_targets, dtype=torch.long)
    probabilities = torch.softmax(test_logits, dim=1)[:, 1].numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = compute_metrics(test_targets, torch.as_tensor(predictions))
    curves = curve_metrics(test_targets.numpy(), probabilities)
    calibration_before = probability_metrics(test_targets.numpy(), probabilities)

    scaler, temperature = fit_temperature(val_logits, val_targets)
    with torch.no_grad():
        calibrated_probabilities = torch.softmax(scaler(test_logits), dim=1)[:, 1].numpy()
    calibration_after = probability_metrics(
        test_targets.numpy(), calibrated_probabilities
    )
    rows = []
    for index, target in enumerate(test_targets.tolist()):
        metadata = test_metadata[index] if index < len(test_metadata) else {}
        rows.append(
            {
                "sample_id": metadata.get("sample_id", str(index)),
                "image_path": metadata.get("image_path", ""),
                "target": target,
                "prediction": int(predictions[index]),
                "p_normal": float(1.0 - probabilities[index]),
                "p_pneumonia": float(probabilities[index]),
                "p_pneumonia_calibrated": float(calibrated_probabilities[index]),
            }
        )
    report = build_classification_report(metrics)
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(
        format_classification_report(report) + "\n", encoding="utf-8"
    )
    save_confusion_matrix_figure(metrics, output_dir / "confusion_matrix.png")
    save_curve_figure(curves, output_dir / "roc_pr_curves.png")
    save_reliability_figure(
        calibration_before,
        calibration_after,
        output_dir / "reliability_diagram.png",
    )
    errors = export_errors(rows, output_dir / "error_analysis")
    result = {
        "strategy": strategy,
        "run_type": run_type,
        **metrics,
        "roc_auc": curves["roc_auc"],
        "pr_auc": curves["pr_auc"],
        "temperature": temperature,
        "ece_before": calibration_before["ece"],
        "ece_after": calibration_after["ece"],
        "brier_before": calibration_before["brier_score"],
        "brier_after": calibration_after["brier_score"],
        "nll_before": calibration_before["negative_log_likelihood"],
        "nll_after": calibration_after["negative_log_likelihood"],
        "error_export": errors,
    }
    (output_dir / "calibration_report.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "before": calibration_before,
                "after": calibration_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "curves.json").write_text(
        json.dumps(curves, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
