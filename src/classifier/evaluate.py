import csv
import json
from pathlib import Path

import torch

from src.classifier.dataset import unpack_batch


def confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred, loss=None):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return metrics


def build_classification_report(metrics):
    normal_precision = (
        metrics["tn"] / (metrics["tn"] + metrics["fn"])
        if metrics["tn"] + metrics["fn"]
        else 0.0
    )
    normal_recall = metrics["specificity"]
    normal_f1 = (
        2 * normal_precision * normal_recall / (normal_precision + normal_recall)
        if normal_precision + normal_recall
        else 0.0
    )
    normal_support = metrics["tn"] + metrics["fp"]
    pneumonia_support = metrics["tp"] + metrics["fn"]
    total = normal_support + pneumonia_support
    macro_precision = (normal_precision + metrics["precision"]) / 2
    macro_recall = (normal_recall + metrics["recall"]) / 2
    macro_f1 = (normal_f1 + metrics["f1"]) / 2
    weighted_precision = (
        normal_precision * normal_support
        + metrics["precision"] * pneumonia_support
    ) / total if total else 0.0
    weighted_recall = (
        normal_recall * normal_support + metrics["recall"] * pneumonia_support
    ) / total if total else 0.0
    weighted_f1 = (
        normal_f1 * normal_support + metrics["f1"] * pneumonia_support
    ) / total if total else 0.0
    return {
        "NORMAL": {
            "precision": normal_precision,
            "recall": normal_recall,
            "f1-score": normal_f1,
            "support": normal_support,
        },
        "PNEUMONIA": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1-score": metrics["f1"],
            "support": pneumonia_support,
        },
        "accuracy": metrics["accuracy"],
        "macro avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": total,
        },
        "weighted avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1,
            "support": total,
        },
    }


def format_classification_report(report):
    lines = [
        "Classification report:",
        "",
        f"{'class':>14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>9}",
        "",
    ]
    for name in ("NORMAL", "PNEUMONIA"):
        row = report[name]
        lines.append(
            f"{name:>14} {row['precision']:10.4f} {row['recall']:10.4f} "
            f"{row['f1-score']:10.4f} {row['support']:9d}"
        )
    total = report["macro avg"]["support"]
    lines.extend(
        [
            "",
            f"{'accuracy':>14} {'':>10} {'':>10} "
            f"{report['accuracy']:10.4f} {total:9d}",
        ]
    )
    for name in ("macro avg", "weighted avg"):
        row = report[name]
        lines.append(
            f"{name:>14} {row['precision']:10.4f} {row['recall']:10.4f} "
            f"{row['f1-score']:10.4f} {row['support']:9d}"
        )
    return "\n".join(lines)


def format_confusion_matrix(metrics):
    return (
        "Confusion matrix:\n"
        "Rows = true label, Columns = predicted label\n"
        "\n"
        "                 pred_NORMAL  pred_PNEUMONIA\n"
        f"true_NORMAL      {metrics['tn']:11d}  {metrics['fp']:14d}\n"
        f"true_PNEUMONIA   {metrics['fn']:11d}  {metrics['tp']:14d}"
    )


def save_confusion_matrix_figure(metrics, output_path):
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
    ax.set_yticklabels(["NORMAL", "PNEUMONIA"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_model(model, dataloader, criterion, device, return_details=False):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true_all = []
    y_pred_all = []
    probabilities_all = []
    metadata_all = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels, metadata = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            y_true_all.append(labels.detach().cpu())
            y_pred_all.append(preds.detach().cpu())
            if return_details:
                probabilities_all.append(probabilities.detach().cpu())
                metadata_all.extend(
                    metadata or [{} for _index in range(labels.size(0))]
                )

    y_true = torch.cat(y_true_all) if y_true_all else torch.empty(0, dtype=torch.long)
    y_pred = torch.cat(y_pred_all) if y_pred_all else torch.empty(0, dtype=torch.long)
    avg_loss = total_loss / total_samples if total_samples else 0.0
    metrics = compute_metrics(y_true, y_pred, loss=avg_loss)
    if not return_details:
        return metrics
    probabilities = (
        torch.cat(probabilities_all)
        if probabilities_all
        else torch.empty((0, 2), dtype=torch.float32)
    )
    predictions = []
    for index in range(len(y_true)):
        metadata = metadata_all[index] if index < len(metadata_all) else {}
        predictions.append(
            {
                "sample_id": metadata.get("sample_id", str(index)),
                "image_path": metadata.get("image_path", ""),
                "target": int(y_true[index].item()),
                "prediction": int(y_pred[index].item()),
                "p_normal": float(probabilities[index, 0].item()),
                "p_pneumonia": float(probabilities[index, 1].item()),
            }
        )
    return metrics, {
        "classification_report": build_classification_report(metrics),
        "predictions": predictions,
    }


def save_evaluation_artifacts(
    metrics,
    details,
    *,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = details["classification_report"]
    report_text = format_classification_report(report)

    report_json_path = output_dir / "classification_report.json"
    report_text_path = output_dir / "classification_report.txt"
    predictions_path = output_dir / "test_predictions.csv"
    confusion_matrix_path = output_dir / "confusion_matrix.png"

    report_json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    report_text_path.write_text(report_text + "\n", encoding="utf-8")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_id",
            "image_path",
            "target",
            "prediction",
            "p_normal",
            "p_pneumonia",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details["predictions"])
    save_confusion_matrix_figure(metrics, confusion_matrix_path)
    return {
        "classification_report_json": str(report_json_path),
        "classification_report_text": str(report_text_path),
        "test_predictions_csv": str(predictions_path),
        "confusion_matrix_png": str(confusion_matrix_path),
    }
