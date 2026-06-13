from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def binary_segmentation_scores(predictions, targets, *, epsilon=1e-7):
    predictions = predictions.float().flatten(1)
    targets = targets.float().flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    prediction_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (
        prediction_sum + target_sum + epsilon
    )
    union = prediction_sum + target_sum - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return dice, iou


def evaluate_segmentation_model(
    model,
    dataloader,
    criterion,
    device,
    *,
    threshold=0.5,
    return_details=False,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    details = []
    dice_values = []
    iou_values = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            predictions = (torch.sigmoid(logits) >= threshold).float()
            dice, iou = binary_segmentation_scores(predictions, masks)
            batch_size = masks.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            dice_values.extend(dice.cpu().tolist())
            iou_values.extend(iou.cpu().tolist())
            for index, sample_id in enumerate(batch["sample_id"]):
                details.append(
                    {
                        "sample_id": sample_id,
                        "image_path": batch["image_path"][index],
                        "mask_path": batch["mask_path"][index],
                        "dice": float(dice[index].item()),
                        "iou": float(iou[index].item()),
                        "prediction": predictions[index, 0].cpu().numpy(),
                    }
                )
    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "iou": float(np.mean(iou_values)) if iou_values else 0.0,
        "samples": total_samples,
        "threshold": float(threshold),
    }
    return (metrics, details) if return_details else metrics


def save_segmentation_evaluation(metrics, details, output_dir):
    output_dir = Path(output_dir)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    csv_path = output_dir / "per_sample_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("sample_id", "image_path", "mask_path", "dice", "iou"),
        )
        writer.writeheader()
        for row in details:
            writer.writerow({key: row[key] for key in writer.fieldnames})
            cv2.imwrite(
                str(prediction_dir / f"{row['sample_id']}_mask.png"),
                (row["prediction"] * 255).astype(np.uint8),
            )
    return {
        "metrics": str(metrics_path),
        "per_sample_metrics": str(csv_path),
        "predictions": str(prediction_dir),
    }
