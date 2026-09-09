from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "legacy_baselines"
MANIFEST_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "manifests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_manifest(name: str) -> list[dict[str, str]]:
    path = MANIFEST_ROOT / name
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=True, default=to_json_value)


def to_json_value(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def capture_environment() -> None:
    packages = {}
    for module_name in (
        "torch",
        "torchvision",
        "cv2",
        "monai",
        "monailabel",
        "segmentation_models_pytorch",
        "SimpleITK",
    ):
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "installed")
        except ImportError:
            packages[module_name] = None

    torch_info = {}
    try:
        import torch

        torch_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass

    write_json(
        OUTPUT_ROOT / "environment.json",
        {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": packages,
            "torch": torch_info,
        },
    )


def capture_checkpoints() -> None:
    import torch

    checkpoint_paths = [
        PROJECT_ROOT / "checkpoints" / "lung_segmentation" / "unet_lung_segmentation.pth",
        PROJECT_ROOT
        / "monai_apps"
        / "lung_monai_app"
        / "model"
        / "unet_lung_segmentation.pth",
        PROJECT_ROOT
        / "checkpoints"
        / "pneumonia_classifier"
        / "mobilenet_2025_lung_crop_corrected.pth",
        PROJECT_ROOT
        / "checkpoints"
        / "pneumonia_classifier"
        / "pneumonia_classifier.pth",
    ]

    metadata = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        summary = {
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "keys": list(checkpoint) if isinstance(checkpoint, dict) else [],
        }
        if isinstance(checkpoint, dict):
            for key in (
                "model_name",
                "encoder",
                "img_size",
                "image_size",
                "best_val_loss",
                "epoch",
                "metrics",
                "best_method",
                "test_metrics",
                "temperature",
                "class_names",
                "class_to_idx",
            ):
                if key in checkpoint:
                    summary[key] = checkpoint[key]
        metadata.append(summary)

    write_json(OUTPUT_ROOT / "checkpoints.json", metadata)


def build_classifier() -> object:
    import torch.nn as nn
    from torchvision import models

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def capture_classification_2025() -> None:
    import torch
    from PIL import Image
    from torchvision import transforms

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "pneumonia_classifier"
        / "mobilenet_2025_lung_crop_corrected.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_classifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    rows = []
    with torch.no_grad():
        for item in read_manifest("classification_2025.csv"):
            relative_path = Path(item["image_path"])
            image = Image.open(PROJECT_ROOT / relative_path).convert("RGB")
            logits = model(transform(image).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0]
            prediction = int(torch.argmax(probabilities).item())
            rows.append(
                {
                    "image_path": relative_path.as_posix(),
                    "label": int(item["label"]),
                    "prediction": prediction,
                    "p_normal": float(probabilities[0].item()),
                    "p_pneumonia": float(probabilities[1].item()),
                }
            )

    output_dir = OUTPUT_ROOT / "classification_2025"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tp = sum(row["label"] == 1 and row["prediction"] == 1 for row in rows)
    tn = sum(row["label"] == 0 and row["prediction"] == 0 for row in rows)
    fp = sum(row["label"] == 0 and row["prediction"] == 1 for row in rows)
    fn = sum(row["label"] == 1 and row["prediction"] == 0 for row in rows)
    total = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    write_json(
        output_dir / "metrics.json",
        {
            "sample_count": total,
            "accuracy": (tp + tn) / total if total else 0.0,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    )


def capture_segmentation_artifacts() -> None:
    output_dir = OUTPUT_ROOT / "segmentation"
    mask_dir = output_dir / "masks"
    crop_dir = output_dir / "crops"
    mask_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in read_manifest("segmentation.csv"):
        mask_source = PROJECT_ROOT / item["expected_mask_path"]
        crop_source = PROJECT_ROOT / item["expected_crop_path"]
        mask_target = mask_dir / mask_source.name
        crop_target = crop_dir / crop_source.name
        shutil.copy2(mask_source, mask_target)
        shutil.copy2(crop_source, crop_target)
        rows.append(
            {
                **item,
                "mask_sha256": sha256(mask_source),
                "crop_sha256": sha256(crop_source),
            }
        )

    with (output_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_monai_artifacts() -> None:
    output_dir = OUTPUT_ROOT / "monai"
    predicted_dir = output_dir / "predicted_masks"
    corrected_dir = output_dir / "corrected_masks"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in read_manifest("monai.csv"):
        predicted_source = PROJECT_ROOT / item["legacy_predicted_mask_path"]
        corrected_source = PROJECT_ROOT / item["corrected_mask_path"]
        shutil.copy2(predicted_source, predicted_dir / predicted_source.name)
        shutil.copy2(corrected_source, corrected_dir / corrected_source.name)
        rows.append(
            {
                **item,
                "predicted_mask_sha256": sha256(predicted_source),
                "corrected_mask_sha256": sha256(corrected_source),
            }
        )

    with (output_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_existing_reports() -> None:
    report_dir = OUTPUT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        PROJECT_ROOT / "data" / "lung_seg_outputs" / "2025_all" / "qc_report_2025.csv",
        PROJECT_ROOT
        / "data"
        / "lung_seg_outputs"
        / "2025_all"
        / "final_masks_merge_report.csv",
        PROJECT_ROOT
        / "data"
        / "final"
        / "xray_2025_lung_crop_corrected"
        / "final_dataset_report.csv",
    ]
    for source in sources:
        shutil.copy2(source, report_dir / source.name)


def main() -> None:
    capture_environment()
    capture_checkpoints()
    capture_classification_2025()
    capture_segmentation_artifacts()
    capture_monai_artifacts()
    capture_existing_reports()
    print(f"Legacy baseline captured at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
