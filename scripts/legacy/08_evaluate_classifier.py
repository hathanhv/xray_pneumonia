from pathlib import Path
import argparse
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.classifier.dataset import create_imagefolder, get_eval_transforms
from src.classifier.evaluate import (
    evaluate_model,
    format_confusion_matrix,
    save_confusion_matrix_figure,
)
from src.classifier.model import build_mobilenet_v2, load_checkpoint_if_available


DATA_DIR = PROJECT_ROOT / "data" / "final" / "xray_2025_lung_crop_corrected"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "pneumonia_classifier"
    / "mobilenet_2025_lung_crop_corrected.pth"
)
CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT / "outputs" / "confusion_matrix" / "classifier_2025_eval_confusion_matrix.png"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--confusion-matrix-path", default=str(CONFUSION_MATRIX_PATH))
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = create_imagefolder(
        Path(args.data_dir) / args.split,
        transform=get_eval_transforms(args.img_size),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_mobilenet_v2(num_classes=2, pretrained=False)
    model = model.to(device)
    model = load_checkpoint_if_available(model, args.checkpoint, device)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, dataloader, criterion, device)

    print(f"Dataset: {Path(args.data_dir) / args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Class mapping: {dataset.class_to_idx}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(format_confusion_matrix(metrics))
    save_confusion_matrix_figure(metrics, args.confusion_matrix_path)
    print(f"Saved confusion matrix PNG: {args.confusion_matrix_path}")


if __name__ == "__main__":
    main()
