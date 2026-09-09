import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core.logging import CSVMetricLogger
from src.lung_segmentation.dataset import (
    LungSegmentationDataset,
    build_segmentation_manifest,
    load_segmentation_manifest,
)
from src.lung_segmentation.evaluate import binary_segmentation_scores
from src.lung_segmentation.export import export_monai_checkpoint
from src.lung_segmentation.losses import build_segmentation_loss
from src.lung_segmentation.model import load_checkpoint
from src.lung_segmentation.trainer import SegmentationTrainer
from src.training import CheckpointManager, EarlyStopping


def _write_pair(image_dir, mask_dir, index):
    image = np.zeros((12, 8), dtype=np.uint8)
    image[2:10, 2:6] = 100 + index
    mask = np.zeros_like(image)
    mask[3:9, 2:6] = 255
    cv2.imwrite(str(image_dir / f"sample_{index}.png"), image)
    cv2.imwrite(str(mask_dir / f"sample_{index}_mask.png"), mask)


def test_manifest_dataset_and_split_are_reproducible(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    for index in range(10):
        _write_pair(image_dir, mask_dir, index)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    build_segmentation_manifest(
        image_dir,
        mask_dir,
        first_path,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    build_segmentation_manifest(
        image_dir,
        mask_dir,
        second_path,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    with first_path.open(encoding="utf-8") as file:
        first_rows = list(csv.DictReader(file))
    with second_path.open(encoding="utf-8") as file:
        second_rows = list(csv.DictReader(file))
    assert [row["split"] for row in first_rows] == [
        row["split"] for row in second_rows
    ]
    assert {row["split"] for row in first_rows} == {"train", "val", "test"}

    dataset = LungSegmentationDataset(
        load_segmentation_manifest(first_path, split="train"),
        image_size=16,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["mask"].shape == (1, 16, 16)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert torch.equal(sample["image"][0], sample["image"][1])


def test_losses_and_metrics():
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    perfect_logits = torch.tensor([[[[20.0, -20.0], [20.0, -20.0]]]])
    wrong_logits = -perfect_logits
    loss = build_segmentation_loss(
        {"name": "dice_bce", "dice_weight": 0.5, "bce_weight": 0.5}
    )
    assert loss(perfect_logits, targets) < loss(wrong_logits, targets)
    dice, iou = binary_segmentation_scores(
        (torch.sigmoid(perfect_logits) >= 0.5).float(),
        targets,
    )
    assert dice.item() == 1.0
    assert iou.item() == 1.0


def test_exported_checkpoint_keeps_monai_legacy_schema(tmp_path):
    source = tmp_path / "best_model.pth"
    destination = tmp_path / "model" / "unet_lung_segmentation.pth"
    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(1)},
            "metadata": {
                "encoder": "resnet34",
                "img_size": 256,
                "best_val_loss": 0.25,
            },
        },
        source,
    )
    exported_path, backup = export_monai_checkpoint(source, destination)
    assert backup is None
    exported = load_checkpoint(exported_path)
    assert set(exported) == {
        "model_state_dict",
        "encoder",
        "img_size",
        "best_val_loss",
    }
    assert exported["encoder"] == "resnet34"
    assert exported["img_size"] == 256


def test_segmentation_trainer_saves_best_and_last_checkpoints(tmp_path):
    samples = [
        {
            "image": torch.zeros(3, 8, 8),
            "mask": torch.zeros(1, 8, 8),
            "sample_id": f"sample_{index}",
            "image_path": "image.png",
            "mask_path": "mask.png",
        }
        for index in range(4)
    ]
    loader = DataLoader(samples, batch_size=2)
    model = nn.Conv2d(3, 1, kernel_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = CheckpointManager(
        tmp_path / "best_model.pth",
        tmp_path / "last_model.pth",
    )
    trainer = SegmentationTrainer(
        model=model,
        criterion=build_segmentation_loss({"name": "dice"}),
        optimizer=optimizer,
        device="cpu",
        scheduler=None,
        early_stopping=EarlyStopping(monitor="val_loss", patience=2),
        checkpoint_manager=manager,
        metric_logger=CSVMetricLogger(tmp_path / "training_log.csv"),
        checkpoint_metadata={"encoder": "resnet34", "img_size": 256},
    )
    history = trainer.fit(loader, loader, epochs=1)
    assert len(history) == 1
    assert manager.best_path.exists()
    assert manager.last_path.exists()
    checkpoint = load_checkpoint(manager.best_path)
    assert checkpoint["encoder"] == "resnet34"
    assert checkpoint["img_size"] == 256
