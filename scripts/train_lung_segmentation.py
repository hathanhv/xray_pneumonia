from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.experiment import create_experiment
from src.core.reproducibility import seed_everything
from src.lung_segmentation.dataset import (
    build_segmentation_manifest,
    create_segmentation_loaders,
)
from src.lung_segmentation.evaluate import (
    evaluate_segmentation_model,
    save_segmentation_evaluation,
)
from src.lung_segmentation.export import export_monai_checkpoint
from src.lung_segmentation.losses import build_segmentation_loss
from src.lung_segmentation.model import build_unet_from_config
from src.lung_segmentation.trainer import SegmentationTrainer
from src.training import (
    CheckpointManager,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/seg_unet_resnet34.yaml",
    )
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Re-pair images and masks and replace the configured manifest.",
    )
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def _overrides(args):
    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.device:
        overrides["device"] = args.device
    return overrides


def _ensure_manifest(config, rebuild=False):
    dataset = config["dataset"]
    manifest_path = Path(dataset["manifest_path"])
    if manifest_path.exists() and not rebuild:
        return manifest_path
    build_segmentation_manifest(
        dataset["images_dir"],
        dataset["masks_dir"],
        manifest_path,
        mask_suffix=str(dataset.get("mask_suffix", "_mask")),
        val_fraction=float(dataset.get("val_fraction", 0.15)),
        test_fraction=float(dataset.get("test_fraction", 0.10)),
        seed=int(config.get("seed", config["reproducibility"]["seed"])),
    )
    return manifest_path


def main():
    args = parse_args()
    config = load_config(args.config, overrides=_overrides(args))
    seed = int(config.get("seed", config["reproducibility"]["seed"]))
    seed_everything(
        seed,
        deterministic=bool(config["reproducibility"].get("deterministic", True)),
        warn_only=bool(config["reproducibility"].get("warn_only", True)),
    )
    manifest_path = _ensure_manifest(config, rebuild=args.rebuild_manifest)
    loaders, datasets = create_segmentation_loaders(config)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Manifest: {manifest_path}")
    print(
        "Dataset sizes: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items())
    )
    print(f"Device: {device}")

    model, model_metadata = build_unet_from_config(config["model"])
    criterion = build_segmentation_loss(config["training"]["loss"])
    optimizer = build_optimizer(model, config["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, config["training"].get("scheduler"))
    early_stopping = build_early_stopping(config["training"]["early_stopping"])
    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))

    with create_experiment(config, run_id=args.run_id) as experiment:
        checkpoint_manager = CheckpointManager(
            experiment.best_checkpoint_path,
            experiment.last_checkpoint_path,
        )
        trainer = SegmentationTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            early_stopping=early_stopping,
            checkpoint_manager=checkpoint_manager,
            metric_logger=experiment.metric_logger,
            logger=experiment.logger,
            checkpoint_metadata={
                **model_metadata,
                "config_path": str(Path(args.config)),
                "manifest_path": str(manifest_path),
            },
            threshold=threshold,
        )
        history = trainer.fit(
            loaders["train"],
            loaders["val"],
            epochs=int(config["training"]["epochs"]),
            resume_from=args.resume,
        )
        if not experiment.best_checkpoint_path.exists():
            raise RuntimeError("Training did not produce best_model.pth.")
        checkpoint_manager.resume(
            experiment.best_checkpoint_path,
            model=model,
            device=device,
        )
        test_metrics, details = evaluate_segmentation_model(
            model,
            loaders["test"],
            criterion,
            device,
            threshold=threshold,
            return_details=True,
        )
        artifacts = save_segmentation_evaluation(
            test_metrics,
            details,
            experiment.run_dir / "evaluation",
        )
        exports = []
        if not args.no_export:
            for destination in config.get("export", {}).get("destinations", []):
                destination = Path(destination)
                if not destination.is_absolute():
                    destination = PROJECT_ROOT / destination
                path, backup = export_monai_checkpoint(
                    experiment.best_checkpoint_path,
                    destination,
                    backup_existing=bool(
                        config.get("export", {}).get("backup_existing", True)
                    ),
                )
                exports.append(
                    {
                        "path": str(path),
                        "backup": str(backup) if backup else None,
                    }
                )
        experiment.save_metrics(
            {
                "best_epoch": early_stopping.best_epoch,
                "best_score": early_stopping.best_score,
                "monitor": early_stopping.monitor,
                "test": test_metrics,
                "evaluation_artifacts": artifacts,
                "exports": exports,
                "epochs_completed": len(history),
            }
        )
        print(
            f"\nTest loss={test_metrics['loss']:.4f} | "
            f"Dice={test_metrics['dice']:.4f} | IoU={test_metrics['iou']:.4f}"
        )
        print(f"Run directory: {experiment.run_dir}")
        print(f"Best checkpoint: {experiment.best_checkpoint_path}")
        for item in exports:
            print(f"Exported: {item['path']}")
            if item["backup"]:
                print(f"Previous model backup: {item['backup']}")


if __name__ == "__main__":
    main()
