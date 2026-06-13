from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.dataset import (
    CLASS_TO_IDX,
    create_dataloaders,
    create_loaders_from_config,
)
from src.classifier.losses import build_loss, compute_class_weights
from src.classifier.evaluate import (
    evaluate_model,
    format_classification_report,
    format_confusion_matrix,
    save_evaluation_artifacts,
)
from src.classifier.model import (
    build_mobilenet_v2_from_config,
    load_classifier_checkpoint,
)
from src.classifier.train import train_classifier
from src.core.config import load_config
from src.core.experiment import create_experiment
from src.core.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/classification_2025_task10_base.yaml",
    )
    parser.add_argument(
        "--finetune-mode",
        choices=["auto", "full", "head", "last_blocks"],
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument(
        "--monitor",
        choices=["val_loss", "val_accuracy", "val_f1"],
    )
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr-patience", type=int)
    parser.add_argument("--lr-factor", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--unfreeze-blocks", type=int)
    parser.add_argument(
        "--checkpoint",
        help="Initial checkpoint. Defaults to the path in the YAML config.",
    )
    parser.add_argument(
        "--no-load-checkpoint",
        action="store_true",
        help="Start from ImageNet weights instead of the 2018 checkpoint.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download/use ImageNet weights when no checkpoint is loaded.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved YAML plus CLI configuration and exit.",
    )
    parser.add_argument(
        "--legacy-reproduction",
        action="store_true",
        help=(
            "Explicitly select the historical test-as-validation protocol. "
            "This is already the default for backward compatibility."
        ),
    )
    parser.add_argument(
        "--held-out-validation",
        action="store_true",
        help=(
            "Use a separate train/validation/test split. Select this for new "
            "scientific experiments without test leakage."
        ),
    )
    return parser.parse_args()


def build_cli_overrides(args):
    overrides = {}

    def set_value(section, key, value):
        if value is not None:
            overrides.setdefault(section, {})[key] = value

    set_value("model", "finetune_mode", args.finetune_mode)
    set_value("model", "unfreeze_blocks", args.unfreeze_blocks)
    set_value("training", "epochs", args.epochs)
    set_value("training", "monitor", args.monitor)
    set_value("training", "early_stopping_patience", args.patience)
    set_value("dataloader", "batch_size", args.batch_size)
    set_value("dataloader", "num_workers", args.num_workers)

    if args.lr is not None:
        overrides.setdefault("training", {}).setdefault("optimizer", {})[
            "learning_rate"
        ] = args.lr
    if args.weight_decay is not None:
        overrides.setdefault("training", {}).setdefault("optimizer", {})[
            "weight_decay"
        ] = args.weight_decay
    if args.lr_patience is not None:
        overrides.setdefault("training", {}).setdefault("scheduler", {})[
            "patience"
        ] = args.lr_patience
    if args.lr_factor is not None:
        overrides.setdefault("training", {}).setdefault("scheduler", {})[
            "factor"
        ] = args.lr_factor
    if args.label_smoothing is not None:
        overrides.setdefault("training", {}).setdefault("loss", {})[
            "label_smoothing"
        ] = args.label_smoothing
    if args.seed is not None:
        overrides["seed"] = args.seed
        overrides.setdefault("reproducibility", {})["seed"] = args.seed
        overrides.setdefault("dataset", {}).setdefault("split", {})[
            "seed"
        ] = args.seed
    if args.checkpoint is not None:
        overrides.setdefault("model", {})["init_checkpoint_path"] = args.checkpoint
    if args.no_load_checkpoint:
        overrides.setdefault("model", {})["init_checkpoint_path"] = None
    if args.no_pretrained:
        overrides.setdefault("model", {})["pretrained"] = False
    if args.legacy_reproduction and args.held_out_validation:
        raise ValueError(
            "--legacy-reproduction and --held-out-validation are mutually exclusive"
        )
    if args.held_out_validation:
        overrides.setdefault("evaluation", {})["protocol"] = "held_out_validation"
    elif args.legacy_reproduction:
        overrides.setdefault("evaluation", {})["protocol"] = (
            "legacy_test_as_validation"
        )
    return overrides


def print_run_configuration(config, device):
    protocol = config.get("evaluation", {}).get(
        "protocol",
        "held_out_validation",
    )
    print("=" * 72)
    print("CLASSIFICATION 2025 TRAINING CONFIGURATION")
    print(f"Device: {device}")
    data_source = config["dataset"].get(
        "manifest_path",
        config["dataset"].get("data_dir", "not configured"),
    )
    print(f"Data: {data_source}")
    print(f"Seed: {config['seed']}")
    print(f"Fine-tune mode: {config['model']['finetune_mode']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['dataloader']['batch_size']}")
    print(f"Learning rate: {config['training']['optimizer']['learning_rate']}")
    print(f"Monitor: {config['training']['monitor']}")
    print(f"Evaluation protocol: {protocol}")
    print(
        "Label smoothing: "
        f"{config['training']['loss'].get('label_smoothing', 0.0)}"
    )
    print("=" * 72)


def create_training_data(config):
    protocol = config.get("evaluation", {}).get(
        "protocol",
        "held_out_validation",
    )
    if protocol != "legacy_test_as_validation":
        loaders, datasets = create_loaders_from_config(config)
        return loaders, datasets, False

    print()
    print("WARNING: LEGACY REPRODUCTION MODE")
    print(
        "The 57-image test split is used for validation, early stopping, "
        "checkpoint selection, and final reporting."
    )
    print("Use this mode only to reproduce the historical checkpoint.")
    print()
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        img_size=config["preprocessing"].get("size", 224),
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        seed=config["seed"],
    )
    loaders = {
        "train": train_loader,
        "val": test_loader,
        "test": test_loader,
    }
    datasets = {
        "train": train_dataset,
        "val": test_dataset,
        "test": test_dataset,
    }
    print(f"Legacy train images: {len(train_dataset)}")
    print(f"Legacy test/validation images: {len(test_dataset)}")
    return loaders, datasets, True


def main():
    args = parse_args()
    config = load_config(args.config, overrides=build_cli_overrides(args))
    seed = int(config.get("seed", config["reproducibility"]["seed"]))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_run_configuration(config, device)
    if args.print_config:
        return

    loaders, datasets, legacy_reproduction = create_training_data(config)
    init_checkpoint_path = config["model"].get("init_checkpoint_path")
    model_config = dict(config["model"])
    if init_checkpoint_path:
        model_config["pretrained"] = False
    model, model_metadata = build_mobilenet_v2_from_config(model_config)
    if init_checkpoint_path:
        _, load_report = load_classifier_checkpoint(
            model,
            init_checkpoint_path,
            device="cpu",
            strict=True,
        )
        model_metadata["initialization_checkpoint"] = load_report.path
        model_metadata["initialization_epoch"] = load_report.epoch
        print(f"Initial checkpoint: {load_report.path}")
        print(
            "Checkpoint load: "
            f"missing={len(load_report.missing_keys)}, "
            f"unexpected={len(load_report.unexpected_keys)}"
        )
    else:
        print("Initial checkpoint: ImageNet weights")
    model.to(device)

    class_weights = compute_class_weights(
        datasets["train"].targets,
        len(CLASS_TO_IDX),
    ).to(device)
    criterion = build_loss(
        config["training"]["loss"],
        class_weights=class_weights,
    )
    optimizer_config = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )
    scheduler_config = config["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if config["training"]["monitor"] == "val_loss" else "max",
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
    )

    with create_experiment(config) as experiment:
        best_metrics, history = train_classifier(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"] if legacy_reproduction else None,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=config["training"]["epochs"],
            checkpoint_path=experiment.best_checkpoint_path,
            class_to_idx=CLASS_TO_IDX,
            scheduler=scheduler,
            monitor=config["training"]["monitor"],
            early_stopping_patience=config["training"]["early_stopping_patience"],
            checkpoint_metadata=model_metadata,
        )
        for row in history:
            experiment.metric_logger.log(row)
        if best_metrics is None:
            raise RuntimeError("Training completed without producing a best checkpoint")
        best_checkpoint, _load_report = load_classifier_checkpoint(
            model,
            experiment.best_checkpoint_path,
            device=device,
            strict=True,
        )
        test_metrics, test_details = evaluate_model(
            model=model,
            dataloader=loaders["test"],
            criterion=criterion,
            device=device,
            return_details=True,
        )
        best_metrics["test"] = test_metrics
        report = test_details["classification_report"]
        artifact_paths = save_evaluation_artifacts(
            test_metrics,
            test_details,
            output_dir=experiment.run_dir / "evaluation",
        )
        best_metrics["classification_report"] = report
        best_metrics["evaluation_artifacts"] = artifact_paths
        best_metrics["evaluated_checkpoint_epoch"] = best_checkpoint.get("epoch")
        experiment.save_metrics(best_metrics or {})

        print("\n" + "=" * 72)
        print(
            "EVALUATION PROTOCOL: "
            + (
                "LEGACY TEST-AS-VALIDATION (TEST LEAKAGE)"
                if legacy_reproduction
                else "HELD-OUT VALIDATION"
            )
        )
        print(f"BEST VALIDATION EPOCH: {best_metrics['best_epoch']}")
        print(format_confusion_matrix(test_metrics))
        print()
        print(format_classification_report(report))
        print("=" * 72)
        print(f"Run directory: {experiment.run_dir}")
        print(f"Best checkpoint: {experiment.best_checkpoint_path}")
        print(f"Confusion matrix PNG: {artifact_paths['confusion_matrix_png']}")
        print(
            "Classification report: "
            f"{artifact_paths['classification_report_text']}"
        )


if __name__ == "__main__":
    main()
