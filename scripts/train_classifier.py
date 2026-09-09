from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.dataset import CLASS_TO_IDX, create_loaders_from_config
from src.classifier.evaluate import (
    evaluate_model,
    format_classification_report,
    format_confusion_matrix,
    save_evaluation_artifacts,
)
from src.classifier.losses import build_loss, compute_class_weights
from src.classifier.model import (
    build_mobilenet_v2_from_config,
    load_classifier_checkpoint,
)
from src.core.config import load_config
from src.core.experiment import create_experiment
from src.core.reproducibility import seed_everything
from src.training import (
    CheckpointManager,
    ClassificationTrainer,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-id")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def build_overrides(args):
    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.device is not None:
        overrides["device"] = args.device
    return overrides


def save_training_curves(history, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="train",
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] for row in history],
        label="validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    config = load_config(args.config, overrides=build_overrides(args))
    if args.print_config:
        print(config)
        return

    seed = int(config.get("seed", config["reproducibility"]["seed"]))
    seed_everything(seed)
    device = torch.device(
        config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    try:
        loaders, datasets = create_loaders_from_config(config)
    except ValueError as error:
        if "No classification images found" not in str(error):
            raise
        raise SystemExit(
            f"{error}\n\n"
            "Expected ImageFolder layout:\n"
            f"{config['dataset']['data_dir']}\\train\\NORMAL\n"
            f"{config['dataset']['data_dir']}\\train\\PNEUMONIA\n"
            f"{config['dataset']['data_dir']}\\test\\NORMAL\n"
            f"{config['dataset']['data_dir']}\\test\\PNEUMONIA"
        ) from error
    print(
        "Dataset sizes: "
        + ", ".join(f"{name}={len(dataset)}" for name, dataset in datasets.items())
    )
    print(f"Class mapping: {CLASS_TO_IDX}")
    print(f"Device: {device}")

    model_config = dict(config["model"])
    if args.resume or model_config.get("init_checkpoint_path"):
        model_config["pretrained"] = False
    model, model_metadata = build_mobilenet_v2_from_config(model_config)
    init_checkpoint = config["model"].get("init_checkpoint_path")
    if init_checkpoint and not args.resume:
        _, report = load_classifier_checkpoint(
            model,
            init_checkpoint,
            device="cpu",
            strict=True,
        )
        model_metadata["initialization_checkpoint"] = report.path

    loss_config = config["training"]["loss"]
    class_weights = None
    if loss_config["name"] in {"weighted_cross_entropy", "weighted_ce", "focal"}:
        class_weights = compute_class_weights(
            datasets["train"].targets,
            len(CLASS_TO_IDX),
        ).to(device)
    criterion = build_loss(loss_config, class_weights=class_weights).to(device)
    optimizer = build_optimizer(model, config["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, config["training"].get("scheduler"))
    early_stopping = build_early_stopping(config["training"]["early_stopping"])

    with create_experiment(config, run_id=args.run_id) as experiment:
        checkpoint_manager = CheckpointManager(
            experiment.best_checkpoint_path,
            experiment.last_checkpoint_path,
        )
        trainer = ClassificationTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            checkpoint_manager=checkpoint_manager,
            device=device,
            metric_logger=experiment.metric_logger,
            logger=experiment.logger,
            checkpoint_metadata={
                **model_metadata,
                "class_to_idx": CLASS_TO_IDX,
                "config_path": str(Path(args.config)),
            },
        )
        history = trainer.fit(
            loaders["train"],
            loaders["val"],
            epochs=config["training"]["epochs"],
            resume_from=args.resume,
        )
        if not experiment.best_checkpoint_path.exists():
            raise RuntimeError(
                "No best checkpoint satisfied the monitor constraints. "
                f"Last checkpoint: {experiment.last_checkpoint_path}"
            )
        checkpoint_manager.resume(
            experiment.best_checkpoint_path,
            model=model,
            device=device,
        )
        test_metrics, test_details = evaluate_model(
            model,
            loaders["test"],
            criterion,
            device,
            return_details=True,
        )
        artifacts = save_evaluation_artifacts(
            test_metrics,
            test_details,
            output_dir=experiment.run_dir / "evaluation",
        )
        curve_path = save_training_curves(
            history,
            experiment.figure_dir / "training_curves.png",
        )
        result = {
            "best_epoch": early_stopping.best_epoch,
            "best_score": early_stopping.best_score,
            "monitor": early_stopping.monitor,
            "test": test_metrics,
            "classification_report": test_details["classification_report"],
            "evaluation_artifacts": artifacts,
            "training_curves": str(curve_path),
        }
        experiment.save_metrics(result)
        print()
        print(format_confusion_matrix(test_metrics))
        print()
        print(format_classification_report(test_details["classification_report"]))
        print(f"\nRun directory: {experiment.run_dir}")
        print(f"Best checkpoint: {experiment.best_checkpoint_path}")
        print(f"Last checkpoint: {experiment.last_checkpoint_path}")


if __name__ == "__main__":
    main()
