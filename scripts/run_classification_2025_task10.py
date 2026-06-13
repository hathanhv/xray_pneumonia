from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.dataset import CLASS_TO_IDX, create_loaders_from_config
from src.classifier.evaluate import evaluate_model, save_evaluation_artifacts
from src.classifier.losses import build_loss
from src.classifier.model import (
    build_mobilenet_v2_from_config,
    load_classifier_checkpoint,
)
from src.core.config import load_config


CONFIG_DIR = PROJECT_ROOT / "configs/experiments"
OUTPUT_DIR = PROJECT_ROOT / "outputs/task10_classification_2025"
PREPROCESSING_CONFIGS = {
    "raw_baseline": CONFIG_DIR / "classification_2025_task10_raw.yaml",
    "histogram_matching": CONFIG_DIR / "classification_2025_task10_histogram.yaml",
    "lung_roi": CONFIG_DIR / "classification_2025_task10_roi.yaml",
    "refined_roi": CONFIG_DIR / "classification_2025_task10_refined_roi.yaml",
}
TRAINING_CONFIGS = {
    "head_warmup": CONFIG_DIR / "classification_2025_task10_head_warmup.yaml",
    "full_finetune": CONFIG_DIR / "classification_2025_task10_full_finetune.yaml",
    "few_shot": CONFIG_DIR / "classification_2025_task10_few_shot.yaml",
}


def evaluate_strategy(name, config_path, device):
    config = load_config(config_path)
    loaders, _datasets = create_loaders_from_config(config)
    model_config = dict(config["model"])
    model_config["pretrained"] = False
    model, _metadata = build_mobilenet_v2_from_config(model_config)
    load_classifier_checkpoint(
        model,
        config["model"]["init_checkpoint_path"],
        device="cpu",
        strict=True,
    )
    model.to(device)
    criterion = build_loss("cross_entropy").to(device)
    metrics, details = evaluate_model(
        model,
        loaders["test"],
        criterion,
        device,
        return_details=True,
    )
    strategy_dir = OUTPUT_DIR / name
    artifacts = save_evaluation_artifacts(
        metrics,
        details,
        output_dir=strategy_dir,
    )
    result = {
        "strategy": name,
        "kind": "preprocessing",
        "config": str(config_path),
        **metrics,
        "artifacts": artifacts,
    }
    (strategy_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def latest_run_checkpoint(experiment_name):
    root = PROJECT_ROOT / "outputs/experiments" / experiment_name
    runs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    if not runs:
        raise FileNotFoundError(f"No completed run found for {experiment_name}")
    checkpoint = runs[0] / "checkpoints/best_model.pth"
    metrics = runs[0] / "metrics.json"
    if not checkpoint.exists() or not metrics.exists():
        raise FileNotFoundError(f"Incomplete run: {runs[0]}")
    return runs[0], checkpoint, json.loads(metrics.read_text(encoding="utf-8"))


def run_training(config_path, *, checkpoint=None, epochs=None):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/07_train_classifier_2025.py"),
        "--config",
        str(config_path),
        "--held-out-validation",
    ]
    if checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])
    if epochs:
        command.extend(["--epochs", str(epochs)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    config = load_config(config_path)
    return latest_run_checkpoint(config["experiment"]["name"])


def flatten_training_result(strategy, run_dir, metrics):
    test = metrics.get("test", {})
    config_used = yaml.safe_load(
        (Path(run_dir) / "config_used.yaml").read_text(encoding="utf-8")
    )
    epochs = int(config_used["training"]["epochs"])
    return {
        "strategy": strategy,
        "kind": "adaptation",
        "run_type": "smoke" if epochs == 1 else "full",
        "epochs": epochs,
        "run_dir": str(run_dir),
        "loss": test.get("loss"),
        "accuracy": test.get("accuracy"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "specificity": test.get("specificity"),
        "f1": test.get("f1"),
        "tp": test.get("tp"),
        "tn": test.get("tn"),
        "fp": test.get("fp"),
        "fn": test.get("fn"),
    }


def collect_existing_results():
    results = []
    for name in PREPROCESSING_CONFIGS:
        path = OUTPUT_DIR / name / "metrics.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    for name, config_path in TRAINING_CONFIGS.items():
        config = load_config(config_path)
        try:
            run_dir, _checkpoint, metrics = latest_run_checkpoint(
                config["experiment"]["name"]
            )
        except FileNotFoundError:
            continue
        results.append(flatten_training_result(name, run_dir, metrics))
    return results


def write_comparison(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "strategy",
        "kind",
        "run_type",
        "epochs",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "run_dir",
        "config",
    ]
    with (OUTPUT_DIR / "domain_shift_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    official_results = [
        row for row in results if row.get("run_type", "full") != "smoke"
    ]
    ranked = sorted(
        official_results,
        key=lambda row: (
            row.get("f1") if row.get("f1") is not None else -1,
            row.get("accuracy") if row.get("accuracy") is not None else -1,
        ),
        reverse=True,
    )
    smoke_results = [
        row["strategy"] for row in results if row.get("run_type") == "smoke"
    ]
    summary = {
        "results": results,
        "ranking": [row["strategy"] for row in ranked],
        "smoke_only": smoke_results,
    }
    (OUTPUT_DIR / "domain_shift_comparison.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    save_comparison_figure(results)
    return summary


def save_comparison_figure(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [
        row["strategy"] + ("*" if row.get("run_type") == "smoke" else "")
        for row in results
    ]
    metrics = ["accuracy", "f1", "recall", "specificity"]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    for index, metric in enumerate(metrics):
        values = [float(row.get(metric) or 0.0) for row in results]
        ax.bar(x + (index - 1.5) * width, values, width, label=metric)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Task 10: Classification 2025 Domain-Shift Strategies")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_shift_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["prepare", "evaluate", "train", "compare", "all"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one epoch for each adaptation stage.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.command in {"prepare", "all"}:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/prepare_classification_2025_task10.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )

    if args.command in {"evaluate", "all"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for name, config_path in PREPROCESSING_CONFIGS.items():
            result = evaluate_strategy(name, config_path, device)
            print(name, result["accuracy"], result["f1"])

    if args.command in {"train", "all"}:
        epochs = 1 if args.smoke else None
        head_run, head_checkpoint, head_metrics = run_training(
            TRAINING_CONFIGS["head_warmup"],
            epochs=epochs,
        )
        print(json.dumps(flatten_training_result("head_warmup", head_run, head_metrics), indent=2))
        full_run, _full_checkpoint, full_metrics = run_training(
            TRAINING_CONFIGS["full_finetune"],
            checkpoint=head_checkpoint,
            epochs=epochs,
        )
        print(json.dumps(flatten_training_result("full_finetune", full_run, full_metrics), indent=2))
        few_run, _few_checkpoint, few_metrics = run_training(
            TRAINING_CONFIGS["few_shot"],
            epochs=epochs,
        )
        print(json.dumps(flatten_training_result("few_shot", few_run, few_metrics), indent=2))

    if args.command in {"compare", "all"}:
        summary = write_comparison(collect_existing_results())
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
