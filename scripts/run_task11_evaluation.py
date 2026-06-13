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

from src.classifier.dataset import create_loaders_from_config
from src.classifier.evaluation_suite import evaluate_logits
from src.classifier.model import (
    build_mobilenet_v2_from_config,
    load_classifier_checkpoint,
)
from src.classifier.prediction import collect_logits
from src.core.config import load_config


TASK10_DIR = PROJECT_ROOT / "outputs/task10_classification_2025"
OUTPUT_DIR = PROJECT_ROOT / "outputs/task11_evaluation"
PREPROCESSING = {
    "raw_baseline": "configs/experiments/classification_2025_task10_raw.yaml",
    "histogram_matching": "configs/experiments/classification_2025_task10_histogram.yaml",
    "lung_roi": "configs/experiments/classification_2025_task10_roi.yaml",
    "refined_roi": "configs/experiments/classification_2025_task10_refined_roi.yaml",
}
ADAPTATION = {
    "head_warmup": "classification_2025_task10_head_warmup",
    "full_finetune": "classification_2025_task10_full_finetune",
    "few_shot": "classification_2025_task10_few_shot",
}


def latest_run(experiment_name):
    root = PROJECT_ROOT / "outputs/experiments" / experiment_name
    runs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    if not runs:
        raise FileNotFoundError(f"No run found for {experiment_name}")
    return runs[0]


def strategy_source(name):
    if name in PREPROCESSING:
        config_path = PROJECT_ROOT / PREPROCESSING[name]
        config = load_config(config_path)
        checkpoint = Path(config["model"]["init_checkpoint_path"])
        return config, checkpoint, "full"
    run_dir = latest_run(ADAPTATION[name])
    config_path = run_dir / "config_used.yaml"
    config = load_config(config_path)
    checkpoint = run_dir / "checkpoints/best_model.pth"
    run_type = "smoke" if int(config["training"]["epochs"]) == 1 else "full"
    return config, checkpoint, run_type


def evaluate_strategy(name, device):
    config, checkpoint, run_type = strategy_source(name)
    loaders, _datasets = create_loaders_from_config(config)
    model_config = dict(config["model"])
    model_config["pretrained"] = False
    model, _metadata = build_mobilenet_v2_from_config(model_config)
    load_classifier_checkpoint(model, checkpoint, device="cpu", strict=True)
    model.to(device)
    val_logits, val_targets, _val_metadata = collect_logits(
        model, loaders["val"], device
    )
    test_logits, test_targets, test_metadata = collect_logits(
        model, loaders["test"], device
    )
    return evaluate_logits(
        strategy=name,
        test_logits=test_logits,
        test_targets=test_targets,
        test_metadata=test_metadata,
        val_logits=val_logits,
        val_targets=val_targets,
        output_dir=OUTPUT_DIR / name,
        run_type=run_type,
    )


def collect_results():
    results = []
    for path in sorted(OUTPUT_DIR.glob("*/metrics.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def write_summary(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "strategy",
        "run_type",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "ece_before",
        "ece_after",
        "brier_before",
        "brier_after",
        "temperature",
        "fp",
        "fn",
    ]
    with (OUTPUT_DIR / "experiment_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    official = [row for row in results if row.get("run_type") != "smoke"]
    ranked = sorted(
        official,
        key=lambda row: (row["f1"], row["roc_auc"], -row["ece_after"]),
        reverse=True,
    )
    summary = {
        "results": results,
        "official_ranking": [row["strategy"] for row in ranked],
        "smoke_only": [
            row["strategy"] for row in results if row.get("run_type") == "smoke"
        ],
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_summary_figure(results)
    return summary


def save_summary_figure(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [
        row["strategy"] + ("*" if row.get("run_type") == "smoke" else "")
        for row in results
    ]
    metrics = ["f1", "roc_auc", "pr_auc", "specificity"]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(11, 1.5 * len(names)), 6))
    for index, metric in enumerate(metrics):
        ax.bar(
            x + (index - 1.5) * width,
            [row[metric] for row in results],
            width,
            label=metric,
        )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title("Task 11 Experiment Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "experiment_summary.png", dpi=160)
    plt.close(fig)


def run_ablation(smoke=False):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_classification_2025_task10.py"),
        "all",
    ]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["evaluate", "summary", "ablation", "all"],
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        choices=[*PREPROCESSING, *ADAPTATION],
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.command in {"ablation", "all"}:
        run_ablation(args.smoke)
    if args.command in {"evaluate", "all"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        strategies = args.strategies or [*PREPROCESSING, *ADAPTATION]
        for strategy in strategies:
            result = evaluate_strategy(strategy, device)
            print(strategy, result["f1"], result["roc_auc"], result["ece_after"])
    if args.command in {"summary", "all"}:
        print(json.dumps(write_summary(collect_results()), indent=2))


if __name__ == "__main__":
    main()
