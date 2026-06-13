from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.experiment import create_experiment
from src.core.reproducibility import seed_everything
from src.pipelines.hubris_training import HubrisAwareTrainingPipeline


def _latest_existing_path(project_root, patterns):
    candidates = []
    for pattern in patterns:
        candidates.extend(project_root.glob(pattern))
    existing = [path for path in candidates if path.exists()]
    return max(existing, key=lambda path: path.stat().st_mtime, default=None)


def resolve_hubris_inputs(config, project_root=PROJECT_ROOT):
    hubris_config = config["hubris_training"]
    boundary_config = hubris_config["boundary"]
    project_root = Path(project_root)

    checkpoint = Path(hubris_config["base_checkpoint_path"])
    if not checkpoint.exists():
        discovered = _latest_existing_path(
            project_root,
            [
                "outputs/experiments/cls_2018_hard_negative_mining/"
                "*/stages/hard_negative_finetuning/best_model.pth",
                "outputs/experiments/cls_2018_hard_negative_mining/"
                "*/checkpoints/best_model.pth",
            ],
        )
        if discovered is not None:
            hubris_config["base_checkpoint_path"] = str(discovered)

    image_dir = Path(boundary_config["image_dir"])
    metadata_path = Path(boundary_config["metadata_path"])
    if not image_dir.exists() or not metadata_path.exists():
        discovered_metadata = _latest_existing_path(
            project_root,
            [
                "outputs/experiments/ambigan_boundary_generation/"
                "*/boundary_images/metadata.csv",
                "outputs/experiments/ambigan_boundary_generation_smoke/"
                "*/boundary_images/metadata.csv",
            ],
        )
        if discovered_metadata is not None:
            boundary_dir = discovered_metadata.parent
            discovered_images = boundary_dir / "NORMAL"
            if discovered_images.exists():
                boundary_config["image_dir"] = str(discovered_images)
                boundary_config["metadata_path"] = str(discovered_metadata)

    return config


def validate_hubris_inputs(config, project_root=PROJECT_ROOT):
    resolve_hubris_inputs(config, project_root=project_root)
    hubris_config = config["hubris_training"]
    boundary_config = hubris_config["boundary"]
    required_paths = {
        "base checkpoint": Path(hubris_config["base_checkpoint_path"]),
        "boundary image directory": Path(boundary_config["image_dir"]),
        "boundary metadata": Path(boundary_config["metadata_path"]),
    }
    missing = [
        f"- {label}: {path}"
        for label, path in required_paths.items()
        if not path.exists()
    ]
    if missing:
        details = "\n".join(missing)
        raise FileNotFoundError(
            "Hubris-aware training inputs are missing:\n"
            f"{details}\n"
            "Run the hard-negative stage with:\n"
            "  python scripts/run_advanced_classification.py "
            "--config configs/experiments/"
            "cls_2018_hard_negative_mining.yaml\n"
            "Then run notebook 07 before retrying notebook 08."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    args = parser.parse_args()
    overrides = {"device": args.device} if args.device else None
    config = load_config(args.config, overrides=overrides)
    validate_hubris_inputs(config)
    seed_everything(int(config.get("seed", 42)))
    device = config.get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    with create_experiment(config, run_id=args.run_id) as experiment:
        HubrisAwareTrainingPipeline(
            config,
            experiment,
            device,
        ).run()
        print(f"\nRun directory: {experiment.run_dir}")


if __name__ == "__main__":
    main()
