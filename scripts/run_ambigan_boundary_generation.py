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
from src.pipelines.ambigan_boundary import AmbiGANBoundaryPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    args = parser.parse_args()
    overrides = {"device": args.device} if args.device else None
    config = load_config(args.config, overrides=overrides)
    seed_everything(int(config.get("seed", 42)))
    device = config.get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    try:
        with create_experiment(config, run_id=args.run_id) as experiment:
            result = AmbiGANBoundaryPipeline(
                config,
                experiment,
                device,
            ).run()
            print(f"Boundary images: {result['boundary']['saved_count']}")
            print(f"Run directory: {experiment.run_dir}")
    except ValueError as error:
        if "No classification images found" not in str(error):
            raise
        data_dir = config["dataset"]["data_dir"]
        raise SystemExit(
            f"{error}\n\n"
            "Expected 2018 dataset layout:\n"
            f"{data_dir}\\train\\NORMAL\\*.jpeg\n"
            f"{data_dir}\\train\\PNEUMONIA\\*.jpeg\n"
            f"{data_dir}\\test\\NORMAL\\*.jpeg\n"
            f"{data_dir}\\test\\PNEUMONIA\\*.jpeg\n\n"
            "Add the images or change dataset.data_dir in the YAML."
        ) from error


if __name__ == "__main__":
    main()
