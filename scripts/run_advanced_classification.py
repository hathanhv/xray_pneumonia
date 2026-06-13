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
from src.pipelines.advanced_classification import AdvancedClassificationPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {"device": args.device} if args.device else None
    config = load_config(args.config, overrides=overrides)
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
    with create_experiment(config, run_id=args.run_id) as experiment:
        pipeline = AdvancedClassificationPipeline(config, experiment, device)
        pipeline.run()
        print(f"\nRun directory: {experiment.run_dir}")


if __name__ == "__main__":
    main()
