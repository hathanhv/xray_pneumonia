from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import create_experiment, load_config, seed_everything


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "dummy.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a lightweight experiment to verify the project foundation."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config,
        required_keys=(
            "experiment.name",
            "reproducibility.seed",
            "training.epochs",
        ),
    )

    reproducibility = config["reproducibility"]
    seed_info = seed_everything(
        reproducibility["seed"],
        deterministic=reproducibility.get("deterministic", True),
        warn_only=reproducibility.get("warn_only", True),
    )
    with create_experiment(config, run_id=args.run_id) as experiment:
        experiment.logger.info("Starting dummy experiment")

        epochs = int(config["training"]["epochs"])
        for epoch in range(1, epochs + 1):
            metrics = {
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "val_loss": 1.0 / (epoch + 0.5),
            }
            experiment.metric_logger.log(metrics)
            experiment.logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f",
                epoch,
                metrics["train_loss"],
                metrics["val_loss"],
            )

        experiment.save_metrics(
            {
                "status": "completed",
                "epochs": epochs,
                "final_val_loss": 1.0 / (epochs + 0.5),
                "reproducibility": seed_info,
            }
        )
        experiment.logger.info("Dummy experiment completed: %s", experiment.run_dir)
        print(experiment.run_dir)


if __name__ == "__main__":
    main()
