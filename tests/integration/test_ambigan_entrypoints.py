import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.classifier.model import build_mobilenet_v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


def _create_data(root):
    data_dir = root / "data"
    for split, count in (("train", 4), ("test", 2)):
        for class_name, color in (("NORMAL", 30), ("PNEUMONIA", 220)):
            directory = data_dir / split / class_name
            directory.mkdir(parents=True)
            for index in range(count):
                Image.new(
                    "RGB",
                    (32, 32),
                    color=(color, color, color),
                ).save(directory / f"{class_name}_{index}.png")
    return data_dir


def _base_config(root, data_dir):
    return {
        "experiment": {
            "name": "smoke",
            "output_root": str(root / "outputs"),
        },
        "reproducibility": {"seed": 42},
        "seed": 42,
        "dataset": {
            "data_dir": str(data_dir),
            "split": {
                "method": "sklearn_stratified",
                "val_fraction": 0.25,
                "seed": 42,
                "preserve_existing_test": True,
                "manifest_output": str(root / "manifest.csv"),
            },
        },
        "model": {
            "num_classes": 2,
            "pretrained": False,
            "dropout": 0.2,
            "finetune_mode": "head",
            "freeze_batchnorm": True,
        },
        "preprocessing": {"name": "resize", "size": 32},
        "input": {"grayscale": False, "normalize": True},
        "augmentation": {"name": "none"},
        "sampler": {"name": "random"},
        "dataloader": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
    }


class AmbiGANEntrypointTests(unittest.TestCase):
    def test_boundary_generation_runs_from_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = _create_data(root)
            oracle_path = root / "oracle.pth"
            oracle = build_mobilenet_v2(pretrained=False)
            torch.save(
                {"model_state_dict": oracle.state_dict()},
                oracle_path,
            )
            config = _base_config(root, data_dir)
            config["experiment"]["name"] = "ambigan_smoke"
            config["oracle"] = {
                "checkpoint_path": str(oracle_path),
                "image_size": 32,
            }
            config["ambigan"] = {
                "image_size": 16,
                "image_channels": 1,
                "latent_dim": 8,
                "generator_filters": 2,
                "discriminator_filters": 2,
                "gradient_clip": 1.0,
                "drop_last": False,
                "dcgan_training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "generator_lr": 0.0002,
                    "discriminator_lr": 0.0002,
                    "betas": [0.0, 0.999],
                },
                "ambiguity_training": {
                    "epochs": 1,
                    "generator_lr": 0.0001,
                    "discriminator_lr": 0.0002,
                    "betas": [0.0, 0.999],
                    "alpha": 0.2,
                    "variance": 0.1,
                    "oracle_temperature": 3.0,
                },
                "generation": {
                    "count": 2,
                    "ambiguity_threshold": 0.51,
                    "batch_size": 2,
                    "max_attempt_multiplier": 2,
                },
            }
            config_path = root / "ambigan.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = subprocess.run(
                [
                    str(PYTHON),
                    "scripts/run_ambigan_boundary_generation.py",
                    "--config",
                    str(config_path),
                    "--run-id",
                    "smoke",
                    "--device",
                    "cpu",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=240,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            run_dir = root / "outputs" / "ambigan_smoke" / "smoke"
            self.assertTrue(
                (run_dir / "checkpoints" / "ambigan_model.pth").exists()
            )
            self.assertTrue(
                (run_dir / "boundary_images" / "metadata.csv").exists()
            )

    def test_hubris_training_runs_from_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = _create_data(root)
            checkpoint_path = root / "base.pth"
            model = build_mobilenet_v2(pretrained=False)
            torch.save(
                {"model_state_dict": model.state_dict()},
                checkpoint_path,
            )
            boundary_dir = root / "boundary" / "NORMAL"
            boundary_dir.mkdir(parents=True)
            metadata_path = root / "boundary" / "metadata.csv"
            with metadata_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "filename",
                        "p_pneumonia",
                        "confusion_distance",
                    ],
                )
                writer.writeheader()
                for index, probability in enumerate((0.48, 0.52)):
                    filename = f"boundary_{index:06d}.png"
                    Image.new("RGB", (16, 16), color=(128,) * 3).save(
                        boundary_dir / filename
                    )
                    writer.writerow(
                        {
                            "filename": filename,
                            "p_pneumonia": probability,
                            "confusion_distance": abs(probability - 0.5),
                        }
                    )
            config = _base_config(root, data_dir)
            config["experiment"]["name"] = "hubris_smoke"
            config["hubris_training"] = {
                "base_checkpoint_path": str(checkpoint_path),
                "strategies": ["soft"],
                "boundary": {
                    "image_dir": str(boundary_dir),
                    "metadata_path": str(metadata_path),
                    "max_confusion_distance": 0.08,
                    "max_images": 2,
                    "oversample": 1,
                },
                "training": {
                    "epochs": 1,
                    "class_weighting": {
                        "strategy": "manual",
                        "values": [2.0, 0.5],
                    },
                    "loss": {"name": "soft_focal", "gamma": 2.0},
                    "optimizer": {
                        "name": "adam",
                        "learning_rate": 0.001,
                    },
                    "scheduler": {"name": "cosine", "t_max": 1},
                    "early_stopping": {
                        "selection": "recall_constrained",
                        "monitor": "val_f1",
                        "mode": "max",
                        "fallback_monitor": "val_specificity",
                        "fallback_mode": "max",
                        "patience": 0,
                        "constraints": [
                            {
                                "metric": "val_recall",
                                "operator": ">=",
                                "value": 0.5,
                            }
                        ],
                    },
                },
                "selection": {"min_recall": 0.0},
            }
            config_path = root / "hubris.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = subprocess.run(
                [
                    str(PYTHON),
                    "scripts/train_hubris_aware_classifier.py",
                    "--config",
                    str(config_path),
                    "--run-id",
                    "smoke",
                    "--device",
                    "cpu",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=240,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            run_dir = root / "outputs" / "hubris_smoke" / "smoke"
            self.assertTrue((run_dir / "checkpoints" / "best_model.pth").exists())
            self.assertTrue((run_dir / "hubris_comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
