import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


class AdvancedClassificationEntrypointTests(unittest.TestCase):
    def test_class_weighting_threshold_yaml_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            for split, count in (("train", 6), ("test", 2)):
                for class_name, color in (("NORMAL", 30), ("PNEUMONIA", 220)):
                    directory = data_dir / split / class_name
                    directory.mkdir(parents=True)
                    for index in range(count):
                        Image.new(
                            "RGB",
                            (32, 32),
                            color=(color, color, color),
                        ).save(directory / f"{class_name}_{index}.png")

            config = {
                "experiment": {
                    "name": "advanced_smoke",
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
                "advanced": {
                    "protocol": "class_weighting_threshold",
                    "training": {
                        "epochs": 1,
                        "class_weighting": {
                            "strategy": "manual",
                            "values": [3.0, 1.0],
                        },
                        "loss": {"name": "weighted_cross_entropy"},
                        "optimizer": {
                            "name": "adam",
                            "learning_rate": 0.001,
                        },
                        "scheduler": {"name": "cosine", "t_max": 1},
                        "early_stopping": {
                            "monitor": "val_accuracy",
                            "mode": "max",
                            "patience": 0,
                            "constraints": [],
                        },
                    },
                    "threshold_tuning": {
                        "start": 0.1,
                        "stop": 0.9,
                        "step": 0.1,
                        "min_recall": 0.5,
                        "objective": "min_fp",
                    },
                    "model_selection": {
                        "min_recall": 0.5,
                        "min_specificity": 0.0,
                        "min_accuracy": 0.0,
                    },
                },
            }
            config_path = root / "advanced.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = subprocess.run(
                [
                    str(PYTHON),
                    "scripts/run_advanced_classification.py",
                    "--config",
                    str(config_path),
                    "--run-id",
                    "smoke",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            run_dir = root / "outputs" / "advanced_smoke" / "smoke"
            self.assertTrue((run_dir / "checkpoints" / "best_model.pth").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "training_log.csv").exists())
            self.assertTrue(
                (run_dir / "evaluation" / "confusion_matrix.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
