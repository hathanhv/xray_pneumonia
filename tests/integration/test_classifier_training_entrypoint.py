import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


class ClassifierTrainingEntrypointTests(unittest.TestCase):
    def test_baseline_runs_end_to_end_without_notebook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            for split, count in (("train", 6), ("test", 2)):
                for class_name, color in (("NORMAL", 40), ("PNEUMONIA", 210)):
                    directory = data_dir / split / class_name
                    directory.mkdir(parents=True)
                    for index in range(count):
                        Image.new(
                            "RGB",
                            (40, 32),
                            color=(color, color, color),
                        ).save(directory / f"{class_name}_{index}.png")

            output_root = root / "outputs"
            config = {
                "experiment": {
                    "name": "smoke_2018",
                    "output_root": str(output_root),
                },
                "reproducibility": {
                    "seed": 42,
                    "deterministic": True,
                    "warn_only": True,
                },
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
                    "name": "mobilenet_v2",
                    "num_classes": 2,
                    "pretrained": False,
                    "init_checkpoint_path": None,
                    "dropout": 0.2,
                    "finetune_mode": "head",
                    "unfreeze_blocks": 1,
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
                "training": {
                    "epochs": 1,
                    "optimizer": {
                        "name": "adam",
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                    },
                    "scheduler": {
                        "name": "reduce_on_plateau",
                        "mode": "min",
                        "factor": 0.1,
                        "patience": 1,
                    },
                    "loss": {
                        "name": "cross_entropy",
                        "label_smoothing": 0.0,
                    },
                    "early_stopping": {
                        "monitor": "val_loss",
                        "mode": "min",
                        "patience": 0,
                        "constraints": [],
                    },
                },
                "return_metadata": True,
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = subprocess.run(
                [
                    str(PYTHON),
                    "scripts/train_classifier.py",
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
            run_dir = output_root / "smoke_2018" / "smoke"
            self.assertTrue((run_dir / "checkpoints" / "best_model.pth").exists())
            self.assertTrue((run_dir / "checkpoints" / "last_model.pth").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue(
                (run_dir / "evaluation" / "confusion_matrix.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
