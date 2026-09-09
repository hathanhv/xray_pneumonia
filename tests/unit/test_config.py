from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.core.config import ConfigError, deep_merge, load_config


class ConfigTests(unittest.TestCase):
    def test_deep_merge_keeps_nested_values(self):
        merged = deep_merge(
            {"training": {"epochs": 10, "lr": 1e-4}},
            {"training": {"epochs": 3}},
        )
        self.assertEqual(merged["training"]["epochs"], 3)
        self.assertEqual(merged["training"]["lr"], 1e-4)

    def test_loads_defaults_overrides_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs" / "base").mkdir(parents=True)
            (root / "configs" / "experiments").mkdir(parents=True)
            (root / "configs" / "base" / "default.yaml").write_text(
                "dataset:\n  data_dir: data/raw\ntraining:\n  epochs: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "experiments" / "test.yaml").write_text(
                "defaults:\n  - base/default\ntraining:\n  epochs: 3\n",
                encoding="utf-8",
            )

            config = load_config(
                "configs/experiments/test.yaml",
                project_root=root,
                required_keys=("dataset.data_dir", "training.epochs"),
            )

            self.assertEqual(config["training"]["epochs"], 3)
            self.assertEqual(
                Path(config["dataset"]["data_dir"]),
                (root / "data" / "raw").resolve(),
            )

    def test_rejects_missing_required_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs" / "test.yaml").write_text(
                "experiment:\n  name: demo\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(
                    "configs/test.yaml",
                    project_root=root,
                    required_keys=("training.epochs",),
                )

    def test_rejects_circular_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs" / "a.yaml").write_text(
                "defaults: [b]\n",
                encoding="utf-8",
            )
            (root / "configs" / "b.yaml").write_text(
                "defaults: [a]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config("configs/a.yaml", project_root=root)


if __name__ == "__main__":
    unittest.main()
