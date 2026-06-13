from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.core.experiment import create_experiment
from src.core.logging import CSVMetricLogger


class ExperimentTests(unittest.TestCase):
    def test_creates_complete_run_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "experiment": {
                    "name": "demo",
                    "output_root": "outputs/experiments",
                }
            }
            context = create_experiment(
                config,
                project_root=root,
                run_id="fixed_run",
            )
            context.metric_logger.log({"epoch": 1, "loss": 0.5})
            context.save_metrics({"accuracy": 0.9})

            self.assertTrue(context.config_path.exists())
            self.assertTrue(context.training_log_path.exists())
            self.assertTrue(context.metrics_path.exists())
            self.assertTrue(context.checkpoint_dir.is_dir())
            self.assertTrue(context.figure_dir.is_dir())
            self.assertTrue(context.error_analysis_dir.is_dir())

            with context.training_log_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["epoch"], "1")
            self.assertEqual(
                json.loads(context.metrics_path.read_text(encoding="utf-8")),
                {"accuracy": 0.9},
            )
            context.close()

    def test_refuses_to_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"experiment": {"name": "demo"}}
            context = create_experiment(config, project_root=root, run_id="same")
            with self.assertRaises(FileExistsError):
                create_experiment(config, project_root=root, run_id="same")
            context.close()

    def test_csv_logger_rejects_schema_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = CSVMetricLogger(Path(directory) / "metrics.csv")
            logger.log({"epoch": 1, "loss": 0.5})
            with self.assertRaises(ValueError):
                logger.log({"epoch": 2, "accuracy": 0.9})


if __name__ == "__main__":
    unittest.main()
