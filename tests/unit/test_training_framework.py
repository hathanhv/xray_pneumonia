import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    CheckpointManager,
    ClassificationTrainer,
    MonitorConstraint,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
)


class TrainingFrameworkTests(unittest.TestCase):
    def _loaders(self):
        inputs = torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
        return loader, loader

    def test_classification_trainer_metrics_and_loss_aggregation(self):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        optimizer = build_optimizer(
            model,
            {"name": "adam", "learning_rate": 0.0},
        )
        trainer = ClassificationTrainer(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device="cpu",
        )
        loader, _ = self._loaders()
        metrics = trainer.run_epoch(loader, training=False)
        self.assertGreater(metrics["loss"], 0.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual((metrics["tp"], metrics["tn"]), (2, 2))
        self.assertEqual((metrics["fp"], metrics["fn"]), (0, 0))

    def test_constraint_requires_recall_before_checkpoint_improves(self):
        constraint = MonitorConstraint("val_recall", ">=", 0.95)
        early_stopping = build_early_stopping(
            {
                "monitor": "val_loss",
                "mode": "min",
                "patience": 3,
                "constraints": [
                    {
                        "metric": "val_recall",
                        "operator": ">=",
                        "value": 0.95,
                    }
                ],
            }
        )
        self.assertFalse(constraint.satisfied({"val_recall": 0.90}))
        improved, _ = early_stopping.update(
            1,
            {"val_loss": 0.1, "val_recall": 0.90},
        )
        self.assertFalse(improved)
        improved, _ = early_stopping.update(
            2,
            {"val_loss": 0.2, "val_recall": 0.96},
        )
        self.assertTrue(improved)
        self.assertEqual(early_stopping.best_epoch, 2)

    def test_optimizer_parameter_groups_and_all_schedulers(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Linear(4, 2),
        )
        optimizer = build_optimizer(
            model,
            {
                "name": "adamw",
                "learning_rate": 0.001,
                "parameter_groups": [
                    {"prefixes": ["1"], "lr": 0.01},
                ],
            },
        )
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        for config, expected in (
            ({"name": "cosine", "t_max": 3}, "CosineAnnealingLR"),
            ({"name": "step", "step_size": 2}, "StepLR"),
            ({"name": "reduce_on_plateau"}, "ReduceLROnPlateau"),
        ):
            scheduler = build_scheduler(optimizer, config)
            self.assertEqual(type(scheduler).__name__, expected)

    def test_best_last_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_loader, val_loader = self._loaders()
            model = torch.nn.Linear(2, 2)
            optimizer = build_optimizer(
                model,
                {"name": "adamw", "learning_rate": 0.01},
            )
            scheduler = build_scheduler(
                optimizer,
                {"name": "step", "step_size": 1, "gamma": 0.5},
            )
            early_stopping = build_early_stopping(
                {
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": 0,
                }
            )
            manager = CheckpointManager(
                root / "best_model.pth",
                root / "last_model.pth",
            )
            trainer = ClassificationTrainer(
                model=model,
                criterion=torch.nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint_manager=manager,
                device="cpu",
            )
            trainer.fit(train_loader, val_loader, epochs=2)
            self.assertTrue(manager.best_path.exists())
            self.assertTrue(manager.last_path.exists())

            resumed_model = torch.nn.Linear(2, 2)
            resumed_optimizer = build_optimizer(
                resumed_model,
                {"name": "adamw", "learning_rate": 0.01},
            )
            resumed_scheduler = build_scheduler(
                resumed_optimizer,
                {"name": "step", "step_size": 1, "gamma": 0.5},
            )
            resumed_early_stopping = build_early_stopping(
                {
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": 0,
                }
            )
            resumed = ClassificationTrainer(
                model=resumed_model,
                criterion=torch.nn.CrossEntropyLoss(),
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                early_stopping=resumed_early_stopping,
                checkpoint_manager=manager,
                device="cpu",
            )
            history = resumed.fit(
                train_loader,
                val_loader,
                epochs=3,
                resume_from=manager.last_path,
            )
            self.assertEqual(resumed.start_epoch, 3)
            self.assertEqual(history[-1]["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
