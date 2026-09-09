import pickle
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.classifier.calibration import TemperatureScaler
from src.classifier.dataset import ClassificationRecord, ManifestClassificationDataset
from src.classifier.ensemble import (
    search_ensemble_weight,
    select_recall_constrained_method,
)
from src.classifier.hard_negative import mine_hard_negatives
from src.classifier.losses import resolve_class_weights
from src.classifier.samplers import build_sampler
from src.classifier.thresholding import metrics_at_threshold, tune_threshold
from src.classifier.tta import build_tta_views
from src.training import RecallConstrainedEarlyStopping, build_early_stopping


class AdvancedClassificationTests(unittest.TestCase):
    def test_manual_and_balanced_class_weighting(self):
        manual = resolve_class_weights(
            {"strategy": "manual", "values": [3.0, 1.0]},
            labels=[0, 1, 1, 1],
            num_classes=2,
        )
        balanced = resolve_class_weights(
            {"strategy": "balanced"},
            labels=[0, 1, 1, 1],
            num_classes=2,
        )
        self.assertTrue(torch.equal(manual, torch.tensor([3.0, 1.0])))
        self.assertGreater(float(balanced[0]), float(balanced[1]))

    def test_weighted_sampler_accepts_explicit_class_weights(self):
        sampler = build_sampler(
            [0, 1, 1],
            {
                "name": "weighted",
                "class_weights": [4.0, 1.0],
                "num_samples": 9,
            },
        )
        self.assertEqual(sampler.num_samples, 9)
        self.assertEqual(sampler.weights.tolist(), [4.0, 1.0, 1.0])

    def test_threshold_tuning_honors_recall_constraint(self):
        targets = torch.tensor([0, 0, 1, 1])
        probabilities = torch.tensor([0.1, 0.6, 0.7, 0.9])
        result = tune_threshold(
            targets,
            probabilities,
            {
                "start": 0.1,
                "stop": 0.9,
                "step": 0.1,
                "min_recall": 1.0,
                "objective": "min_fp",
            },
        )
        self.assertTrue(result.constraint_satisfied)
        self.assertAlmostEqual(result.threshold, 0.7, places=6)
        self.assertEqual(result.metrics["fp"], 0)

    def test_temperature_scaling_returns_positive_temperature(self):
        logits = torch.tensor([[4.0, -1.0], [-1.0, 4.0], [2.0, 0.0]])
        targets = torch.tensor([0, 1, 1])
        scaler = TemperatureScaler(1.5)
        temperature = scaler.fit(logits, targets, max_iter=10)
        self.assertGreater(temperature, 0.0)
        self.assertEqual(tuple(scaler(logits).shape), (3, 2))

    def test_tta_has_eight_deterministic_views(self):
        views = build_tta_views(image_size=224)
        self.assertEqual(len(views), 8)
        pickle.dumps(views)

    def test_ensemble_and_final_recall_constrained_selection(self):
        targets = torch.tensor([0, 0, 1, 1])
        first = torch.tensor([0.2, 0.8, 0.8, 0.9])
        second = torch.tensor([0.1, 0.4, 0.7, 0.8])
        ensemble = search_ensemble_weight(
            targets,
            first,
            second,
            weights=[0.0, 1.0],
            min_recall=1.0,
        )
        self.assertEqual(ensemble.weight_first, 0.0)
        methods = [
            {"name": "a", "metrics": metrics_at_threshold(targets, first, 0.5)},
            {"name": "b", "metrics": metrics_at_threshold(targets, second, 0.5)},
        ]
        selected = select_recall_constrained_method(
            methods,
            min_recall=1.0,
            min_specificity=0.5,
            min_accuracy=0.5,
        )
        self.assertEqual(selected["name"], "b")

    def test_recall_constrained_early_stopping_keeps_fallback(self):
        stopping = build_early_stopping(
            {
                "selection": "recall_constrained",
                "monitor": "val_f1",
                "mode": "max",
                "fallback_monitor": "val_specificity",
                "constraints": [
                    {"metric": "val_recall", "operator": ">=", "value": 0.95}
                ],
                "patience": 0,
            }
        )
        self.assertIsInstance(stopping, RecallConstrainedEarlyStopping)
        improved, _ = stopping.update(
            1,
            {"val_f1": 0.8, "val_recall": 0.9, "val_specificity": 0.7},
        )
        self.assertTrue(improved)
        improved, _ = stopping.update(
            2,
            {"val_f1": 0.75, "val_recall": 0.96, "val_specificity": 0.5},
        )
        self.assertTrue(improved)
        self.assertTrue(stopping.has_eligible_model)

    def test_hard_negative_mining_only_selects_true_normal(self):
        class AlwaysPneumonia(torch.nn.Module):
            def forward(self, inputs):
                return torch.tensor(
                    [[0.0, 2.0]] * len(inputs),
                    device=inputs.device,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = []
            for index, (name, label) in enumerate(
                [("NORMAL", 0), ("PNEUMONIA", 1), ("NORMAL", 0)]
            ):
                path = root / f"{index}.png"
                Image.new("RGB", (16, 16), color=(index * 40,) * 3).save(path)
                records.append(
                    ClassificationRecord(
                        image_path=path,
                        label=label,
                        class_name=name,
                    )
                )
            dataset = ManifestClassificationDataset(records)
            result = mine_hard_negatives(
                AlwaysPneumonia(),
                dataset,
                device="cpu",
                threshold=0.3,
                batch_size=2,
            )
            self.assertEqual(result.indices, (0, 2))


if __name__ == "__main__":
    unittest.main()
