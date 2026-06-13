import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.classifier.dataset import (
    CLASS_TO_IDX,
    ClassificationRecord,
    ManifestClassificationDataset,
    assert_no_patient_leakage,
    build_tensor_transform,
    get_eval_transforms,
    load_manifest,
    select_few_shot_records,
    split_records,
)
from src.classifier.losses import build_loss, compute_class_weights
from src.classifier.evaluate import (
    build_classification_report,
    format_classification_report,
    format_confusion_matrix,
)
from src.classifier.model import (
    build_mobilenet_v2,
    configure_finetuning,
    load_classifier_checkpoint,
)
from src.classifier.preprocessing import build_preprocessing
from src.classifier.samplers import build_sampler


class ClassifierCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _image(self, name, size=(40, 20), value=100):
        path = self.root / name
        Image.fromarray(np.full((size[1], size[0]), value, dtype=np.uint8)).save(path)
        return path

    def test_few_shot_selection_is_balanced_and_deterministic(self):
        records = []
        for label, class_name in enumerate(("NORMAL", "PNEUMONIA")):
            for index in range(5):
                records.append(
                    ClassificationRecord(
                        image_path=self._image(f"{class_name}_{index}.png"),
                        label=label,
                        class_name=class_name,
                        sample_id=f"{class_name}_{index}",
                    )
                )
        first = select_few_shot_records(records, shots_per_class=2, seed=7)
        second = select_few_shot_records(records, shots_per_class=2, seed=7)
        self.assertEqual([record.sample_id for record in first], [
            record.sample_id for record in second
        ])
        self.assertEqual([record.label for record in first].count(0), 2)
        self.assertEqual([record.label for record in first].count(1), 2)

    def test_manifest_dataset_returns_tensor_label_and_metadata(self):
        image_path = self._image("sample.png")
        manifest_path = self.root / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_path", "label", "split", "patient_id"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_path": image_path.name,
                    "label": "NORMAL",
                    "split": "train",
                    "patient_id": "patient-1",
                }
            )
        records = load_manifest(manifest_path)
        dataset = ManifestClassificationDataset(
            records,
            preprocessing=build_preprocessing({"name": "resize", "size": 32}),
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
        self.assertEqual(sample["label"], CLASS_TO_IDX["NORMAL"])
        self.assertEqual(sample["metadata"]["patient_id"], "patient-1")

    def test_patient_level_stratified_split_has_no_leakage(self):
        records = []
        for label, class_name in enumerate(CLASS_TO_IDX):
            for index in range(10):
                records.append(
                    ClassificationRecord(
                        image_path=self._image(f"{class_name}_{index}.png"),
                        label=label,
                        class_name=class_name,
                        patient_id=f"{class_name}-patient-{index}",
                    )
                )
        splits = split_records(
            records,
            val_fraction=0.2,
            test_fraction=0.2,
            preserve_existing_test=False,
        )
        assert_no_patient_leakage(splits)
        for split in ("train", "val", "test"):
            self.assertEqual({record.label for record in splits[split]}, {0, 1})

    def test_roi_fallback_and_resize_with_padding(self):
        image = Image.open(self._image("roi.png", size=(80, 40)))
        strategy = build_preprocessing(
            {
                "name": "composite",
                "strategies": [
                    {"name": "lung_roi", "fallback": {"name": "raw"}},
                    {"name": "resize_with_padding", "size": 64},
                ],
            }
        )
        result = strategy(image, {})
        self.assertEqual(result.image.size, (64, 64))
        self.assertEqual(result.metadata["roi_source"], "fallback")

    def test_losses_sampler_and_finetuning_factories(self):
        weights = compute_class_weights([0, 0, 1], 2)
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        for name in ("cross_entropy", "weighted_cross_entropy", "focal"):
            loss = build_loss({"name": name}, class_weights=weights)(logits, labels)
            self.assertTrue(torch.isfinite(loss))
        soft = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        for name in ("soft_cross_entropy", "soft_focal"):
            loss = build_loss({"name": name})(logits, soft)
            self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(list(build_sampler([0, 0, 1], "class_balanced"))), 3)

        model = build_mobilenet_v2(pretrained=False, dropout=0.35)
        info = configure_finetuning(
            model,
            mode="head",
            freeze_batchnorm=True,
        )
        self.assertEqual(model.classifier[0].p, 0.35)
        self.assertEqual(info["selected_mode"], "head")
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))

    def test_classification_report_and_confusion_matrix_are_printable(self):
        metrics = {
            "accuracy": 0.75,
            "precision": 2 / 3,
            "recall": 1.0,
            "specificity": 0.5,
            "f1": 0.8,
            "tp": 2,
            "tn": 1,
            "fp": 1,
            "fn": 0,
        }
        report = build_classification_report(metrics)
        self.assertIn("PNEUMONIA", format_classification_report(report))
        matrix = format_confusion_matrix(metrics)
        self.assertIn("true_NORMAL", matrix)
        self.assertIn("true_PNEUMONIA", matrix)

    def test_legacy_checkpoint_load_is_strict_and_transform_is_exact(self):
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_path = (
            project_root
            / "checkpoints"
            / "pneumonia_classifier"
            / "mobilenet_2025_lung_crop_corrected.pth"
        )
        model = build_mobilenet_v2(pretrained=False)
        _, report = load_classifier_checkpoint(
            model,
            checkpoint_path,
            device="cpu",
            strict=True,
        )
        self.assertEqual(report.missing_keys, ())
        self.assertEqual(report.unexpected_keys, ())
        self.assertEqual(report.class_to_idx, CLASS_TO_IDX)

        channels = np.stack(
            [
                np.arange(256, dtype=np.uint8).reshape(16, 16),
                np.full((16, 16), 80, dtype=np.uint8),
                np.full((16, 16), 180, dtype=np.uint8),
            ],
            axis=2,
        )
        image = Image.fromarray(channels)
        expected = get_eval_transforms(224)(image)
        prepared = build_preprocessing(
            {"name": "legacy_classifier", "size": 224}
        )(image).image
        actual = build_tensor_transform()(prepared)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
