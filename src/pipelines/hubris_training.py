from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from src.ambigan.boundary import BoundaryDataset, OneHotDatasetWrapper
from src.ambigan.hubris import compute_hubris
from src.classifier.dataset import (
    CLASS_TO_IDX,
    create_loaders_from_config,
)
from src.classifier.evaluate import (
    evaluate_model,
    format_classification_report,
    format_confusion_matrix,
    save_evaluation_artifacts,
)
from src.classifier.losses import build_loss, resolve_class_weights
from src.classifier.model import (
    build_mobilenet_v2_from_config,
    load_classifier_checkpoint,
)
from src.core.logging import CSVMetricLogger
from src.training import (
    CheckpointManager,
    SoftLabelClassificationTrainer,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
)


class HubrisAwareTrainingPipeline:
    def __init__(self, config, experiment, device):
        self.config = deepcopy(config)
        self.experiment = experiment
        self.device = torch.device(device)
        self.loaders, self.datasets = create_loaders_from_config(config)
        self.hubris_config = config["hubris_training"]

    def _build_model(self):
        model_config = deepcopy(self.config["model"])
        model_config["pretrained"] = False
        model, metadata = build_mobilenet_v2_from_config(model_config)
        load_classifier_checkpoint(
            model,
            self.hubris_config["base_checkpoint_path"],
            device="cpu",
            strict=True,
        )
        metadata["base_checkpoint_path"] = str(
            self.hubris_config["base_checkpoint_path"]
        )
        return model, metadata

    def _boundary_transform(self, training):
        image_size = int(
            self.config.get("preprocessing", {}).get("size", 224)
        )
        operations = [transforms.Resize((image_size, image_size))]
        if training:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        return transforms.Compose(operations)

    def _boundary_dataset(self, label_strategy, training):
        boundary = self.hubris_config["boundary"]
        dataset = BoundaryDataset(
            boundary["image_dir"],
            boundary["metadata_path"],
            transform=self._boundary_transform(training),
            max_confusion_distance=boundary.get(
                "max_confusion_distance",
                0.08,
            ),
            max_images=boundary.get("max_images"),
            label_strategy=label_strategy,
        )
        if not dataset:
            raise ValueError(
                "No boundary images passed the configured metadata filter"
            )
        return dataset

    def _combined_loader(self, label_strategy):
        boundary_dataset = self._boundary_dataset(label_strategy, True)
        oversample = int(
            self.hubris_config["boundary"].get("oversample", 1)
        )
        original = OneHotDatasetWrapper(self.datasets["train"])
        combined = ConcatDataset(
            [original] + [boundary_dataset] * oversample
        )
        loader_config = self.config.get("dataloader", {})
        return (
            DataLoader(
                combined,
                batch_size=int(loader_config.get("batch_size", 32)),
                shuffle=True,
                num_workers=int(loader_config.get("num_workers", 0)),
                pin_memory=bool(loader_config.get("pin_memory", True)),
            ),
            boundary_dataset,
        )

    def _train_branch(self, name, label_strategy):
        model, metadata = self._build_model()
        branch_config = self.hubris_config["training"]
        train_loader, boundary_dataset = self._combined_loader(
            label_strategy
        )
        weights = resolve_class_weights(
            branch_config.get("class_weighting"),
            labels=self.datasets["train"].targets,
            num_classes=len(CLASS_TO_IDX),
        )
        if weights is not None:
            weights = weights.to(self.device)
        criterion = build_loss(
            branch_config["loss"],
            class_weights=weights,
        ).to(self.device)
        optimizer = build_optimizer(model, branch_config["optimizer"])
        scheduler = build_scheduler(
            optimizer,
            branch_config.get("scheduler"),
        )
        early_stopping = build_early_stopping(
            branch_config["early_stopping"]
        )
        branch_dir = self.experiment.run_dir / "stages" / name
        manager = CheckpointManager(
            branch_dir / "best_model.pth",
            branch_dir / "last_model.pth",
        )
        trainer = SoftLabelClassificationTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            checkpoint_manager=manager,
            device=self.device,
            metric_logger=CSVMetricLogger(branch_dir / "training_log.csv"),
            logger=self.experiment.logger,
            checkpoint_metadata={
                **metadata,
                "boundary_label_strategy": label_strategy,
                "boundary_count": len(boundary_dataset),
            },
        )
        history = trainer.fit(
            train_loader,
            self.loaders["val"],
            epochs=int(branch_config["epochs"]),
        )
        manager.resume(manager.best_path, model=model, device=self.device)
        validation_metrics = evaluate_model(
            model,
            self.loaders["val"],
            criterion,
            self.device,
        )
        test_metrics, test_details = evaluate_model(
            model,
            self.loaders["test"],
            criterion,
            self.device,
            return_details=True,
        )
        boundary_eval = self._boundary_dataset("soft", False)
        loader_config = self.config.get("dataloader", {})
        boundary_loader = DataLoader(
            boundary_eval,
            batch_size=int(loader_config.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(loader_config.get("num_workers", 0)),
        )
        hubris, probabilities = compute_hubris(
            model,
            boundary_loader,
            self.device,
        )
        return {
            "name": name,
            "model": model,
            "checkpoint": str(manager.best_path),
            "history": history,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "details": test_details,
            "hubris": hubris,
            "boundary_probabilities": probabilities.tolist(),
            "boundary_count": len(boundary_dataset),
        }

    def _baseline(self):
        model, _metadata = self._build_model()
        criterion = build_loss("cross_entropy").to(self.device)
        validation_metrics = evaluate_model(
            model,
            self.loaders["val"],
            criterion,
            self.device,
        )
        test_metrics, test_details = evaluate_model(
            model,
            self.loaders["test"],
            criterion,
            self.device,
            return_details=True,
        )
        boundary_eval = self._boundary_dataset("soft", False)
        loader_config = self.config.get("dataloader", {})
        boundary_loader = DataLoader(
            boundary_eval,
            batch_size=int(loader_config.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(loader_config.get("num_workers", 0)),
        )
        hubris, probabilities = compute_hubris(
            model,
            boundary_loader,
            self.device,
        )
        return {
            "name": "hard_negative_baseline",
            "model": model,
            "checkpoint": str(
                self.hubris_config["base_checkpoint_path"]
            ),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "details": test_details,
            "hubris": hubris,
            "boundary_probabilities": probabilities.tolist(),
            "boundary_count": len(boundary_eval),
        }

    def _select(self, candidates):
        min_recall = float(
            self.hubris_config.get("selection", {}).get(
                "min_recall",
                0.95,
            )
        )
        eligible = [
            candidate
            for candidate in candidates
            if candidate["validation_metrics"]["recall"] >= min_recall
        ]
        pool = eligible or candidates
        return min(
            pool,
            key=lambda candidate: (
                candidate["hubris"],
                -candidate["validation_metrics"]["specificity"],
                -candidate["validation_metrics"]["f1"],
            ),
        )

    def run(self):
        candidates = [self._baseline()]
        strategies = self.hubris_config.get(
            "strategies",
            ["hard_normal", "soft"],
        )
        if "hard_normal" in strategies:
            candidates.append(
                self._train_branch(
                    "boundary_hard_label",
                    "hard_normal",
                )
            )
        if "soft" in strategies:
            candidates.append(
                self._train_branch(
                    "boundary_soft_label",
                    "soft",
                )
            )
        history_rows = []
        for candidate in candidates:
            for row in candidate.get("history", []):
                history_rows.append({"method": candidate["name"], **row})
        if history_rows:
            fieldnames = []
            for row in history_rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with self.experiment.training_log_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(history_rows)
        selected = self._select(candidates)
        final_checkpoint = {
            "model_name": "mobilenet_v2",
            "model_state_dict": selected["model"].state_dict(),
            "class_to_idx": CLASS_TO_IDX,
            "metrics": {
                "validation": selected["validation_metrics"],
                "test": selected["test_metrics"],
            },
            "metadata": {
                "method": selected["name"],
                "hubris": selected["hubris"],
                "base_checkpoint_path": str(
                    self.hubris_config["base_checkpoint_path"]
                ),
            },
        }
        torch.save(final_checkpoint, self.experiment.best_checkpoint_path)
        torch.save(final_checkpoint, self.experiment.last_checkpoint_path)
        artifacts = save_evaluation_artifacts(
            selected["test_metrics"],
            selected["details"],
            output_dir=self.experiment.run_dir / "evaluation",
        )
        summary = {
            "selected_method": selected["name"],
            "selected_checkpoint": selected["checkpoint"],
            "validation_metrics": selected["validation_metrics"],
            "test_metrics": selected["test_metrics"],
            "hubris": selected["hubris"],
            "evaluation_artifacts": artifacts,
            "methods": [
                {
                    "name": candidate["name"],
                    "checkpoint": candidate["checkpoint"],
                    "validation_metrics": candidate["validation_metrics"],
                    "test_metrics": candidate["test_metrics"],
                    "hubris": candidate["hubris"],
                    "boundary_count": candidate["boundary_count"],
                }
                for candidate in candidates
            ],
        }
        self.experiment.save_metrics(summary)
        (self.experiment.run_dir / "hubris_comparison.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(f"\nSelected method: {selected['name']}")
        print(f"Hubris: {selected['hubris']:.4f}")
        print(format_confusion_matrix(selected["test_metrics"]))
        print()
        print(
            format_classification_report(
                selected["details"]["classification_report"]
            )
        )
        return summary
