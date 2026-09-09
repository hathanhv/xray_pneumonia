from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.classifier.dataset import (
    CLASS_TO_IDX,
    classification_collate,
    create_datasets_from_config,
)
from src.classifier.ensemble import (
    blend_probabilities,
    search_ensemble_weight,
    select_recall_constrained_method,
)
from src.classifier.evaluate import (
    build_classification_report,
    format_classification_report,
    format_confusion_matrix,
    save_evaluation_artifacts,
)
from src.classifier.hard_negative import (
    hard_negative_sampler_config,
    mine_hard_negatives,
)
from src.classifier.losses import build_loss, resolve_class_weights
from src.classifier.model import (
    build_mobilenet_v2_from_config,
    load_classifier_checkpoint,
)
from src.classifier.prediction import collect_logits, positive_probabilities
from src.classifier.samplers import build_sampler
from src.classifier.thresholding import metrics_at_threshold, tune_threshold
from src.classifier.tta import build_tta_views, predict_tta
from src.classifier.calibration import TemperatureScaler
from src.core.logging import CSVMetricLogger
from src.core.reproducibility import create_torch_generator, seed_worker
from src.training import (
    CheckpointManager,
    ClassificationTrainer,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
)


class _CombinedMetricLogger:
    def __init__(self, *loggers):
        self.loggers = loggers

    def log(self, metrics):
        for logger in self.loggers:
            logger.log(metrics)


class AdvancedClassificationPipeline:
    """YAML-driven advanced classification strategies."""

    def __init__(self, config, experiment, device):
        self.config = deepcopy(config)
        self.experiment = experiment
        self.device = torch.device(device)
        self.datasets = create_datasets_from_config(config)
        self.loader_config = config.get("dataloader", {})
        self.seed = int(config.get("seed", 42))
        self.models = {}
        self.stage_artifacts = {}

    def _loader(self, split, sampler_config=None):
        dataset = self.datasets[split]
        generator = create_torch_generator(self.seed)
        sampler = None
        if split == "train":
            sampler = build_sampler(
                dataset.targets,
                sampler_config or self.config.get("sampler", {"name": "random"}),
                generator=generator,
            )
        return DataLoader(
            dataset,
            batch_size=int(self.loader_config.get("batch_size", 32)),
            shuffle=False,
            sampler=sampler,
            num_workers=int(self.loader_config.get("num_workers", 0)),
            pin_memory=bool(self.loader_config.get("pin_memory", True)),
            worker_init_fn=seed_worker,
            generator=generator,
            collate_fn=classification_collate,
        )

    def _build_model(self, model_config=None, checkpoint_path=None):
        model_config = deepcopy(model_config or self.config["model"])
        if checkpoint_path:
            model_config["pretrained"] = False
        model, metadata = build_mobilenet_v2_from_config(model_config)
        if checkpoint_path:
            load_classifier_checkpoint(
                model,
                checkpoint_path,
                device="cpu",
                strict=True,
            )
            metadata["initialization_checkpoint"] = str(checkpoint_path)
        return model, metadata

    def _train_stage(
        self,
        name,
        model,
        stage_config,
        *,
        sampler_config=None,
        model_metadata=None,
    ):
        stage_dir = self.experiment.run_dir / "stages" / name
        manager = CheckpointManager(
            stage_dir / "best_model.pth",
            stage_dir / "last_model.pth",
        )
        weights = resolve_class_weights(
            stage_config.get("class_weighting"),
            labels=self.datasets["train"].targets,
            num_classes=len(CLASS_TO_IDX),
        )
        if weights is not None:
            weights = weights.to(self.device)
        criterion = build_loss(
            stage_config["loss"],
            class_weights=weights,
        ).to(self.device)
        optimizer = build_optimizer(model, stage_config["optimizer"])
        scheduler = build_scheduler(
            optimizer,
            stage_config.get("scheduler"),
        )
        early_stopping = build_early_stopping(
            stage_config["early_stopping"]
        )
        trainer = ClassificationTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            checkpoint_manager=manager,
            device=self.device,
            metric_logger=_CombinedMetricLogger(
                self.experiment.metric_logger,
                CSVMetricLogger(stage_dir / "training_log.csv"),
            ),
            logger=self.experiment.logger,
            checkpoint_metadata={
                **(model_metadata or {}),
                "class_to_idx": CLASS_TO_IDX,
                "stage": name,
            },
        )
        history = trainer.fit(
            self._loader("train", sampler_config),
            self._loader("val"),
            epochs=int(stage_config["epochs"]),
        )
        manager.resume(manager.best_path, model=model, device=self.device)
        self.stage_artifacts[name] = {
            "best_checkpoint": str(manager.best_path),
            "last_checkpoint": str(manager.last_path),
            "best_epoch": early_stopping.best_epoch,
            "history": history,
        }
        self.models[name] = model
        return model

    def _probabilities(self, model, split):
        logits, targets, metadata = collect_logits(
            model,
            self._loader(split),
            self.device,
        )
        return positive_probabilities(logits), targets, metadata, logits

    def _method_from_standard(self, name, model):
        val_probs, val_targets, _, val_logits = self._probabilities(model, "val")
        test_probs, test_targets, test_metadata, test_logits = self._probabilities(
            model,
            "test",
        )
        return {
            "name": name,
            "val_probabilities": val_probs,
            "test_probabilities": test_probs,
            "val_targets": val_targets,
            "test_targets": test_targets,
            "test_metadata": test_metadata,
            "val_logits": val_logits,
            "test_logits": test_logits,
        }

    def _add_threshold(self, method, threshold_config):
        search = tune_threshold(
            method["val_targets"],
            method["val_probabilities"],
            threshold_config,
        )
        method = dict(method)
        method["threshold"] = search.threshold
        method["metrics"] = search.metrics
        method["constraint_satisfied"] = search.constraint_satisfied
        method["fallback_used"] = search.fallback_used
        method["test_metrics"] = metrics_at_threshold(
            method["test_targets"],
            method["test_probabilities"],
            search.threshold,
        )
        return method

    def _tta_method(self, name, model, tta_config):
        views = build_tta_views(
            tta_config,
            image_size=int(tta_config.get("image_size", 224)),
        )
        val_all, val_targets, _ = predict_tta(
            model,
            self.datasets["val"],
            device=self.device,
            batch_size=self.loader_config.get("batch_size", 32),
            num_workers=self.loader_config.get("num_workers", 0),
            views=views,
        )
        test_all, test_targets, test_metadata = predict_tta(
            model,
            self.datasets["test"],
            device=self.device,
            batch_size=self.loader_config.get("batch_size", 32),
            num_workers=self.loader_config.get("num_workers", 0),
            views=views,
        )
        return {
            "name": name,
            "val_probabilities": val_all[:, 1],
            "test_probabilities": test_all[:, 1],
            "val_targets": val_targets,
            "test_targets": test_targets,
            "test_metadata": test_metadata,
        }

    def _temperature_method(self, name, base_method, calibration_config):
        scaler = TemperatureScaler(
            calibration_config.get("initial_temperature", 1.5)
        ).to(self.device)
        temperature = scaler.fit(
            base_method["val_logits"].to(self.device),
            base_method["val_targets"].to(self.device),
            lr=calibration_config.get("lr", 0.01),
            max_iter=calibration_config.get("max_iter", 100),
        )
        with torch.no_grad():
            val_probs = torch.softmax(
                scaler(base_method["val_logits"].to(self.device)),
                dim=1,
            )[:, 1].cpu()
            test_probs = torch.softmax(
                scaler(base_method["test_logits"].to(self.device)),
                dim=1,
            )[:, 1].cpu()
        result = dict(base_method)
        result.update(
            {
                "name": name,
                "val_probabilities": val_probs,
                "test_probabilities": test_probs,
                "temperature": temperature,
            }
        )
        return result

    def _run_class_weighting_threshold(self, advanced):
        model, metadata = self._build_model()
        model = self._train_stage(
            "class_weighting",
            model,
            advanced["training"],
            model_metadata=metadata,
        )
        return [
            self._add_threshold(
                self._method_from_standard("weighted_threshold", model),
                advanced["threshold_tuning"],
            )
        ]

    def _run_hard_negative_mining(self, advanced):
        model, metadata = self._build_model()
        model = self._train_stage(
            "focal_training",
            model,
            advanced["stage1"],
            model_metadata=metadata,
        )
        mining_config = advanced["hard_negative_mining"]
        mined = mine_hard_negatives(
            model,
            self.datasets["train"],
            device=self.device,
            threshold=mining_config.get("threshold", 0.3),
            batch_size=self.loader_config.get("batch_size", 32),
            num_workers=self.loader_config.get("num_workers", 0),
        )
        sampler_config = hard_negative_sampler_config(
            mined,
            dataset_size=len(self.datasets["train"]),
            oversample_factor=mining_config.get("oversample_factor", 3),
        )
        model = self._train_stage(
            "hard_negative_finetuning",
            model,
            advanced["stage2"],
            sampler_config=sampler_config,
            model_metadata=metadata,
        )
        self.stage_artifacts["hard_negative_mining"] = {
            "count": len(mined.indices),
            "indices": list(mined.indices),
            "probabilities": list(mined.probabilities),
        }
        base = self._method_from_standard("hard_negative", model)
        methods = [
            self._add_threshold(base, advanced["threshold_tuning"])
        ]
        if advanced.get("tta", {}).get("enabled", False):
            methods.append(
                self._add_threshold(
                    self._tta_method("hard_negative_tta", model, advanced["tta"]),
                    advanced["threshold_tuning"],
                )
            )
        if advanced.get("temperature_scaling", {}).get("enabled", False):
            methods.append(
                self._add_threshold(
                    self._temperature_method(
                        "hard_negative_temperature",
                        base,
                        advanced["temperature_scaling"],
                    ),
                    advanced["threshold_tuning"],
                )
            )
        return methods

    def _run_head_finetuning_ensemble(self, advanced):
        stage1, _ = self._build_model(
            checkpoint_path=advanced["checkpoints"]["stage1_path"]
        )
        stage2, _ = self._build_model(
            checkpoint_path=advanced["checkpoints"]["stage2_path"]
        )
        base1 = self._method_from_standard("stage1", stage1)
        base2 = self._method_from_standard("stage2", stage2)
        methods = [
            self._add_threshold(base2, advanced["threshold_tuning"])
        ]

        if advanced.get("tta", {}).get("enabled", True):
            methods.append(
                self._add_threshold(
                    self._tta_method("tta", stage2, advanced["tta"]),
                    advanced["threshold_tuning"],
                )
            )
        if advanced.get("temperature_scaling", {}).get("enabled", True):
            methods.append(
                self._add_threshold(
                    self._temperature_method(
                        "temperature_scaling",
                        base2,
                        advanced["temperature_scaling"],
                    ),
                    advanced["threshold_tuning"],
                )
            )

        head_config = advanced.get("head_finetuning", {})
        if head_config.get("enabled", True):
            for alpha_normal in head_config.get(
                "normal_alpha_candidates",
                [3.0, 5.0, 8.0],
            ):
                model_config = deepcopy(self.config["model"])
                model_config["finetune_mode"] = "head"
                head_model, metadata = self._build_model(
                    model_config,
                    advanced["checkpoints"]["stage2_path"],
                )
                stage_config = deepcopy(head_config["training"])
                stage_config["class_weighting"] = {
                    "strategy": "manual",
                    "values": [
                        float(alpha_normal),
                        float(head_config.get("pneumonia_alpha", 0.5)),
                    ],
                }
                trained = self._train_stage(
                    f"head_alpha_{alpha_normal:g}",
                    head_model,
                    stage_config,
                    model_metadata=metadata,
                )
                methods.append(
                    self._add_threshold(
                        self._method_from_standard(
                            f"head_alpha_{alpha_normal:g}",
                            trained,
                        ),
                        advanced["threshold_tuning"],
                    )
                )

        ensemble_config = advanced.get("ensemble", {})
        if ensemble_config.get("enabled", True):
            ensemble = search_ensemble_weight(
                base1["val_targets"],
                base1["val_probabilities"],
                base2["val_probabilities"],
                weights=ensemble_config.get("weights", [0.2, 0.3, 0.4, 0.5]),
                min_recall=ensemble_config.get("min_recall", 0.99),
            )
            ensemble_method = {
                "name": "ensemble",
                "val_probabilities": ensemble.probabilities,
                "test_probabilities": blend_probabilities(
                    base1["test_probabilities"],
                    base2["test_probabilities"],
                    ensemble.weight_first,
                ),
                "val_targets": base1["val_targets"],
                "test_targets": base1["test_targets"],
                "test_metadata": base1["test_metadata"],
                "ensemble_weight_stage1": ensemble.weight_first,
            }
            methods.append(
                self._add_threshold(
                    ensemble_method,
                    advanced["threshold_tuning"],
                )
            )
        return methods

    def _save_final(self, selected, methods):
        metrics = selected["test_metrics"]
        report = build_classification_report(metrics)
        predictions = []
        threshold = selected["threshold"]
        predicted = (selected["test_probabilities"] >= threshold).long()
        for index, target in enumerate(selected["test_targets"]):
            metadata = selected["test_metadata"][index]
            probability = float(selected["test_probabilities"][index])
            predictions.append(
                {
                    "sample_id": metadata.get("sample_id", str(index)),
                    "image_path": metadata.get("image_path", ""),
                    "target": int(target),
                    "prediction": int(predicted[index]),
                    "p_normal": 1.0 - probability,
                    "p_pneumonia": probability,
                }
            )
        artifacts = save_evaluation_artifacts(
            metrics,
            {
                "classification_report": report,
                "predictions": predictions,
            },
            output_dir=self.experiment.run_dir / "evaluation",
        )
        method_summary = [
            {
                key: value
                for key, value in method.items()
                if key
                in {
                    "name",
                    "threshold",
                    "metrics",
                    "test_metrics",
                    "constraint_satisfied",
                    "fallback_used",
                    "temperature",
                    "ensemble_weight_stage1",
                }
            }
            for method in methods
        ]
        result = {
            "selected_method": selected["name"],
            "selected_threshold": threshold,
            "validation_metrics": selected["metrics"],
            "test_metrics": metrics,
            "classification_report": report,
            "methods": method_summary,
            "stages": self.stage_artifacts,
            "evaluation_artifacts": artifacts,
        }
        self.experiment.save_metrics(result)
        (self.experiment.run_dir / "advanced_summary.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        checkpoint_source = None
        selected_name = selected["name"]
        if selected_name.startswith("head_alpha"):
            stage_name = selected_name
            artifact = self.stage_artifacts.get(stage_name, {})
            if artifact.get("best_checkpoint"):
                checkpoint_source = Path(artifact["best_checkpoint"])
        elif (
            str(self.config["advanced"]["protocol"]).lower()
            == "head_finetuning_ensemble"
        ):
            checkpoint_source = Path(
                self.config["advanced"]["checkpoints"]["stage2_path"]
            )
        else:
            for artifact in reversed(list(self.stage_artifacts.values())):
                if isinstance(artifact, dict) and artifact.get("best_checkpoint"):
                    checkpoint_source = Path(artifact["best_checkpoint"])
                    break
        if checkpoint_source and checkpoint_source.exists():
            try:
                checkpoint = torch.load(
                    checkpoint_source,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_source, map_location="cpu")
            checkpoint.setdefault("metadata", {}).update(
                {
                    "advanced_method": selected_name,
                    "decision_threshold": threshold,
                    "validation_metrics": selected["metrics"],
                    "test_metrics": selected["test_metrics"],
                }
            )
            if "temperature" in selected:
                checkpoint["metadata"]["temperature"] = selected["temperature"]
            if "ensemble_weight_stage1" in selected:
                checkpoint["metadata"]["ensemble_weight_stage1"] = selected[
                    "ensemble_weight_stage1"
                ]
            torch.save(checkpoint, self.experiment.best_checkpoint_path)
            shutil.copy2(
                self.experiment.best_checkpoint_path,
                self.experiment.last_checkpoint_path,
            )
        print()
        print(f"Selected method: {selected['name']}")
        print(f"Threshold: {threshold:.4f}")
        print(format_confusion_matrix(metrics))
        print()
        print(format_classification_report(report))
        return result

    def run(self):
        advanced = self.config["advanced"]
        protocol = str(advanced["protocol"]).lower()
        if protocol == "class_weighting_threshold":
            methods = self._run_class_weighting_threshold(advanced)
        elif protocol == "hard_negative_mining":
            methods = self._run_hard_negative_mining(advanced)
        elif protocol == "head_finetuning_ensemble":
            methods = self._run_head_finetuning_ensemble(advanced)
        else:
            raise ValueError(f"Unknown advanced protocol: {protocol}")

        selection = advanced.get("model_selection", {})
        selected = select_recall_constrained_method(
            methods,
            min_recall=selection.get("min_recall", 0.95),
            min_specificity=selection.get("min_specificity", 0.0),
            min_accuracy=selection.get("min_accuracy", 0.0),
        )
        return self._save_final(selected, methods)
