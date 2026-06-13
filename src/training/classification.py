from __future__ import annotations

import torch
import torch.nn as nn

from src.classifier.dataset import unpack_batch
from src.classifier.evaluate import compute_metrics
from src.training.base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def unpack_batch(self, batch):
        return unpack_batch(batch)

    def create_metric_state(self):
        return {"targets": [], "predictions": []}

    def update_metric_state(self, state, outputs, targets):
        state["targets"].append(targets.detach().cpu())
        state["predictions"].append(outputs.argmax(dim=1).detach().cpu())

    def finalize_metrics(self, state):
        targets = (
            torch.cat(state["targets"])
            if state["targets"]
            else torch.empty(0, dtype=torch.long)
        )
        predictions = (
            torch.cat(state["predictions"])
            if state["predictions"]
            else torch.empty(0, dtype=torch.long)
        )
        metrics = compute_metrics(targets, predictions)
        metrics.pop("loss", None)
        return metrics

    def on_train_mode(self):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                parameters = list(module.parameters())
                if parameters and not any(
                    parameter.requires_grad for parameter in parameters
                ):
                    module.eval()


class SoftLabelClassificationTrainer(ClassificationTrainer):
    def update_metric_state(self, state, outputs, targets):
        hard_targets = (
            targets.argmax(dim=1)
            if targets.ndim == 2
            else targets
        )
        state["targets"].append(hard_targets.detach().cpu())
        state["predictions"].append(outputs.argmax(dim=1).detach().cpu())
