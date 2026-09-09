from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch

from src.training.schedulers import step_scheduler


class BaseTrainer(ABC):
    def __init__(
        self,
        *,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        early_stopping=None,
        checkpoint_manager=None,
        metric_logger=None,
        logger=None,
        checkpoint_metadata=None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager
        self.metric_logger = metric_logger
        self.logger = logger
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.history = []
        self.start_epoch = 1

    def move_to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: self.move_to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self.move_to_device(item) for item in value]
        return value

    @abstractmethod
    def unpack_batch(self, batch):
        raise NotImplementedError

    @abstractmethod
    def create_metric_state(self):
        raise NotImplementedError

    @abstractmethod
    def update_metric_state(self, state, outputs, targets):
        raise NotImplementedError

    @abstractmethod
    def finalize_metrics(self, state):
        raise NotImplementedError

    def on_train_mode(self):
        self.model.train()

    def run_epoch(self, dataloader, *, training):
        self.on_train_mode() if training else self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metric_state = self.create_metric_state()
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in dataloader:
                inputs, targets, _metadata = self.unpack_batch(batch)
                inputs = self.move_to_device(inputs)
                targets = self.move_to_device(targets)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                if training:
                    loss.backward()
                    self.optimizer.step()
                batch_size = int(targets.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                self.update_metric_state(metric_state, outputs, targets)
        metrics = self.finalize_metrics(metric_state)
        metrics["loss"] = total_loss / total_samples if total_samples else 0.0
        return metrics

    def fit(self, train_loader, val_loader, *, epochs, resume_from=None):
        if resume_from:
            checkpoint = self.checkpoint_manager.resume(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                device=self.device,
            )
            self.start_epoch = int(checkpoint["epoch"]) + 1
            self.history = list(checkpoint.get("history", []))

        for epoch in range(self.start_epoch, int(epochs) + 1):
            started = time.time()
            train_metrics = self.run_epoch(train_loader, training=True)
            val_metrics = self.run_epoch(val_loader, training=False)
            row = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            row["epoch_seconds"] = time.time() - started
            self.history.append(row)

            monitor_metrics = {f"val_{key}": value for key, value in val_metrics.items()}
            improved, should_stop = self.early_stopping.update(
                epoch,
                monitor_metrics,
            )
            step_scheduler(
                self.scheduler,
                monitor_metrics,
                self.early_stopping.monitor,
            )
            if self.metric_logger:
                self.metric_logger.log(row)
            message = (
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics.get('accuracy', 0.0):.4f} | "
                f"val_f1={val_metrics.get('f1', 0.0):.4f}"
            )
            print(message)
            if self.logger:
                self.logger.info(message)

            state = self.checkpoint_manager.build_state(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                metrics={"train": train_metrics, "val": val_metrics},
                history=self.history,
                metadata=self.checkpoint_metadata,
            )
            self.checkpoint_manager.save(state, is_best=improved)
            if should_stop:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch={self.early_stopping.best_epoch}"
                )
                break
        return self.history
