from __future__ import annotations

import time

import torch

from src.lung_segmentation.evaluate import binary_segmentation_scores
from src.training.schedulers import step_scheduler


class SegmentationTrainer:
    def __init__(
        self,
        *,
        model,
        criterion,
        optimizer,
        device,
        scheduler,
        early_stopping,
        checkpoint_manager,
        metric_logger=None,
        logger=None,
        checkpoint_metadata=None,
        threshold=0.5,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager
        self.metric_logger = metric_logger
        self.logger = logger
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.threshold = float(threshold)
        self.history = []
        self.start_epoch = 1

    def run_epoch(self, dataloader, *, training):
        self.model.train(training)
        total_loss = 0.0
        total_samples = 0
        dice_values = []
        iou_values = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in dataloader:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                if training:
                    loss.backward()
                    self.optimizer.step()
                predictions = (torch.sigmoid(logits) >= self.threshold).float()
                dice, iou = binary_segmentation_scores(predictions, masks)
                batch_size = masks.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                dice_values.extend(dice.detach().cpu().tolist())
                iou_values.extend(iou.detach().cpu().tolist())
        return {
            "loss": total_loss / max(total_samples, 1),
            "dice": sum(dice_values) / max(len(dice_values), 1),
            "iou": sum(iou_values) / max(len(iou_values), 1),
        }

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
            improved, should_stop = self.early_stopping.update(epoch, monitor_metrics)
            step_scheduler(self.scheduler, monitor_metrics, self.early_stopping.monitor)
            if self.metric_logger:
                self.metric_logger.log(row)
            message = (
                f"Epoch {epoch:03d}/{epochs} | train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | val_dice={val_metrics['dice']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f}"
            )
            print(message)
            if self.logger:
                self.logger.info(message)
            metadata = {
                **self.checkpoint_metadata,
                "encoder": self.checkpoint_metadata.get("encoder", "resnet34"),
                "img_size": int(self.checkpoint_metadata.get("img_size", 256)),
                "best_val_loss": self.early_stopping.best_score,
            }
            state = self.checkpoint_manager.build_state(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                metrics={"train": train_metrics, "val": val_metrics},
                history=self.history,
                metadata=metadata,
            )
            state.update(
                {
                    "encoder": metadata["encoder"],
                    "img_size": metadata["img_size"],
                    "best_val_loss": metadata["best_val_loss"],
                }
            )
            self.checkpoint_manager.save(state, is_best=improved)
            if should_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        return self.history
