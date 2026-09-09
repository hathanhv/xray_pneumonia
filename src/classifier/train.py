import time

import torch
import torch.nn as nn

from src.classifier.evaluate import evaluate_model
from src.classifier.dataset import unpack_batch
from src.classifier.model import save_checkpoint


def set_frozen_batchnorm_eval(model):
    """Prevent frozen BatchNorm layers from updating running statistics."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters())
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    set_frozen_batchnorm_eval(model)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images, labels, _ = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
    }


def train_classifier(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path,
    class_to_idx,
    scheduler=None,
    monitor="val_loss",
    early_stopping_patience=5,
    val_loader=None,
    test_loader=None,
    checkpoint_metadata=None,
):
    if val_loader is None:
        raise ValueError(
            "val_loader is required. The test split must not be used for model selection."
        )
    if monitor not in {"val_loss", "val_accuracy", "val_f1"}:
        raise ValueError("monitor must be one of: val_loss, val_accuracy, val_f1")

    best_score = float("inf") if monitor == "val_loss" else -1.0
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={time.time() - start:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_specificity": val_metrics["specificity"],
                "val_f1": val_metrics["f1"],
                "val_tp": val_metrics["tp"],
                "val_tn": val_metrics["tn"],
                "val_fp": val_metrics["fp"],
                "val_fn": val_metrics["fn"],
            }
        )

        if scheduler is not None:
            if monitor == "val_loss":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step(val_metrics[monitor.replace("val_", "")])

        if monitor == "val_loss":
            current_score = val_metrics["loss"]
            improved = current_score < best_score
        elif monitor == "val_accuracy":
            current_score = val_metrics["accuracy"]
            improved = current_score > best_score
        else:
            current_score = val_metrics["f1"]
            improved = current_score > best_score

        if improved:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_metrics = {
                "train": train_metrics,
                "val": val_metrics,
                "best_epoch": epoch,
                "monitor": monitor,
                "best_score": best_score,
            }
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=best_metrics,
                class_to_idx=class_to_idx,
                metadata=checkpoint_metadata,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/"
                f"{early_stopping_patience} epoch(s). Best epoch: {best_epoch}"
            )

        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_metrics is not None and test_loader is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        best_metrics["test"] = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )
    return best_metrics, history
