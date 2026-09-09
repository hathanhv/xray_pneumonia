"""Reusable training loops, callbacks, losses, and optimization factories."""

from .base import BaseTrainer
from .checkpoints import CheckpointManager
from .classification import ClassificationTrainer, SoftLabelClassificationTrainer
from .early_stopping import (
    EarlyStopping,
    MonitorConstraint,
    RecallConstrainedEarlyStopping,
    build_early_stopping,
)
from .optimizers import build_optimizer, build_parameter_groups
from .schedulers import build_scheduler

__all__ = [
    "BaseTrainer",
    "CheckpointManager",
    "ClassificationTrainer",
    "SoftLabelClassificationTrainer",
    "EarlyStopping",
    "MonitorConstraint",
    "RecallConstrainedEarlyStopping",
    "build_early_stopping",
    "build_optimizer",
    "build_parameter_groups",
    "build_scheduler",
]
