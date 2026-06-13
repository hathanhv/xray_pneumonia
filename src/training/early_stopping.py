from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class MonitorConstraint:
    metric: str
    operator: str
    value: float

    def satisfied(self, metrics: Mapping[str, float]) -> bool:
        current = float(metrics[self.metric])
        if self.operator in {">=", "ge"}:
            return current >= self.value
        if self.operator in {"<=", "le"}:
            return current <= self.value
        if self.operator in {">", "gt"}:
            return current > self.value
        if self.operator in {"<", "lt"}:
            return current < self.value
        raise ValueError(f"Unsupported constraint operator: {self.operator}")


@dataclass
class EarlyStopping:
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 5
    min_delta: float = 0.0
    constraints: list[MonitorConstraint] = field(default_factory=list)
    best_score: float | None = None
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, epoch: int, metrics: Mapping[str, float]) -> tuple[bool, bool]:
        constraints_satisfied = all(
            constraint.satisfied(metrics) for constraint in self.constraints
        )
        score = float(metrics[self.monitor])
        if self.best_score is None:
            improved = constraints_satisfied
        elif self.mode == "min":
            improved = constraints_satisfied and score < self.best_score - self.min_delta
        elif self.mode == "max":
            improved = constraints_satisfied and score > self.best_score + self.min_delta
        else:
            raise ValueError("Early stopping mode must be 'min' or 'max'")

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, should_stop

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state):
        self.best_score = state.get("best_score")
        self.best_epoch = int(state.get("best_epoch", 0))
        self.bad_epochs = int(state.get("bad_epochs", 0))


@dataclass
class RecallConstrainedEarlyStopping(EarlyStopping):
    fallback_monitor: str = "val_specificity"
    fallback_mode: str = "max"
    fallback_score: float | None = None
    has_eligible_model: bool = False

    @staticmethod
    def _improves(current, best, mode, min_delta):
        if best is None:
            return True
        if mode == "min":
            return current < best - min_delta
        if mode == "max":
            return current > best + min_delta
        raise ValueError("Monitor mode must be 'min' or 'max'")

    def update(self, epoch: int, metrics: Mapping[str, float]) -> tuple[bool, bool]:
        eligible = all(
            constraint.satisfied(metrics) for constraint in self.constraints
        )
        improved = False
        if eligible:
            score = float(metrics[self.monitor])
            improved = (
                not self.has_eligible_model
                or self._improves(score, self.best_score, self.mode, self.min_delta)
            )
            if improved:
                self.best_score = score
                self.best_epoch = epoch
                self.has_eligible_model = True
        elif not self.has_eligible_model:
            fallback_score = float(metrics[self.fallback_monitor])
            improved = self._improves(
                fallback_score,
                self.fallback_score,
                self.fallback_mode,
                self.min_delta,
            )
            if improved:
                self.fallback_score = fallback_score
                self.best_epoch = epoch

        self.bad_epochs = 0 if improved else self.bad_epochs + 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, should_stop

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "fallback_score": self.fallback_score,
                "has_eligible_model": self.has_eligible_model,
            }
        )
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.fallback_score = state.get("fallback_score")
        self.has_eligible_model = bool(state.get("has_eligible_model", False))


def build_early_stopping(config):
    constraints = [
        MonitorConstraint(
            metric=item["metric"],
            operator=item.get("operator", ">="),
            value=float(item["value"]),
        )
        for item in config.get("constraints", [])
    ]
    monitor = config.get("monitor", "val_loss")
    mode = config.get("mode")
    if mode is None:
        mode = "min" if monitor.endswith("loss") else "max"
    common = dict(
        monitor=monitor,
        mode=mode,
        patience=int(config.get("patience", 5)),
        min_delta=float(config.get("min_delta", 0.0)),
        constraints=constraints,
    )
    selection = str(config.get("selection", "standard")).lower()
    if selection in {"recall_constrained", "constrained_with_fallback"}:
        return RecallConstrainedEarlyStopping(
            **common,
            fallback_monitor=str(
                config.get("fallback_monitor", "val_specificity")
            ),
            fallback_mode=str(config.get("fallback_mode", "max")),
        )
    return EarlyStopping(**common)
