from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .config import save_config
from .logging import CSVMetricLogger, close_logger, configure_logger
from .paths import PROJECT_ROOT, resolve_path


def build_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_%f")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


@dataclass(frozen=True)
class ExperimentContext:
    name: str
    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    figure_dir: Path
    error_analysis_dir: Path
    config_path: Path
    metrics_path: Path
    training_log_path: Path
    app_log_path: Path
    logger: Any
    metric_logger: CSVMetricLogger

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_model.pth"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last_model.pth"

    def save_metrics(self, metrics: Mapping[str, Any]) -> Path:
        with self.metrics_path.open("w", encoding="utf-8") as file:
            json.dump(
                dict(metrics),
                file,
                indent=2,
                ensure_ascii=True,
                default=_json_default,
            )
        return self.metrics_path

    def close(self) -> None:
        close_logger(self.logger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


def create_experiment(
    config: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_id: str | None = None,
) -> ExperimentContext:
    project_root = Path(project_root).resolve(strict=False)
    experiment_config = config.get("experiment", {})
    if not isinstance(experiment_config, Mapping):
        raise ValueError("'experiment' config must be a mapping.")

    name = str(experiment_config.get("name", "")).strip()
    if not name:
        raise ValueError("Missing required experiment.name.")

    output_root_value = experiment_config.get(
        "output_root",
        "outputs/experiments",
    )
    output_root = resolve_path(output_root_value, project_root)
    run_id = run_id or build_run_id()
    run_dir = output_root / name / run_id
    if run_dir.exists():
        raise FileExistsError(f"Experiment run already exists: {run_dir}")

    checkpoint_dir = run_dir / "checkpoints"
    figure_dir = run_dir / "figures"
    error_analysis_dir = run_dir / "error_analysis"
    for directory in (
        checkpoint_dir,
        figure_dir,
        error_analysis_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    config_path = save_config(config, run_dir / "config_used.yaml")
    metrics_path = run_dir / "metrics.json"
    training_log_path = run_dir / "training_log.csv"
    app_log_path = run_dir / "run.log"
    logger = configure_logger(f"experiment.{name}.{run_id}", app_log_path)
    metric_logger = CSVMetricLogger(training_log_path)

    return ExperimentContext(
        name=name,
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        figure_dir=figure_dir,
        error_analysis_dir=error_analysis_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        training_log_path=training_log_path,
        app_log_path=app_log_path,
        logger=logger,
        metric_logger=metric_logger,
    )
