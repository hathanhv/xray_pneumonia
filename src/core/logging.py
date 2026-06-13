from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping


def configure_logger(
    name: str,
    log_path: str | Path,
    *,
    level: int = logging.INFO,
) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


class CSVMetricLogger:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = self._read_existing_header()

    def _read_existing_header(self) -> list[str] | None:
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return None
        with self.output_path.open("r", encoding="utf-8", newline="") as file:
            return next(csv.reader(file), None)

    def log(self, metrics: Mapping[str, Any]) -> None:
        row = dict(metrics)
        row_fields = list(row)
        if self.fieldnames is None:
            self.fieldnames = row_fields
        elif row_fields != self.fieldnames:
            raise ValueError(
                "Metric columns changed. "
                f"Expected {self.fieldnames}, received {row_fields}."
            )

        write_header = not self.output_path.exists()
        with self.output_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
