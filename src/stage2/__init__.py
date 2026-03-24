import logging
import sys
from typing import Any


class _CurrentStdout:
    def write(self, msg: str) -> int:
        return sys.stdout.write(msg)

    def flush(self) -> None:
        sys.stdout.flush()


class Stage2Logger:
    def __init__(self, name: str = "stage2") -> None:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(_CurrentStdout())
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        self._logger = logger

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        payload = " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))
        self._logger.log(level, f"[Stage2][{event}]" + (f" {payload}" if payload else ""))

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def metrics(self, event: str, metrics: dict[str, Any], **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields, **metrics)


def get_stage2_logger(name: str = "stage2") -> Stage2Logger:
    return Stage2Logger(name=name)

from .config import (
    DataConfig,
    DynamicsConfig,
    StimulusConfig,
    BehaviorConfig,
    TrainConfig,
    EvalConfig,
    OutputConfig,
    Stage2PTConfig,
    make_config,
)
from .model import Stage2ModelPT
from .train import train_stage2

__all__ = [
    "Stage2Logger",
    "get_stage2_logger",
    "DataConfig",
    "DynamicsConfig",
    "StimulusConfig",
    "BehaviorConfig",
    "TrainConfig",
    "EvalConfig",
    "OutputConfig",
    "Stage2PTConfig",
    "make_config",
    "Stage2ModelPT",
    "train_stage2",
]
