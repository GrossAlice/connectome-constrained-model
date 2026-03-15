import logging
import sys
from typing import Any

# --------------------------------------------------------------------------- #
#  Structured logger (merged from logging_utils.py)                            #
# --------------------------------------------------------------------------- #

class Stage2Logger:
    def __init__(self, name: str = "stage2") -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        payload = " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))
        message = f"[Stage2][{event}]"
        if payload:
            message = f"{message} {payload}"
        self._logger.log(level, message)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def metrics(self, event: str, metrics: dict[str, Any], **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields, **metrics)


def get_stage2_logger(name: str = "stage2") -> Stage2Logger:
    return Stage2Logger(name=name)


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

from .config import (  # noqa: F401
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
from .model import Stage2ModelPT  # noqa: F401
from .train import train_stage2  # noqa: F401
