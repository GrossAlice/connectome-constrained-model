"""Tiny shared helpers used by multiple stage2 modules."""
from __future__ import annotations
import numpy as np


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    # Use 1e-12 floor (not NaN) for near-constant neurons, matching
    # the ridge-MLP baseline convention.
    return float(1.0 - np.sum((yt - yp) ** 2) / max(ss_tot, 1e-12))


def _cfg_val(cfg, attr: str, default, cast=float):
    if cfg is None:
        return cast(default)
    value = getattr(cfg, attr, default)
    if value is None:
        value = default
    return cast(value)
