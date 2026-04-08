"""Shared helpers used by multiple stage2 modules.

All pure-numpy / pure-torch metric and utility functions live here so that
``evaluate.py``, ``evaluate_multi.py``, ``plot_eval.py``, and external
consumers (``atlas_transformer``, ``scripts/``) can import from one place.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ── scalar metrics ──────────────────────────────────────────────────────────

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    # Use 1e-12 floor (not NaN) for near-constant neurons, matching
    # the ridge-MLP baseline convention.
    return float(1.0 - np.sum((yt - yp) ** 2) / max(ss_tot, 1e-12))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xv, yv = x[m], y[m]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m].astype(np.float64) - y_pred[m].astype(np.float64)) ** 2)))


# ── per-neuron metrics (vectorised) ────────────────────────────────────────

def _per_neuron_metrics(
    u_true: np.ndarray, u_pred: np.ndarray, indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = u_true.shape[1]
    r2, corr, rmse = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    if not indices:
        return r2, corr, rmse
    idx = np.asarray(indices)
    yt = u_true[:, idx].astype(np.float64)
    yp = u_pred[:, idx].astype(np.float64)
    fin = np.isfinite(yt) & np.isfinite(yp)
    if fin.all():
        # ---- vectorised fast path (no NaN in data) ----
        n = yt.shape[0]
        yt_m = yt.mean(0)
        ss_tot = ((yt - yt_m) ** 2).sum(0)
        ss_res = ((yt - yp) ** 2).sum(0)
        r2[idx] = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
        yp_m = yp.mean(0)
        yt_c, yp_c = yt - yt_m, yp - yp_m
        s_yt = np.sqrt((yt_c ** 2).sum(0))
        s_yp = np.sqrt((yp_c ** 2).sum(0))
        ok = (s_yt > 1e-12) & (s_yp > 1e-12) & (n >= 3)
        corr[idx] = np.where(ok, (yt_c * yp_c).sum(0) / (s_yt * s_yp), np.nan)
        rmse[idx] = np.sqrt(ss_res / n)
    else:
        # ---- fallback: per-neuron with NaN handling ----
        for i in indices:
            r2[i] = _r2(u_true[:, i], u_pred[:, i])
            corr[i] = _pearson(u_true[:, i], u_pred[:, i])
            rmse[i] = _rmse(u_true[:, i], u_pred[:, i])
    return r2, corr, rmse


# ── config helper ───────────────────────────────────────────────────────────

def _cfg_val(cfg, attr: str, default, cast=float):
    if cfg is None:
        return cast(default)
    value = getattr(cfg, attr, default)
    if value is None:
        value = default
    return cast(value)


# ── tensor / model helpers ──────────────────────────────────────────────────

def _get_clip_bounds(model) -> Tuple[Optional[float], Optional[float]]:
    clip = getattr(model, "u_clip", (None, None))
    if isinstance(clip, (tuple, list)) and len(clip) == 2:
        return (float(clip[0]) if clip[0] is not None else None,
                float(clip[1]) if clip[1] is not None else None)
    return None, None


def _clamp(x: torch.Tensor, lo: Optional[float], hi: Optional[float]) -> torch.Tensor:
    if lo is not None and hi is not None:
        return x.clamp(lo, hi)
    if lo is not None:
        return x.clamp(min=lo)
    if hi is not None:
        return x.clamp(max=hi)
    return x


# ── motor-neuron resolution ────────────────────────────────────────────────

def _resolve_motor_indices(data: Dict[str, Any], N: int) -> list[int]:
    pre = data.get("motor_neurons")
    if pre is not None and len(pre) > 0:
        return sorted({int(i) for i in pre if 0 <= int(i) < N})

    cfg = data.get("_cfg")
    if cfg is None or getattr(cfg, "motor_neurons", None) is None:
        return []

    idx: list[int] = []
    unresolved: list[str] = []
    for m in cfg.motor_neurons:
        try:
            v = int(m)
            if 0 <= v < N:
                idx.append(v)
        except (ValueError, TypeError):
            unresolved.append(str(m).strip())

    if unresolved:
        labels = data.get("neuron_labels", [])
        if labels:
            lbl_upper = [str(l).strip().upper() for l in labels[:N]]
            for name in unresolved:
                key = name.upper()
                if key in lbl_upper:
                    idx.append(lbl_upper.index(key))

    return sorted(set(idx))


# ── AR(1) baseline ──────────────────────────────────────────────────────────

def _ar1_smooth(u: torch.Tensor, lam: torch.Tensor, I0: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = torch.empty_like(u)
        out[0] = u[0]
        out[1:] = (1.0 - lam) * u[:-1] + lam * I0
    return out.cpu().numpy()
