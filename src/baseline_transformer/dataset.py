"""Data loading, sliding-window dataset, and temporal CV splits.

Reuses stage2.io_h5.load_data_pt for loading deconvolved u(t) and
behaviour targets.  New code here handles windowing and splitting only.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TransformerBaselineConfig

__all__ = [
    "SlidingWindowDataset",
    "temporal_train_val_test_split",
    "contiguous_cv_folds",
    "load_worm_data",
    "discover_worm_files",
]


# ── Sliding-window dataset ──────────────────────────────────────────────────


class SlidingWindowDataset(Dataset):
    """Sliding-window dataset: context K frames → predict next frame.

    Parameters
    ----------
    u : np.ndarray, shape (T, N)
        Neural activity time series.
    context_length : int
        Number of past frames in each context window.
    start, end : int
        Time-range [start, end) to draw windows from. The first valid
        window starts at ``start`` and predicts frame ``start + context_length``.
        The last window predicts frame ``end - 1``.

    Each item is ``(context, target)`` where
    * context : (K, N)  — frames [t - K + 1, ..., t]
    * target  : (N,)    — frame t + 1
    """

    def __init__(
        self,
        u: np.ndarray,
        context_length: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
        assert u.ndim == 2, f"Expected (T, N), got {u.shape}"
        T, N = u.shape
        if end is None:
            end = T
        self.u = u.astype(np.float32)
        self.K = int(context_length)
        self.N = N

        # Valid prediction indices: we predict u[t] using u[t-K : t].
        # So t must satisfy t >= K and t < end.
        # Also t-K >= start  =>  t >= start + K.
        first_target = max(self.K, start + self.K)
        last_target = end  # exclusive
        if last_target <= first_target:
            self._target_indices = np.array([], dtype=np.int64)
        else:
            self._target_indices = np.arange(first_target, last_target, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._target_indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        t = int(self._target_indices[idx])
        context = self.u[t - self.K : t]      # (K, N)
        target = self.u[t]                     # (N,)
        return context, target


# ── Temporal splits ──────────────────────────────────────────────────────────


def temporal_train_val_test_split(
    T: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, Tuple[int, int]]:
    """Split a recording into contiguous train / val / test regions.

    Returns dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping
    to ``(start, end)`` half-open intervals.
    """
    assert 0 < train_frac < 1
    assert 0 < val_frac < 1
    assert train_frac + val_frac < 1
    t_train = int(round(T * train_frac))
    t_val = int(round(T * (train_frac + val_frac)))
    return {
        "train": (0, t_train),
        "val": (t_train, t_val),
        "test": (t_val, T),
    }


def contiguous_cv_folds(
    n_samples: int,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate *n_folds* contiguous temporal CV folds.

    Each fold returns ``(train_indices, val_indices)`` where the val
    indices are a contiguous block and the train indices are everything
    else (preserving temporal order but no shuffle).

    This matches the convention in ``stage2.behavior_decoder_eval._make_contiguous_folds``.
    """
    assert n_folds >= 2
    idx = np.arange(n_samples, dtype=np.int64)
    sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    sizes[: n_samples % n_folds] += 1

    folds = []
    start = 0
    for size in sizes:
        val_idx = idx[start : start + size]
        train_idx = np.concatenate([idx[:start], idx[start + size :]])
        folds.append((train_idx, val_idx))
        start += size
    return folds


# ── Per-worm data loader ────────────────────────────────────────────────────


def load_worm_data(
    h5_path: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a single worm's data for the Transformer baseline.

    Reads stage-1 deconvolved activity, neuron labels, behaviour targets,
    and motor neuron indices.  Does NOT load connectome matrices.

    Returns
    -------
    dict with keys:
        u          : np.ndarray (T, N_obs) float32
        T, N_obs   : ints
        dt         : float
        labels     : list[str]  (length N_obs)
        b          : np.ndarray (T, L_b) or None  (behaviour eigenworms)
        b_mask     : np.ndarray (T, L_b) or None
        motor_idx  : list[int] or None  (indices into N_obs)
        h5_path    : str
        worm_id    : str
    """
    path = Path(h5_path)
    worm_id = path.stem

    with h5py.File(path, "r") as f:
        # ---- Neural activity ----
        if "stage1/u_mean" not in f:
            raise KeyError(f"stage1/u_mean not found in {h5_path}")
        u_raw = np.array(f["stage1/u_mean"], dtype=np.float64)
        # Ensure (T, N)
        if u_raw.shape[0] < 400 and u_raw.shape[1] >= 400:
            u_raw = u_raw.T
        T, N_obs = u_raw.shape

        # Drop all-NaN neurons
        keep = np.any(np.isfinite(u_raw), axis=0)
        u_raw = u_raw[:, keep]
        N_obs = u_raw.shape[1]

        # Forward-fill NaN
        nan_count = int(np.sum(~np.isfinite(u_raw)))
        if nan_count > 0:
            for j in range(N_obs):
                col = u_raw[:, j]
                nans = ~np.isfinite(col)
                if nans.any() and not nans.all():
                    good = np.where(~nans)[0]
                    u_raw[:, j] = np.interp(np.arange(len(col)), good, col[good])

        u = u_raw.astype(np.float32)

        # ---- dt ----
        dt = 0.6
        if "stage1/params" in f:
            sr = f["stage1/params"].attrs.get("sample_rate_hz", None)
            if sr is not None and float(sr) > 0:
                dt = 1.0 / float(sr)

        # ---- Labels ----
        labels: List[str] = []
        if "gcamp/neuron_labels" in f:
            raw_labels = f["gcamp/neuron_labels"][:]
            all_labels = [s.decode() if isinstance(s, bytes) else str(s) for s in raw_labels]
            labels = [all_labels[i] for i in range(len(all_labels)) if i < len(keep) and keep[i]]
        if len(labels) != N_obs:
            labels = [f"Neuron_{i}" for i in range(N_obs)]

        # ---- Motor neurons ----
        motor_names_path = Path(__file__).parent.parent / "data/used/masks+motor neurons/motor_neurons_with_control.txt"
        motor_idx: Optional[List[int]] = None
        if motor_names_path.exists():
            motor_names = [ln.strip() for ln in motor_names_path.read_text().splitlines() if ln.strip()]
            labels_upper = [l.strip().upper() for l in labels]
            motor_idx = [
                labels_upper.index(m.strip().upper())
                for m in motor_names
                if m.strip().upper() in labels_upper
            ]
            if not motor_idx:
                motor_idx = None

        # ---- Behaviour ----
        b: Optional[np.ndarray] = None
        b_mask: Optional[np.ndarray] = None
        beh_ds = "behaviour/eigenworms_stephens"
        if beh_ds in f:
            b_raw = np.array(f[beh_ds], dtype=np.float64)
            if b_raw.shape[0] < b_raw.shape[1] and b_raw.shape[1] == T:
                b_raw = b_raw.T
            if b_raw.shape[0] == T:
                b_mask = np.isfinite(b_raw).astype(np.float32)
                b = np.nan_to_num(b_raw, nan=0.0).astype(np.float32)

        # ---- sigma_u (per-neuron noise std from stage 1) ----
        sigma_u: Optional[np.ndarray] = None
        if "stage1/params" in f and "sigma_u" in f["stage1/params"]:
            su = np.array(f["stage1/params"]["sigma_u"], dtype=np.float32).ravel()
            if len(su) >= len(keep):
                sigma_u = su[keep] if not np.all(keep) else su[:N_obs]

    return {
        "u": u,
        "T": T,
        "N_obs": N_obs,
        "dt": dt,
        "labels": labels,
        "b": b,
        "b_mask": b_mask,
        "motor_idx": motor_idx,
        "sigma_u": sigma_u,
        "h5_path": str(path),
        "worm_id": worm_id,
    }


def discover_worm_files(h5_dir: str) -> List[str]:
    """Find all .h5 files in a directory (non-recursive)."""
    return sorted(glob.glob(str(Path(h5_dir) / "*.h5")))
