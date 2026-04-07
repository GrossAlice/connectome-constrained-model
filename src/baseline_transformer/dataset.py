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
    "make_cv_folds",
    "load_worm_data",
    "discover_worm_files",
]


# ── Sliding-window dataset ──────────────────────────────────────────────────


class SlidingWindowDataset(Dataset):
    """Sliding-window dataset: context K frames → predict next frame.

    Handles a joint state ``x = [u, b]`` where ``u`` is neural activity
    (always valid) and ``b`` is behaviour (may be partially masked).

    Parameters
    ----------
    x : np.ndarray, shape (T, D)
        Joint state: first ``n_neural`` columns are neural activity,
        remaining ``n_beh`` columns are behaviour eigenworm amplitudes.
    b_mask : np.ndarray, shape (T, n_beh) or None
        Validity mask for the behaviour columns (1=valid, 0=invalid).
        If None, all behaviour columns are treated as always valid.
    n_neural : int
        Number of neural columns in ``x``.
    context_length : int
        Number of past frames in each context window (K).
    start, end : int
        Time-range [start, end) to draw windows from.

    Each item is ``(context, target, target_mask)`` where
    * context     : (K, D)     — frames [t-K, ..., t-1]
    * target      : (D,)       — frame t
    * target_mask : (D,)       — 1.0 for valid, 0.0 for invalid behaviour
    """

    def __init__(
        self,
        x: np.ndarray,
        n_neural: int,
        context_length: int,
        start: int = 0,
        end: Optional[int] = None,
        b_mask: Optional[np.ndarray] = None,
    ) -> None:
        assert x.ndim == 2, f"Expected (T, D), got {x.shape}"
        T, D = x.shape
        if end is None:
            end = T
        self.x = x.astype(np.float32)
        self.K = int(context_length)
        self.D = D
        self.n_neural = n_neural
        self.n_beh = D - n_neural

        # Build full-row mask: neural columns are always valid
        self._mask = np.ones((T, D), dtype=np.float32)
        if b_mask is not None and self.n_beh > 0:
            self._mask[:, n_neural:] = b_mask.astype(np.float32)

        # Valid prediction indices: predict x[t] from x[t-K : t].
        first_target = max(self.K, start + self.K)
        last_target = end
        if last_target <= first_target:
            self._target_indices = np.array([], dtype=np.int64)
        else:
            self._target_indices = np.arange(first_target, last_target, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._target_indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = int(self._target_indices[idx])
        context = self.x[t - self.K : t]         # (K, D)
        target = self.x[t]                        # (D,)
        target_mask = self._mask[t]               # (D,)
        return context, target, target_mask


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


def make_cv_folds(
    T: int,
    n_folds: int,
    context_length: int,
    val_frac_inner: float = 0.15,
) -> List[Dict[str, Tuple[int, int]]]:
    """Create *n_folds* contiguous temporal CV folds as (start, end) ranges.

    Each fold dict has:
        "test"  : (test_start, test_end) — held-out contiguous block
        "train" : list of (start, end) tuples — training segments
        "val"   : (val_start, val_end) — inner validation from training
                  (last ``val_frac_inner`` of the training data)

    The test regions tile the recording [K, T) without overlap.
    """
    K = context_length
    usable = T - K  # first K frames are context-only
    fold_size = usable // n_folds
    remainder = usable - fold_size * n_folds

    folds = []
    cursor = K
    for fi in range(n_folds):
        # This fold's test region
        size = fold_size + (1 if fi < remainder else 0)
        te_s, te_e = cursor, cursor + size

        # Training: everything outside test region (but >= K)
        train_segments = []
        if te_s > K:
            train_segments.append((K, te_s))
        if te_e < T:
            train_segments.append((te_e, T))

        # Inner val: carve off last val_frac_inner of total training frames
        total_train = sum(e - s for s, e in train_segments)
        val_size = max(1, int(total_train * val_frac_inner))

        # Take val from the *end* of the last training segment
        last_s, last_e = train_segments[-1]
        val_s = last_e - val_size
        val_e = last_e
        # Shrink the last training segment
        new_train_segments = list(train_segments)
        if val_s > last_s:
            new_train_segments[-1] = (last_s, val_s)
        else:
            # val eats the entire last segment — use previous one
            new_train_segments.pop()
            if new_train_segments:
                ps, pe = new_train_segments[-1]
                needed = val_size - (last_e - last_s)
                new_val_s = pe - needed
                val_s = new_val_s
                new_train_segments[-1] = (ps, new_val_s)

        folds.append({
            "test": (te_s, te_e),
            "train": new_train_segments,
            "val": (val_s, val_e),
        })
        cursor = te_e

    return folds


# ── Per-worm data loader ────────────────────────────────────────────────────


def load_worm_data(
    h5_path: str,
    device: str = "cpu",
    n_beh_modes: int = 6,
) -> Dict[str, Any]:
    """Load a single worm's data for the Transformer baseline.

    Reads stage-1 deconvolved activity, neuron labels, behaviour targets,
    and motor neuron indices.  Does NOT load connectome matrices.

    Parameters
    ----------
    n_beh_modes : int
        Clip behaviour eigenworms to this many modes.

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
                # Clip to n_beh_modes
                n_modes_available = b_raw.shape[1]
                n_modes = min(n_beh_modes, n_modes_available)
                b_raw = b_raw[:, :n_modes]
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


def build_joint_state(
    u: np.ndarray,
    b: Optional[np.ndarray],
    b_mask: Optional[np.ndarray],
    include_beh: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
    """Build joint state ``x = [u, b]`` and corresponding mask.

    Returns
    -------
    x      : (T, N+L) float32  joint state
    x_mask : (T, L) float32 or None   mask for behaviour columns only
    n_neural : int
    n_beh    : int
    """
    n_neural = u.shape[1]
    if include_beh and b is not None:
        n_beh = b.shape[1]
        x = np.concatenate([u, b], axis=1).astype(np.float32)
        x_mask = b_mask if b_mask is not None else np.ones_like(b)
        return x, x_mask, n_neural, n_beh
    else:
        return u.astype(np.float32), None, n_neural, 0


def discover_worm_files(h5_dir: str) -> List[str]:
    """Find all .h5 files in a directory (non-recursive)."""
    return sorted(glob.glob(str(Path(h5_dir) / "*.h5")))
