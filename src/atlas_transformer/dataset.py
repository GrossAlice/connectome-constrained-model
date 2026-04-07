"""Multi-worm atlas-embedded dataset for the Atlas Transformer.

Each worm's neurons are mapped into the 302-neuron canonical atlas.
Unobserved positions are zero-padded.  An observation mask is provided
so the model (and loss) knows which neurons are real.

Behaviour eigenworms are included as extra input/output columns when
``include_beh_input`` or ``predict_beh`` is set.

Reuses ``stage2.io_multi`` for atlas label matching and embedding.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from .config import AtlasTransformerConfig

# Re-use the atlas infrastructure from stage2
import sys
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from stage2.io_multi import (
    _load_full_atlas,
    _read_neuron_labels,
    _match_labels_to_atlas,
    _atlas_embedding_indices,
    _embed_in_atlas,
)
from stage2.io_h5 import _ensure_TN

__all__ = [
    "SlidingWindowAtlasDataset",
    "load_atlas_worm_data",
    "load_all_worms_atlas",
    "temporal_train_val_test_split",
    "contiguous_cv_folds",
    "make_cv_folds",
    "build_dataloaders",
    "build_joint_state_atlas",
]


# ── Temporal split (same as per-worm baseline) ──────────────────────────────


def temporal_train_val_test_split(
    T: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, Tuple[int, int]]:
    """Split into contiguous train / val / test regions."""
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
    """Generate *n_folds* contiguous temporal CV folds."""
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
) -> List[Dict[str, Any]]:
    """Create *n_folds* contiguous temporal CV folds as (start, end) ranges.

    Identical to ``baseline_transformer.dataset.make_cv_folds``.
    """
    K = context_length
    usable = T - K
    fold_size = usable // n_folds
    remainder = usable - fold_size * n_folds

    folds = []
    cursor = K
    for fi in range(n_folds):
        size = fold_size + (1 if fi < remainder else 0)
        te_s, te_e = cursor, cursor + size

        train_segments = []
        if te_s > K:
            train_segments.append((K, te_s))
        if te_e < T:
            train_segments.append((te_e, T))

        total_train = sum(e - s for s, e in train_segments)
        val_size = max(1, int(total_train * val_frac_inner))

        last_s, last_e = train_segments[-1]
        val_s = last_e - val_size
        val_e = last_e
        new_train_segments = list(train_segments)
        if val_s > last_s:
            new_train_segments[-1] = (last_s, val_s)
        else:
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


# ── Sliding-window dataset with atlas embedding ─────────────────────────────


class SlidingWindowAtlasDataset(Dataset):
    """Sliding-window dataset for one worm embedded in the atlas.

    The input tensor is the *packed joint state*:
        x = [u_atlas (N_atlas), obs_mask (N_atlas), beh (n_beh)]

    Each item returns:
        context      : (K, D)   — packed joint-state window
        target       : (D,)     — next frame packed state
        target_mask  : (D,)     — 1.0 for valid, 0.0 for invalid
                                  (neural obs mask replicated + beh mask)
    """

    def __init__(
        self,
        x: np.ndarray,           # (T, D) packed joint state
        n_atlas: int,
        context_length: int,
        obs_mask: np.ndarray,     # (N_atlas,) bool
        b_mask: Optional[np.ndarray] = None,  # (T, n_beh)
        start: int = 0,
        end: Optional[int] = None,
        worm_id: str = "",
    ) -> None:
        T, D = x.shape
        if end is None:
            end = T
        self.x = x.astype(np.float32)
        self.K = context_length
        self.D = D
        self.n_atlas = n_atlas
        self.n_beh = D - 2 * n_atlas
        self.worm_id = worm_id

        # Build per-row target mask: neural obs mask (for both u and mask cols)
        # + beh mask
        self._target_mask = np.ones((T, D), dtype=np.float32)
        # Only observed neurons contribute to neural loss
        neural_mask = obs_mask.astype(np.float32)
        self._target_mask[:, :n_atlas] = neural_mask[np.newaxis, :]
        # obs_mask columns are always "valid" (not a prediction target really,
        # but set to 0 so they don't contribute to loss)
        self._target_mask[:, n_atlas:2*n_atlas] = 0.0
        # Behaviour columns
        if b_mask is not None and self.n_beh > 0:
            self._target_mask[:, 2*n_atlas:] = b_mask.astype(np.float32)

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
        context = self.x[t - self.K : t]           # (K, D)
        target = self.x[t]                          # (D,)
        target_mask = self._target_mask[t]          # (D,)
        return context, target, target_mask


# ── Build packed joint state ─────────────────────────────────────────────────


def build_joint_state_atlas(
    u_atlas: np.ndarray,
    obs_mask: np.ndarray,
    b: Optional[np.ndarray] = None,
    b_mask: Optional[np.ndarray] = None,
    include_beh: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Pack atlas activity, observation mask, and (optionally) behaviour.

    Returns
    -------
    x      : (T, D) where D = 2*N_atlas + n_beh
    b_mask : (T, n_beh) or None
    n_beh  : int
    """
    T, N_atlas = u_atlas.shape
    obs_mask_expanded = np.broadcast_to(
        obs_mask.astype(np.float32)[np.newaxis, :], (T, N_atlas),
    ).copy()
    if include_beh and b is not None:
        n_beh = b.shape[1]
        x = np.concatenate([u_atlas, obs_mask_expanded, b], axis=1).astype(np.float32)
        return x, b_mask, n_beh
    else:
        x = np.concatenate([u_atlas, obs_mask_expanded], axis=1).astype(np.float32)
        return x, None, 0


# ── Load a single worm into atlas coordinates ───────────────────────────────


def load_atlas_worm_data(
    h5_path: str,
    full_labels: List[str],
    atlas_idx: np.ndarray,
    n_atlas: int,
    n_beh_modes: int = 6,
) -> Optional[Dict[str, Any]]:
    """Load one worm, embed into atlas.

    Returns
    -------
    dict with:
        u_atlas   : (T, N_atlas) float32  — embedded activity
        obs_mask  : (N_atlas,) bool        — observed neuron mask
        obs_idx   : (N_obs,) int           — atlas positions of observed neurons
        T, N_obs  : ints
        worm_id   : str
        dt        : float
        labels    : list of matched labels
        b, b_mask : behaviour arrays or None
        motor_idx : motor neuron indices in atlas coords or None
    """
    path = Path(h5_path)
    worm_id = path.stem

    try:
        with h5py.File(path, "r") as f:
            # Neural activity
            if "stage1/u_mean" not in f:
                print(f"  [SKIP] {worm_id}: no stage1/u_mean")
                return None
            u_raw = np.array(f["stage1/u_mean"], dtype=np.float64)
            u_raw = _ensure_TN(u_raw)
            T, N_file = u_raw.shape

            # Labels
            file_labels = None
            if "gcamp/neuron_labels" in f:
                raw = f["gcamp/neuron_labels"][:]
                file_labels = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]

            if file_labels is None or len(file_labels) == 0:
                print(f"  [SKIP] {worm_id}: no labels")
                return None

            # Match to atlas
            matched, worm_atlas_idx = _match_labels_to_atlas(file_labels, full_labels)
            if len(matched) == 0:
                print(f"  [SKIP] {worm_id}: no atlas matches")
                return None

            # Map worm neurons → global atlas positions
            embed_idx, keep_mask = _atlas_embedding_indices(
                worm_atlas_idx, atlas_idx,
            )
            N_obs = len(embed_idx)
            if N_obs == 0:
                print(f"  [SKIP] {worm_id}: no neurons in atlas")
                return None

            # Drop NaN neurons, trim to matched count
            if N_file > len(file_labels):
                u_raw = u_raw[:, :len(file_labels)]
                N_file = len(file_labels)

            # Forward-fill NaN
            nan_count = int(np.sum(~np.isfinite(u_raw)))
            if nan_count > 0:
                for j in range(u_raw.shape[1]):
                    col = u_raw[:, j]
                    nans = ~np.isfinite(col)
                    if nans.any() and not nans.all():
                        good = np.where(~nans)[0]
                        u_raw[:, j] = np.interp(np.arange(len(col)), good, col[good])
                u_raw = np.nan_to_num(u_raw, nan=0.0)

            # Embed in atlas
            u_atlas = _embed_in_atlas(
                u_raw.astype(np.float32), embed_idx, keep_mask, T, n_atlas
            )

            # Observation mask
            obs_mask = np.zeros(n_atlas, dtype=bool)
            obs_mask[embed_idx] = True

            # dt
            dt = 0.6
            if "stage1/params" in f:
                sr = f["stage1/params"].attrs.get("sample_rate_hz", None)
                if sr is not None and float(sr) > 0:
                    dt = 1.0 / float(sr)

            # Behaviour
            b = b_mask = None
            beh_ds = "behaviour/eigenworms_stephens"
            if beh_ds in f:
                b_raw = np.array(f[beh_ds], dtype=np.float64)
                if b_raw.shape[0] < b_raw.shape[1] and b_raw.shape[1] == T:
                    b_raw = b_raw.T
                if b_raw.shape[0] == T:
                    # Clip to n_beh_modes (same as baseline)
                    n_modes_available = b_raw.shape[1]
                    n_modes = min(n_beh_modes, n_modes_available)
                    b_raw = b_raw[:, :n_modes]
                    b_mask = np.isfinite(b_raw).astype(np.float32)
                    b = np.nan_to_num(b_raw, nan=0.0).astype(np.float32)

            # Motor neurons in atlas coordinates
            motor_names_path = (
                Path(__file__).parent.parent
                / "data/used/masks+motor neurons/motor_neurons_with_control.txt"
            )
            motor_idx_atlas = None
            if motor_names_path.exists():
                motor_names = [
                    ln.strip() for ln in motor_names_path.read_text().splitlines()
                    if ln.strip()
                ]
                labels_upper = [l.strip().upper() for l in full_labels]
                # Find motor neurons that are in this worm's observed set
                motor_idx_atlas = []
                for mname in motor_names:
                    mu = mname.strip().upper()
                    if mu in labels_upper:
                        atlas_pos = labels_upper.index(mu)
                        # Check if in our global atlas and observed
                        global_pos_list = list(atlas_idx)
                        if atlas_pos in global_pos_list:
                            gpos = global_pos_list.index(atlas_pos)
                            if obs_mask[gpos]:
                                motor_idx_atlas.append(gpos)
                if not motor_idx_atlas:
                    motor_idx_atlas = None

    except Exception as e:
        print(f"  [SKIP] {worm_id}: {e}")
        return None

    return {
        "u_atlas": u_atlas,
        "obs_mask": obs_mask,
        "obs_idx": embed_idx,
        "T": T,
        "N_obs": N_obs,
        "worm_id": worm_id,
        "dt": dt,
        "labels": matched,
        "b": b,
        "b_mask": b_mask,
        "motor_idx_atlas": motor_idx_atlas,
        "h5_path": str(path),
    }


# ── Load all worms ──────────────────────────────────────────────────────────


def load_all_worms_atlas(
    h5_dir: str,
    cfg: AtlasTransformerConfig,
) -> Dict[str, Any]:
    """Load all worms from a directory and embed in the 302-neuron atlas.

    Returns
    -------
    dict with:
        worms       : list of per-worm dicts
        full_labels : 302-neuron label list
        atlas_idx   : (302,) int array (identity for full atlas)
        n_atlas     : 302
    """
    full_labels = _load_full_atlas()
    n_atlas = len(full_labels)
    atlas_idx = np.arange(n_atlas, dtype=np.int64)  # full 302 atlas

    h5_files = sorted(glob.glob(str(Path(h5_dir) / "*.h5")))
    if not h5_files:
        raise ValueError(f"No .h5 files found in {h5_dir}")

    print(f"[Atlas] Found {len(h5_files)} H5 files, loading into {n_atlas}-neuron atlas...")

    worms = []
    for h5_path in h5_files:
        worm = load_atlas_worm_data(
            h5_path, full_labels, atlas_idx, n_atlas,
            n_beh_modes=cfg.n_beh_modes,
        )
        if worm is not None:
            worms.append(worm)

    if not worms:
        raise ValueError("No worms loaded successfully.")

    # Coverage stats
    coverage = np.zeros(n_atlas, dtype=int)
    for w in worms:
        coverage[w["obs_mask"]] += 1

    n_obs_list = [w["N_obs"] for w in worms]
    print(
        f"[Atlas] Loaded {len(worms)} worms into {n_atlas}-neuron atlas\n"
        f"  Neurons per worm: min={min(n_obs_list)} max={max(n_obs_list)} "
        f"mean={np.mean(n_obs_list):.1f}\n"
        f"  Atlas coverage: {int((coverage > 0).sum())} observed in ≥1 worm, "
        f"{int((coverage == 0).sum())} never observed"
    )

    return {
        "worms": worms,
        "full_labels": full_labels,
        "atlas_idx": atlas_idx,
        "n_atlas": n_atlas,
        "coverage": coverage,
    }


# ── Build train/val dataloaders ─────────────────────────────────────────────


def build_dataloaders(
    worms: List[Dict[str, Any]],
    cfg: AtlasTransformerConfig,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, List[Dict[str, Any]]]:
    """Build concatenated train/val DataLoaders from all worms.

    Each worm gets its own temporal split. The train and val datasets
    from all worms are concatenated into one DataLoader each.

    Returns
    -------
    train_loader, val_loader, splits (per-worm split info)
    """
    include_beh = cfg.include_beh_input and cfg.predict_beh
    train_datasets = []
    val_datasets = []
    splits = []

    for worm in worms:
        T = worm["T"]
        split = temporal_train_val_test_split(T, cfg.train_frac, cfg.val_frac)
        splits.append({"worm_id": worm["worm_id"], "split": split, "T": T})

        x, b_mask_out, n_beh = build_joint_state_atlas(
            worm["u_atlas"], worm["obs_mask"],
            worm.get("b"), worm.get("b_mask"),
            include_beh=include_beh,
        )

        tr_s, tr_e = split["train"]
        va_s, va_e = split["val"]

        train_ds = SlidingWindowAtlasDataset(
            x=x,
            n_atlas=cfg.n_atlas,
            context_length=cfg.context_length,
            obs_mask=worm["obs_mask"],
            b_mask=b_mask_out,
            start=tr_s, end=tr_e,
            worm_id=worm["worm_id"],
        )
        val_ds = SlidingWindowAtlasDataset(
            x=x,
            n_atlas=cfg.n_atlas,
            context_length=cfg.context_length,
            obs_mask=worm["obs_mask"],
            b_mask=b_mask_out,
            start=va_s, end=va_e,
            worm_id=worm["worm_id"],
        )

        if len(train_ds) > 0:
            train_datasets.append(train_ds)
        if len(val_ds) > 0:
            val_datasets.append(val_ds)

    if not train_datasets:
        raise ValueError("No training samples across all worms.")

    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets) if val_datasets else combined_train

    print(
        f"[Atlas] Dataset: {len(combined_train)} train samples, "
        f"{len(combined_val)} val samples "
        f"(from {len(train_datasets)} worms)"
    )

    use_pin = device != "cpu"
    train_loader = DataLoader(
        combined_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_pin,
        num_workers=0,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_pin,
        num_workers=0,
    )

    return train_loader, val_loader, splits
