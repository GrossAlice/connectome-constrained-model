"""Multi-worm data loading for joint training with MAP trajectory inference.

Handles two dataset types:
  - Atanas (2023): freely-behaving worms with behaviour + neural activity.
    Contains ``behaviour/eigenworms_calc_6`` and optionally ``stage1/u_mean``.
  - Randi (2023): immobilised worms with optogenetic stimulation + neural
    activity.  Contains ``optogenetics/stim_matrix`` and optionally
    ``stage1/u_mean``.

All worms are embedded into a common atlas coordinate system (up to 302
neurons).  Neurons unobserved in a given worm will later receive MAP
free-variable trajectories during training.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from .config import Stage2PTConfig
from . import get_stage2_logger
# Re-use label-matching utilities from the single-worm loader
from .io_h5 import (
    _match_single_label,
    _recover_labels_to_atlas,
    _resolve_path,
    _ensure_TN,
)

_log = get_stage2_logger("stage2.io_multi")

_SRC_DIR = Path(__file__).resolve().parent.parent
_MASKS_DIR = _SRC_DIR / "data" / "used" / "masks+motor neurons"

__all__ = ["load_multi_worm_data"]


# ═══════════════════════════════════════════════════════════════════════
#  1. Atlas construction
# ═══════════════════════════════════════════════════════════════════════

def _load_full_atlas() -> List[str]:
    """Load the canonical 302-neuron atlas from ``neuron_names.npy``."""
    path = _MASKS_DIR / "neuron_names.npy"
    if not path.exists():
        raise FileNotFoundError(f"Atlas file not found: {path}")
    return [str(n) for n in np.load(str(path))]


def _read_neuron_labels(f: h5py.File) -> Optional[List[str]]:
    """Read neuron labels from an open H5 file."""
    ds = "gcamp/neuron_labels"
    if ds not in f:
        return None
    raw = f[ds][:]
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]


def _match_labels_to_atlas(
    file_labels: List[str],
    full_labels: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Match file labels to atlas, returning (matched_labels, atlas_indices).

    Only labels that resolve to a unique atlas entry are kept.  Returns
    parallel lists: ``matched_labels[i]`` is the atlas name and
    ``atlas_indices[i]`` is its position in the 302-vector.
    """
    recovered = _recover_labels_to_atlas(file_labels, full_labels)
    indices = []
    for lab in recovered:
        try:
            indices.append(full_labels.index(lab))
        except ValueError:
            pass  # should not happen after recovery, but be safe
    return recovered, np.array(indices, dtype=np.int64)


# ── Per-file metadata scan ─────────────────────────────────────────────

def _detect_dataset_type(f: h5py.File) -> str:
    """Return ``'randi'`` if the file has optogenetics data, else ``'atanas'``."""
    return "randi" if "optogenetics" in f else "atanas"


def _has_stage1(f: h5py.File, u_dataset: str = "stage1/u_mean") -> bool:
    return u_dataset in f


def scan_h5(h5_path: str, full_labels: List[str]) -> Optional[Dict[str, Any]]:
    """Read metadata from one H5 file without loading heavy arrays.

    Returns ``None`` if the file cannot be opened or has no neuron labels.
    """
    path = Path(h5_path)
    if not path.exists():
        _log.warning("file_not_found", path=str(h5_path))
        return None
    try:
        with h5py.File(path, "r") as f:
            raw_labels = _read_neuron_labels(f)
            if raw_labels is None or len(raw_labels) == 0:
                return None
            matched, atlas_idx = _match_labels_to_atlas(raw_labels, full_labels)
            if len(matched) == 0:
                return None

            ds_type = _detect_dataset_type(f)
            has_s1 = _has_stage1(f)

            # Trace shape
            trace_ds = "stage1/u_mean" if has_s1 else "gcamp/trace_array_original"
            if trace_ds not in f:
                trace_ds = "gcamp/trace_array_original"
            shape = f[trace_ds].shape
            T_raw = max(shape)  # handles (T,N) or (N,T)

            # Sample rate
            sample_rate_hz = None
            if has_s1 and "stage1/params" in f:
                sample_rate_hz = f["stage1/params"].attrs.get("sample_rate_hz", None)
            if sample_rate_hz is None:
                sample_rate_hz = f.attrs.get("sample_rate_hz", None)
            if sample_rate_hz is not None:
                sample_rate_hz = float(sample_rate_hz)

            return {
                "h5_path": str(path),
                "raw_labels": raw_labels,
                "matched_labels": matched,
                "atlas_idx": atlas_idx,       # into the full 302
                "n_neurons": len(matched),
                "T_raw": T_raw,
                "dataset_type": ds_type,
                "has_stage1": has_s1,
                "sample_rate_hz": sample_rate_hz,
            }
    except Exception as e:
        _log.warning("scan_failed", path=str(h5_path), error=str(e))
        return None


def _build_atlas(
    file_infos: List[Dict[str, Any]],
    full_labels: List[str],
    min_count: int = 0,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build the multi-worm atlas from scanned file infos.

    Parameters
    ----------
    min_count : int
        Minimum number of worms a neuron must appear in.
        0 → include all 302 neurons (full atlas).
        1 → include only neurons observed in at least one worm.

    Returns
    -------
    atlas_idx : ndarray (N_atlas,) int
        Indices into the full 302 atlas.
    atlas_labels : list[str]
        Neuron names in atlas order.
    coverage : ndarray (N_atlas,) int
        Number of worms observing each atlas neuron.
    """
    N_full = len(full_labels)
    count = np.zeros(N_full, dtype=np.int64)
    for info in file_infos:
        for idx in info["atlas_idx"]:
            count[idx] += 1

    if min_count <= 0:
        # Full 302 atlas
        atlas_idx = np.arange(N_full, dtype=np.int64)
    else:
        atlas_idx = np.where(count >= min_count)[0]

    atlas_labels = [full_labels[i] for i in atlas_idx]
    coverage = count[atlas_idx]

    n_never = int((coverage == 0).sum())
    n_rare = int(((coverage > 0) & (coverage < 3)).sum())
    _log.info(
        "atlas_summary",
        N_atlas=len(atlas_idx),
        observed_in_any=int((coverage > 0).sum()),
        never_observed=n_never,
        observed_in_1or2=n_rare,
    )
    return atlas_idx, atlas_labels, coverage


# ═══════════════════════════════════════════════════════════════════════
#  2. Connectivity matrix loading (atlas-scale)
# ═══════════════════════════════════════════════════════════════════════

def _load_npy_matrix(ds_path: Optional[str]) -> Optional[np.ndarray]:
    """Load a .npy matrix, resolving the path against known directories."""
    if ds_path is None:
        return None
    p = _resolve_path(ds_path)
    if not p.exists():
        _log.warning("matrix_not_found", path=str(ds_path))
        return None
    return np.load(str(p)).astype(np.float32)


def _subset_to_atlas(full_mat: np.ndarray, atlas_idx: np.ndarray) -> np.ndarray:
    """Subset a (302, 302) matrix to atlas coordinates."""
    return full_mat[np.ix_(atlas_idx, atlas_idx)]


def _load_and_subset_matrix(
    ds_path: Optional[str],
    atlas_idx: np.ndarray,
    N_atlas: int,
    device: torch.device,
) -> torch.Tensor:
    """Load a connectivity .npy, subset to atlas, return as tensor."""
    mat = _load_npy_matrix(ds_path)
    if mat is None:
        out = np.ones((N_atlas, N_atlas), dtype=np.float32)
        np.fill_diagonal(out, 0.0)
        return torch.tensor(out, device=device)
    if mat.shape[0] > N_atlas or mat.shape[1] > N_atlas:
        mat = _subset_to_atlas(mat, atlas_idx)
    return torch.tensor(mat, dtype=torch.float32, device=device)


def _load_and_subset_sign(
    ds_path: Optional[str],
    atlas_idx: np.ndarray,
    N_atlas: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Load the neurotransmitter sign matrix, subset to atlas."""
    mat = _load_npy_matrix(ds_path)
    if mat is None:
        return None
    if mat.shape[0] > N_atlas or mat.shape[1] > N_atlas:
        mat = _subset_to_atlas(mat, atlas_idx)
    return torch.tensor(mat, dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════
#  3. Per-worm data loading
# ═══════════════════════════════════════════════════════════════════════

def _atlas_embedding_indices(
    worm_atlas_idx: np.ndarray,
    global_atlas_idx: np.ndarray,
) -> np.ndarray:
    """Map per-worm atlas indices to positions in the multi-worm atlas.

    ``worm_atlas_idx[i]``  = index into the full 302 for worm neuron *i*.
    ``global_atlas_idx[j]`` = index into the full 302 for atlas neuron *j*.

    Returns ``embed_idx`` such that atlas position ``embed_idx[i]`` corresponds
    to worm neuron *i*.  Neurons not in the atlas are dropped (should not
    happen if atlas ⊇ union of all worms).
    """
    # Build reverse map: full_302_idx → atlas_position
    rev = {int(g): j for j, g in enumerate(global_atlas_idx)}
    embed = []
    keep = []
    for i, widx in enumerate(worm_atlas_idx):
        pos = rev.get(int(widx))
        if pos is not None:
            embed.append(pos)
            keep.append(i)
    return np.array(embed, dtype=np.int64), np.array(keep, dtype=np.int64)


def _load_neural_data_stage1(
    f: h5py.File,
    info: Dict[str, Any],
    u_dataset: str = "stage1/u_mean",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load deconvolved activity + noise params from stage1."""
    u = _ensure_TN(np.array(f[u_dataset], dtype=np.float64))
    T, N_file = u.shape

    # u_var
    u_var = None
    if "stage1/u_var" in f:
        u_var = _ensure_TN(np.array(f["stage1/u_var"], dtype=np.float64))
        if u_var.shape != (T, N_file):
            u_var = None

    # sigma_u, lambda_u from stage1 params
    sigma_u = None
    lambda_u = None
    if "stage1/params" in f:
        g = f["stage1/params"]
        if "sigma_u" in g:
            sigma_u = np.array(g["sigma_u"], dtype=np.float64).ravel()
        if "rho" in g:
            rho = np.array(g["rho"], dtype=np.float64).ravel()
            # lambda_u = 1 - rho (AR(1) decay to leak rate)
            lambda_u = 1.0 - np.clip(rho, 0.0, 0.9999)

    return u, u_var, sigma_u, lambda_u


def _load_neural_data_raw(
    f: h5py.File,
    trace_dataset: str = "gcamp/trace_array_original",
) -> np.ndarray:
    """Load raw GCaMP traces and z-score per neuron (fallback when no stage1)."""
    raw = _ensure_TN(np.array(f[trace_dataset], dtype=np.float64))
    # Per-neuron z-score
    mu = np.nanmean(raw, axis=0, keepdims=True)
    std = np.nanstd(raw, axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (raw - mu) / std


def _embed_in_atlas(
    data: np.ndarray,
    embed_idx: np.ndarray,
    keep_mask: np.ndarray,
    T: int,
    N_atlas: int,
) -> np.ndarray:
    """Embed (T, N_worm) data into (T, N_atlas) with zeros for unobserved."""
    out = np.zeros((T, N_atlas), dtype=data.dtype)
    out[:, embed_idx] = data[:, keep_mask]
    return out


def _embed_vector_in_atlas(
    vec: np.ndarray,
    embed_idx: np.ndarray,
    keep_mask: np.ndarray,
    N_atlas: int,
    fill_value: float = 1.0,
) -> np.ndarray:
    """Embed a per-neuron (N_worm,) vector into (N_atlas,) with fill for unobs."""
    out = np.full(N_atlas, fill_value, dtype=np.float64)
    out[embed_idx] = vec[keep_mask]
    return out


# ── Resampling ──────────────────────────────────────────────────────────

def _resample_1d(x: np.ndarray, T_old: int, T_new: int) -> np.ndarray:
    """Linearly resample a (T_old, ...) array to (T_new, ...)."""
    if T_old == T_new:
        return x
    old_t = np.linspace(0, 1, T_old)
    new_t = np.linspace(0, 1, T_new)
    if x.ndim == 1:
        return np.interp(new_t, old_t, x)
    # Multi-column: interpolate each column
    out = np.empty((T_new, x.shape[1]), dtype=x.dtype)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(new_t, old_t, x[:, j])
    return out


def _maybe_resample(
    arrays: Dict[str, Optional[np.ndarray]],
    original_dt: float,
    target_dt: float,
    T_original: int,
) -> Tuple[Dict[str, Optional[np.ndarray]], int]:
    """Resample arrays if ``original_dt`` differs from ``target_dt`` by > 5%.

    Returns (resampled_arrays, T_new).
    """
    if original_dt <= 0 or target_dt <= 0:
        return arrays, T_original
    ratio = original_dt / target_dt
    if abs(ratio - 1.0) < 0.05:
        return arrays, T_original

    T_new = max(1, int(round(T_original * ratio)))
    _log.info("resampling", T_old=T_original, T_new=T_new,
              dt_old=f"{original_dt:.4f}", dt_new=f"{target_dt:.4f}")
    out = {}
    for k, arr in arrays.items():
        if arr is None:
            out[k] = None
        elif arr.ndim >= 1 and arr.shape[0] == T_original:
            out[k] = _resample_1d(arr, T_original, T_new)
        else:
            out[k] = arr  # not time-indexed (e.g. per-neuron vectors)
    return out, T_new


# ── Behaviour loading (Atanas only) ────────────────────────────────────

def _load_behaviour(f: h5py.File, T: int) -> Optional[np.ndarray]:
    """Load eigenworm amplitudes from an Atanas file."""
    for ds in ("behaviour/eigenworms_calc_6", "behaviour/eigenworms_calc_5"):
        if ds in f:
            b = _ensure_TN(np.array(f[ds], dtype=np.float64))
            if b.shape[0] == T:
                return b
            _log.warning("behaviour_T_mismatch", ds=ds,
                         T_beh=b.shape[0], T_neural=T)
    return None


# ── Stimulus loading (Randi only) ──────────────────────────────────────

def _load_optogenetic_stimulus(
    f: h5py.File,
    T: int,
    N_file: int,
) -> Optional[np.ndarray]:
    """Load the optogenetic stim_matrix from a Randi file.

    Returns (T, N_file) float array where 1.0 = stimulated.
    """
    ds = "optogenetics/stim_matrix"
    if ds not in f:
        return None
    stim = np.array(f[ds], dtype=np.float64)
    if stim.ndim == 2:
        if stim.shape == (T, N_file):
            return stim
        if stim.shape == (N_file, T):
            return stim.T
    _log.warning("stim_matrix_shape", shape=stim.shape, expected=(T, N_file))
    return None


# ── Temporal validation split ──────────────────────────────────────────

def _build_val_mask(T: int, val_frac: float) -> np.ndarray:
    """Create boolean mask: True = held out (last ``val_frac`` of time).

    Uses contiguous tail split for temporal evaluation (no data leakage
    from future-to-past).
    """
    n_val = max(1, int(round(T * val_frac)))
    mask = np.zeros(T, dtype=bool)
    mask[-n_val:] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  4. Single-worm loader (orchestrates the above)
# ═══════════════════════════════════════════════════════════════════════

def _load_worm(
    info: Dict[str, Any],
    global_atlas_idx: np.ndarray,
    atlas_labels: List[str],
    N_atlas: int,
    common_dt: float,
    val_frac: float,
    cfg: Stage2PTConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Load one worm's data and embed into atlas coordinates.

    Returns a dict with all per-worm tensors on *device*.
    """
    h5_path = info["h5_path"]
    ds_type = info["dataset_type"]
    has_s1 = info["has_stage1"]

    # Map worm neurons → atlas positions
    embed_idx, keep_mask = _atlas_embedding_indices(
        info["atlas_idx"], global_atlas_idx,
    )
    N_worm = len(embed_idx)

    with h5py.File(h5_path, "r") as f:
        # ── Neural activity ──────────────────────────────────────
        if has_s1:
            u_raw, u_var_raw, sigma_u_raw, lambda_u_raw = _load_neural_data_stage1(
                f, info,
            )
        else:
            _log.warning(
                "no_stage1_fallback",
                path=h5_path,
                msg="Using z-scored raw traces (run stage1 for best results)",
            )
            u_raw = _load_neural_data_raw(f)
            u_var_raw = None
            sigma_u_raw = None
            lambda_u_raw = None

        T_raw, N_file = u_raw.shape

        # Drop neurons that didn't match the atlas (keep_mask alignment)
        # keep_mask indexes into the *matched* neurons; we need to handle
        # the case where N_file > len(matched) because of unmatched labels.
        n_matched = info["n_neurons"]
        if N_file > n_matched:
            # The file has more neurons than matched.  We keep only the
            # first n_matched columns (they correspond to matched labels
            # from _recover_labels_to_atlas which preserves order).
            u_raw = u_raw[:, :n_matched]
            if u_var_raw is not None:
                u_var_raw = u_var_raw[:, :n_matched]
            if sigma_u_raw is not None and len(sigma_u_raw) > n_matched:
                sigma_u_raw = sigma_u_raw[:n_matched]
            if lambda_u_raw is not None and len(lambda_u_raw) > n_matched:
                lambda_u_raw = lambda_u_raw[:n_matched]
            N_file = n_matched

        # ── Determine original dt ────────────────────────────────
        original_dt = common_dt  # fallback
        if info["sample_rate_hz"] is not None and info["sample_rate_hz"] > 0:
            original_dt = 1.0 / info["sample_rate_hz"]

        # ── Behaviour (Atanas only) ──────────────────────────────
        behaviour = _load_behaviour(f, T_raw) if ds_type == "atanas" else None

        # ── Stimulus (Randi only) ────────────────────────────────
        stim_raw = None
        if ds_type == "randi":
            stim_raw = _load_optogenetic_stimulus(f, T_raw, N_file)

    # ── Resample to common dt ────────────────────────────────────
    resample_arrays = {"u": u_raw, "u_var": u_var_raw, "behaviour": behaviour,
                       "stim": stim_raw}
    resample_arrays, T = _maybe_resample(resample_arrays, original_dt, common_dt, T_raw)
    u_raw = resample_arrays["u"]
    u_var_raw = resample_arrays["u_var"]
    behaviour = resample_arrays["behaviour"]
    stim_raw = resample_arrays["stim"]

    # ── Replace NaN with 0 in neural data ────────────────────────
    nan_mask = ~np.isfinite(u_raw)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        _log.info("nan_replaced", path=h5_path, n_nan=n_nan)
        u_raw = np.nan_to_num(u_raw, nan=0.0)
    if u_var_raw is not None:
        u_var_raw = np.nan_to_num(u_var_raw, nan=0.0)

    # ── Embed into atlas coordinates ─────────────────────────────
    u_atlas = _embed_in_atlas(u_raw, embed_idx, keep_mask, T, N_atlas)
    u_var_atlas = (
        _embed_in_atlas(u_var_raw, embed_idx, keep_mask, T, N_atlas)
        if u_var_raw is not None else None
    )

    # sigma_u: observed from stage1, unobserved = inflated default
    if sigma_u_raw is not None:
        obs_median_sigma = float(np.median(sigma_u_raw[np.isfinite(sigma_u_raw)]))
    else:
        obs_median_sigma = 1.0
    unobs_sigma = obs_median_sigma * float(cfg.sigma_u_unobs_scale)
    sigma_u_atlas = _embed_vector_in_atlas(
        sigma_u_raw if sigma_u_raw is not None else np.ones(N_file),
        embed_idx, keep_mask, N_atlas, fill_value=unobs_sigma,
    )

    # lambda_u_init: observed from stage1 OLS, unobserved = population median
    if lambda_u_raw is not None:
        obs_median_lam = float(np.median(lambda_u_raw[np.isfinite(lambda_u_raw)]))
    else:
        obs_median_lam = 0.5
    lambda_u_atlas = _embed_vector_in_atlas(
        lambda_u_raw if lambda_u_raw is not None else np.full(N_file, 0.5),
        embed_idx, keep_mask, N_atlas, fill_value=obs_median_lam,
    )

    # Stimulus embedded in atlas (Randi only)
    stim_atlas = None
    if stim_raw is not None:
        stim_atlas = _embed_in_atlas(stim_raw, embed_idx, keep_mask, T, N_atlas)

    # Gating: 1 everywhere (no silencing in these datasets)
    gating_atlas = np.ones((T, N_atlas), dtype=np.float64)

    # obs_mask
    obs_mask = np.zeros(N_atlas, dtype=bool)
    obs_mask[embed_idx] = True

    # val_mask (temporal hold-out)
    val_mask = _build_val_mask(T, val_frac)

    # ── Convert to tensors ───────────────────────────────────────
    def _t(x, dtype=torch.float32):
        return torch.tensor(np.ascontiguousarray(x), dtype=dtype, device=device)

    worm_id = Path(h5_path).stem

    result: Dict[str, Any] = {
        "worm_id": worm_id,
        "h5_path": h5_path,
        "dataset_type": ds_type,
        "has_stage1": has_s1,
        # Neural activity
        "u": _t(u_atlas),                        # (T, N_atlas)
        "u_var": _t(u_var_atlas) if u_var_atlas is not None else None,
        "sigma_u": _t(sigma_u_atlas),             # (N_atlas,)
        "lambda_u_init": _t(lambda_u_atlas),       # (N_atlas,)
        # Masks
        "obs_mask": _t(obs_mask, dtype=torch.bool),  # (N_atlas,)
        "obs_idx": _t(embed_idx, dtype=torch.long),   # (N_obs,)
        "unobs_idx": _t(
            np.setdiff1d(np.arange(N_atlas), embed_idx),
            dtype=torch.long,
        ),                                            # (N_unobs,)
        # Timing
        "T": T,
        "dt": common_dt,
        "dt_original": original_dt,
        # Behaviour (Atanas only)
        "behaviour": _t(behaviour) if behaviour is not None else None,
        # Stimulus (Randi only, atlas-embedded)
        "stim": _t(stim_atlas) if stim_atlas is not None else None,
        # Gating
        "gating": _t(gating_atlas),              # (T, N_atlas)
        # Evaluation
        "val_mask": _t(val_mask, dtype=torch.bool),  # (T,)
        # Display labels (full atlas)
        "neuron_labels": atlas_labels,
        # Bookkeeping
        "N_obs": int(N_worm),
        "N_unobs": int(N_atlas - N_worm),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════
#  5. Worm weighting
# ═══════════════════════════════════════════════════════════════════════

def _assign_worm_weights(
    worms: List[Dict[str, Any]],
    mode: str = "equal",
) -> None:
    """Compute and store ``weight`` in each worm dict.

    Modes:
      ``"equal"``       – 1/W per worm.
      ``"by_observed"`` – weight ∝ N_obs / Σ N_obs (worms observing more
                          neurons contribute proportionally more).
    """
    W = len(worms)
    if W == 0:
        return

    if mode == "by_observed":
        n_obs = np.array([w["N_obs"] for w in worms], dtype=np.float64)
        total = n_obs.sum()
        if total > 0:
            weights = n_obs / total
        else:
            weights = np.full(W, 1.0 / W)
    else:
        # Default: equal
        weights = np.full(W, 1.0 / W)

    for w, wt in zip(worms, weights):
        w["weight"] = float(wt)


# ═══════════════════════════════════════════════════════════════════════
#  6. Main entry point
# ═══════════════════════════════════════════════════════════════════════

def load_multi_worm_data(cfg: Stage2PTConfig) -> Dict[str, Any]:
    """Load data from multiple worms and build a shared atlas.

    Parameters
    ----------
    cfg : Stage2PTConfig
        Must have ``multi.multi_worm = True`` and ``multi.h5_paths`` set.

    Returns
    -------
    dict with keys:
        atlas_labels  : list[str]           – N_atlas neuron names
        atlas_size    : int                 – N_atlas
        atlas_idx     : ndarray (N_atlas,)  – indices into the full 302
        T_e, T_sv, T_dcv : Tensor (N_atlas, N_atlas)
        sign_t        : Tensor | None
        worms         : list[dict]          – per-worm data
        coverage      : ndarray (N_atlas,)  – worms observing each neuron
        full_labels   : list[str]           – the full 302 atlas
    """
    device = torch.device(cfg.device)
    full_labels = _load_full_atlas()

    h5_paths = list(cfg.h5_paths)
    if not h5_paths:
        raise ValueError(
            "multi.h5_paths is empty.  Set --h5_paths or cfg.h5_paths "
            "to a tuple of H5 file paths."
        )

    # ── Scan all files ───────────────────────────────────────────
    require_s1 = bool(cfg.require_stage1)
    file_infos: List[Dict[str, Any]] = []
    n_skipped_no_labels = 0
    n_skipped_no_stage1 = 0

    for p in h5_paths:
        info = scan_h5(p, full_labels)
        if info is None:
            n_skipped_no_labels += 1
            continue
        if require_s1 and not info["has_stage1"]:
            n_skipped_no_stage1 += 1
            continue
        file_infos.append(info)

    if n_skipped_no_labels > 0:
        _log.warning("skipped_no_labels", count=n_skipped_no_labels)
    if n_skipped_no_stage1 > 0:
        _log.warning(
            "skipped_no_stage1",
            count=n_skipped_no_stage1,
            hint="Set require_stage1=False to use raw traces, "
                 "or run stage1 on these files first.",
        )
    if not file_infos:
        raise ValueError(
            f"No valid worms after filtering ({len(h5_paths)} paths given, "
            f"{n_skipped_no_labels} missing labels, "
            f"{n_skipped_no_stage1} missing stage1)."
        )

    # ── Build atlas ──────────────────────────────────────────────
    atlas_idx, atlas_labels, coverage = _build_atlas(
        file_infos, full_labels, min_count=int(cfg.atlas_min_worm_count),
    )
    N_atlas = len(atlas_idx)

    # ── Load connectivity ────────────────────────────────────────
    T_e = _load_and_subset_matrix(cfg.T_e_dataset, atlas_idx, N_atlas, device)
    T_sv = _load_and_subset_matrix(cfg.T_sv_dataset, atlas_idx, N_atlas, device)
    T_sv = T_sv.abs()
    T_dcv = _load_and_subset_matrix(cfg.T_dcv_dataset, atlas_idx, N_atlas, device)
    T_dcv = T_dcv.abs()
    sign_t = _load_and_subset_sign(
        getattr(cfg, "neurotransmitter_sign_dataset", None),
        atlas_idx, N_atlas, device,
    )

    # ── Load each worm ───────────────────────────────────────────
    common_dt = float(cfg.common_dt)
    val_frac = float(cfg.val_frac)

    worms: List[Dict[str, Any]] = []
    n_atanas = n_randi = 0
    for info in file_infos:
        try:
            worm = _load_worm(
                info, atlas_idx, atlas_labels, N_atlas,
                common_dt, val_frac, cfg, device,
            )
            worms.append(worm)
            if worm["dataset_type"] == "atanas":
                n_atanas += 1
            else:
                n_randi += 1
        except Exception as e:
            _log.warning("worm_load_failed", path=info["h5_path"], error=str(e))

    if not worms:
        raise ValueError("No worms loaded successfully after processing.")

    _assign_worm_weights(worms, str(cfg.worm_weight_mode))

    # ── Summary ──────────────────────────────────────────────────
    n_obs_list = [w["N_obs"] for w in worms]
    print(
        f"[MultiWorm] Loaded {len(worms)} worms "
        f"({n_atanas} Atanas, {n_randi} Randi) "
        f"into {N_atlas}-neuron atlas\n"
        f"  Neurons per worm: "
        f"min={min(n_obs_list)} max={max(n_obs_list)} "
        f"mean={np.mean(n_obs_list):.1f}\n"
        f"  Atlas coverage: "
        f"{int((coverage > 0).sum())} observed in ≥1 worm, "
        f"{int((coverage == 0).sum())} never observed"
    )
    for w in worms:
        _log.info(
            "worm_summary",
            id=w["worm_id"],
            type=w["dataset_type"],
            stage1=w["has_stage1"],
            T=w["T"],
            N_obs=w["N_obs"],
            weight=f"{w['weight']:.4f}",
            has_behaviour=w["behaviour"] is not None,
            has_stim=w["stim"] is not None,
        )

    return {
        "atlas_labels": atlas_labels,
        "atlas_size": N_atlas,
        "atlas_idx": atlas_idx,
        "T_e": T_e,
        "T_sv": T_sv,
        "T_dcv": T_dcv,
        "sign_t": sign_t,
        "worms": worms,
        "coverage": coverage,
        "full_labels": full_labels,
    }
