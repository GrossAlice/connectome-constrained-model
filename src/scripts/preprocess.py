#!/usr/bin/env python3
"""Data preprocessing and validation.

Consolidated module for:

* **behaviour** – Process raw H5 files into ``data/used`` (neuronal + behaviour).
* **optogenetics** – Process pumpprobe optogenetic data to H5 format.
* **validate** – Post-fit validation checks for Stage-1 LGSSM outputs.

CLI
---
    python -m scripts.preprocess behaviour    --processed_h5_dir ... --output_dir ...
    python -m scripts.preprocess optogenetics --input_dir ... --output_dir ...

The ``validate`` submodule is a library — imported by ``stage1/run_stage1.py`` and
the ``behaviour`` subcommand — with no CLI entry point.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1.config import Stage1Config


# ====================================================================
# Validation  (was validate.py)
# ====================================================================

__all__ = [
    "validate_results",
    "check_parameters",
    "check_residuals",
    "check_convergence",
    "load_label_table",
    "build_label_lookup",
    "build_traceid_to_roipos0",
    "worm_id_from_path",
    "get_neuron_labels_for_h5",
    # behaviour helpers re-exported for external use
    "normalize_body_angle_fixed_length",
    "compute_eigenworms",
]


def load_label_table(csv_path: str) -> pd.DataFrame:
    df0 = pd.read_csv(csv_path, header=None)
    if df0.shape[1] < 3:
        raise ValueError(
            f"CSV must have >=3 columns (worm, idx, label). Got {df0.shape[1]}"
        )
    df = df0.iloc[:, :3].copy()
    df = df.rename(columns={0: "worm", 1: "idx", 2: "label"})
    df["worm"] = df["worm"].astype(str).str.strip()
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip()
    df = df.dropna(subset=["worm", "idx", "label"])
    df["idx"] = df["idx"].astype(int)
    return df


def build_label_lookup(df: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    return {(r.worm, int(r.idx)): r.label for r in df.itertuples(index=False)}


def build_traceid_to_roipos0(roi_match_1d: np.ndarray) -> Dict[int, int]:
    inv: Dict[int, int] = {}
    for pos0, v in enumerate(roi_match_1d):
        try:
            tid = int(v)
        except Exception:
            continue
        if tid <= 0:
            continue
        if tid not in inv:
            inv[tid] = pos0
    return inv


def worm_id_from_path(p: str) -> str:
    b = os.path.basename(p)
    if b.endswith("-data.h5"):
        return b[: -len("-data.h5")]
    if b.endswith(".h5"):
        return b[: -len(".h5")]
    return b


def get_neuron_labels_for_h5(
    h5_path: str,
    label_lookup: Dict[Tuple[str, int], str],
    n_neurons: int,
    roi_match_dataset: str = "neuropal_registration/roi_match",
) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    with h5py.File(h5_path, "r") as f:
        if roi_match_dataset not in f:
            return labels
        roi_match = np.array(f[roi_match_dataset][()])
        if roi_match.ndim != 1:
            return labels
        traceid_to_roipos0 = build_traceid_to_roipos0(roi_match)
        worm = worm_id_from_path(h5_path)
        for j in range(n_neurons):
            trace_id = j + 1
            roi_pos0 = traceid_to_roipos0.get(trace_id)
            if roi_pos0 is None:
                continue
            csv_idx = roi_pos0 + 1
            lab = label_lookup.get((worm, csv_idx))
            if lab:
                labels[j] = str(lab).strip()
    return labels


# ── parameter / residual / convergence checks ───────────────────────

def check_parameters(
    out: Dict,
    cfg: Stage1Config,
    verbose: bool = True,
) -> Dict[str, bool]:
    checks = {}

    rho = out["rho"]
    is_rho_array = isinstance(rho, np.ndarray)
    lambda_c = out["lambda_c"]
    sigma_c = out["sigma_c"]
    alpha = out["alpha"]
    beta = out["beta"]
    sigma_y = out["sigma_y"]
    sigma_u = out["sigma_u"]

    if verbose:
        print("\nPARAMETER VALIDATION")

    if is_rho_array:
        checks["rho_range"] = np.all((0.5 <= rho) & (rho <= 0.9999))
        if verbose:
            status = "✅" if checks["rho_range"] else "⚠️"
            print(
                f"{status} rho (per-neuron): mean={np.mean(rho):.6f}, std={np.std(rho):.6f}, "
                f"range=[{np.min(rho):.6f}, {np.max(rho):.6f}]"
            )
            if not checks["rho_range"]:
                out_of_range = np.sum((rho < 0.5) | (rho > 0.9999))
                print(f"   ⚠️  {out_of_range} neurons out of expected range 0.5-0.9999")
    else:
        checks["rho_range"] = 0.5 <= rho <= 0.9999
        if verbose:
            status = "✅" if checks["rho_range"] else "⚠️"
            print(
                f"{status} rho={rho:.6f} (should be 0.5-0.9999, got {'OK' if checks['rho_range'] else 'OUT OF RANGE'})"
            )

    dt = 1.0 / float(cfg.sample_rate_hz)

    def _tau_from_lam(lam_val: float) -> float:
        if not np.isfinite(lam_val) or lam_val <= 0.0 or lam_val >= 1.0:
            return np.nan
        return float(-dt / np.log(1.0 - lam_val))

    tau_lo, tau_hi = 0.2, 10.0
    if isinstance(lambda_c, np.ndarray):
        lam_ok = np.isfinite(lambda_c) & (lambda_c > 0.0) & (lambda_c < 1.0)
        taus = np.array([_tau_from_lam(float(v)) for v in lambda_c], dtype=float)
        tau_ok = np.isfinite(taus) & (taus >= tau_lo) & (taus <= tau_hi)
        checks["lambda_c_range"] = bool(np.all(lam_ok & tau_ok))
        lam_desc = (
            f"mean={np.mean(lambda_c):.6f}, std={np.std(lambda_c):.6f}, "
            f"range=[{np.min(lambda_c):.6f}, {np.max(lambda_c):.6f}]"
        )
        tau_desc = (
            f"mean={np.nanmean(taus):.3f}s, std={np.nanstd(taus):.3f}s, "
            f"range=[{np.nanmin(taus):.3f}s, {np.nanmax(taus):.3f}s]"
        )
    else:
        lam_ok = np.isfinite(lambda_c) and (0.0 < float(lambda_c) < 1.0)
        tau = _tau_from_lam(float(lambda_c))
        tau_ok = np.isfinite(tau) and (tau_lo <= tau <= tau_hi)
        checks["lambda_c_range"] = bool(lam_ok and tau_ok)
        lam_desc = f"{float(lambda_c):.6f}"
        tau_desc = f"{tau:.3f}s"

    if verbose:
        status = "✅" if checks["lambda_c_range"] else "⚠️"
        print(
            f"{status} lambda_c: {lam_desc} (implied tau_c={tau_desc} at "
            f"{cfg.sample_rate_hz:g} Hz; expected tau_c={tau_lo}-{tau_hi}s)"
        )

    if verbose:
        print("✅ tau ordering: unrestricted")

    if isinstance(sigma_c, np.ndarray):
        checks["sigma_c_positive"] = np.all(sigma_c > 0)
        sigma_desc = (
            f"mean={np.mean(sigma_c):.6e}, std={np.std(sigma_c):.6e}, "
            f"range=[{np.min(sigma_c):.6e}, {np.max(sigma_c):.6e}]"
        )
    else:
        checks["sigma_c_positive"] = sigma_c > 0
        sigma_desc = f"{sigma_c:.6e}"
    if verbose:
        status = "✅" if checks["sigma_c_positive"] else "❌"
        print(f"{status} sigma_c: {sigma_desc} (should be positive)")

    checks["alpha_clustered"] = (
        0.3 <= np.mean(alpha) <= 3.0 and np.std(alpha) <= np.mean(alpha)
    )
    if verbose:
        status = "✅" if checks["alpha_clustered"] else "⚠️"
        print(
            f"{status} alpha: mean={np.mean(alpha):.3f}±{np.std(alpha):.3f}, "
            f"range=[{np.min(alpha):.3f}, {np.max(alpha):.3f}]"
        )
        if not checks["alpha_clustered"]:
            print("    ⚠️  Unexpected alpha distribution (should be near 1.0)")

    checks["sigma_y_positive"] = np.all(sigma_y > 0)
    if verbose:
        status = "✅" if checks["sigma_y_positive"] else "❌"
        print(
            f"{status} sigma_y: mean={np.mean(sigma_y):.3e}±{np.std(sigma_y):.3e}, "
            f"all positive={'OK' if checks['sigma_y_positive'] else 'CONTAINS ZEROS/NEGATIVES'}"
        )

    checks["sigma_u_positive"] = np.all(sigma_u > 0)
    if verbose:
        status = "✅" if checks["sigma_u_positive"] else "❌"
        print(
            f"{status} sigma_u: mean={np.mean(sigma_u):.3e}±{np.std(sigma_u):.3e}, "
            f"all positive={'OK' if checks['sigma_u_positive'] else 'CONTAINS ZEROS/NEGATIVES'}"
        )

    checks["beta_reasonable"] = np.all(np.isfinite(beta))
    if verbose:
        status = "✅" if checks["beta_reasonable"] else "❌"
        print(
            f"{status} beta: mean={np.nanmean(beta):.3f}, all finite="
            f"{'OK' if checks['beta_reasonable'] else 'CONTAINS NAN/INF'}"
        )

    return checks


def check_residuals(
    X: np.ndarray,
    out: Dict,
    sample_neurons: Optional[list] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    alpha = out["alpha"]
    beta = out["beta"]
    c_mean = out["c_mean"]

    N = X.shape[1]
    if sample_neurons is None:
        sample_neurons = list(range(min(N, 10)))

    results = {}

    if verbose:
        print("\nRESIDUAL ANALYSIS")

    for neuron_id in sample_neurons:
        if neuron_id >= N:
            continue
        y_true = X[:, neuron_id]
        y_recon = alpha[neuron_id] * c_mean[:, neuron_id] + beta[neuron_id]
        residuals = y_true - y_recon

        valid = np.isfinite(residuals)
        residuals_clean = residuals[valid]
        if len(residuals_clean) == 0:
            continue

        mean_resid = np.mean(residuals_clean)
        std_resid = np.std(residuals_clean)

        centered = residuals_clean - np.mean(residuals_clean)
        denom = float(np.dot(centered, centered))
        if denom > 0.0 and centered.size > 1:
            acf_lag1 = float(np.dot(centered[1:], centered[:-1]) / denom)
        else:
            acf_lag1 = np.nan
        if denom > 0.0 and centered.size > 5:
            acf_lag5 = float(np.dot(centered[5:], centered[:-5]) / denom)
        else:
            acf_lag5 = np.nan

        results[neuron_id] = {
            "mean": mean_resid,
            "std": std_resid,
            "acf_lag1": acf_lag1,
            "acf_lag5": acf_lag5,
        }

        if verbose:
            status_mean = "✅" if np.abs(mean_resid) < 0.01 * std_resid else "⚠️"
            status_acf = "✅" if np.isfinite(acf_lag1) and (0.05 < acf_lag1 < 0.3) else "⚠️"
            print(
                f"{status_mean} Neuron {neuron_id}: mean={mean_resid:.3e}, "
                f"std={std_resid:.3e}"
            )
            print(
                f"{status_acf}   ACF lag-1={acf_lag1:.3f} (should be 0.05-0.30), "
                f"lag-5={acf_lag5:.3f}"
            )

    return results


def check_convergence(
    out: Dict,
    verbose: bool = True,
) -> Dict[str, bool]:
    ll_hist = out["ll_hist"]
    checks = {}

    if verbose:
        print("\nCONVERGENCE ANALYSIS")

    checks["ll_increased"] = ll_hist[-1] > ll_hist[0]
    if verbose:
        status = "✅" if checks["ll_increased"] else "❌"
        print(
            f"{status} Log-likelihood: {ll_hist[0]:.1f} → {ll_hist[-1]:.1f} "
            f"(improvement: {ll_hist[-1] - ll_hist[0]:.1f})"
        )

    if len(ll_hist) > 1:
        recent_improvement = ll_hist[-1] - ll_hist[-2]
        improvement_rate = recent_improvement / np.abs(ll_hist[-1])

        checks["plateaued"] = abs(improvement_rate) < 0.001
        if verbose:
            status = "✅" if checks["plateaued"] else "⚠️"
            print(
                f"{status} Recent improvement: {recent_improvement:.3e} "
                f"(rel. change: {improvement_rate:.3e})"
            )
            if not checks["plateaued"]:
                print("    ⚠️  Still improving; consider increasing em_max_iters")

    if len(ll_hist) > 2:
        ll_diff = np.diff(ll_hist)
        n_decreases = np.sum(ll_diff < -0.1)
        checks["smooth"] = n_decreases == 0
        if verbose:
            status = "✅" if checks["smooth"] else "⚠️"
            print(
                f"{status} LL trajectory smoothness: {n_decreases} decreases "
                f"({'smooth' if checks['smooth'] else 'has dips'})"
            )

    return checks


def validate_results(
    X: np.ndarray,
    out: Dict,
    cfg: Stage1Config,
    sample_neurons: Optional[list] = None,
    verbose: bool = True,
) -> Dict[str, bool]:
    all_checks = {}

    param_checks = check_parameters(out, cfg, verbose=verbose)
    all_checks.update({f"param_{k}": v for k, v in param_checks.items()})

    check_residuals(X, out, sample_neurons=sample_neurons, verbose=verbose)

    conv_checks = check_convergence(out, verbose=verbose)
    all_checks.update({f"conv_{k}": v for k, v in conv_checks.items()})

    if verbose:
        n_passed = sum(all_checks.values())
        n_total = len(all_checks)
        print(f"\nPassed {n_passed}/{n_total} checks")
        if all(all_checks.values()):
            print("✅ Results look valid!")
        else:
            failed = [k for k, v in all_checks.items() if not v]
            print(f"⚠️  Failed: {', '.join(failed)}")

    return all_checks


# ====================================================================
# Behaviour processing helpers  (was process_behaviour.py)
# ====================================================================

def ensure_TN(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"trace array must be 2D, got {X.shape}")
    if X.shape[0] < 400 and X.shape[1] >= 400:
        if X.shape[1] / max(X.shape[0], 1) > 2:
            return X.T
    return X


def load_shapes_and_compute_eigenvectors(shapes_path: Path):
    """Load shapes.csv and compute eigenvectors via PCA."""
    data = np.loadtxt(str(shapes_path), delimiter=",")
    N, D = data.shape
    x_mean = np.mean(data, axis=0)
    x_cent = data - x_mean
    cov = (x_cent.T @ x_cent) / N
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvecs, eigvals


def _resample_eigen_basis_to_segments(
    eigvecs: np.ndarray,
    n_target_segments: int,
) -> np.ndarray:
    """Match eigenworm basis segment count to body-angle segment count."""
    if eigvecs.ndim != 2:
        raise ValueError(f"eigvecs must be 2D, got shape {eigvecs.shape}")
    src_segments = eigvecs.shape[0]
    if src_segments == n_target_segments:
        return eigvecs
    s_src = np.linspace(0.0, 1.0, src_segments)
    s_tgt = np.linspace(0.0, 1.0, int(n_target_segments))
    out = np.empty((int(n_target_segments), eigvecs.shape[1]), dtype=float)
    for j in range(eigvecs.shape[1]):
        out[:, j] = np.interp(s_tgt, s_src, eigvecs[:, j])
    q, _ = np.linalg.qr(out)
    return q


def _angles_to_xy(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert tangent angles to centerline points (unit step per segment)."""
    dx = np.cos(angles)
    dy = np.sin(angles)
    x = np.concatenate(([0.0], np.cumsum(dx)))
    y = np.concatenate(([0.0], np.cumsum(dy)))
    return x, y


def _longest_finite_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Return [start, end) of the longest contiguous finite run."""
    best = None
    best_len = 0
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif (not ok) and start is not None:
            n = i - start
            if n > best_len:
                best_len = n
                best = (start, i)
            start = None
    if start is not None:
        n = len(mask) - start
        if n > best_len:
            best = (start, len(mask))
            best_len = n
    if best is None or best_len < 2:
        return None
    return best


def normalize_body_angle_fixed_length(
    body_angle: np.ndarray,
    n_target_segments: int | None = None,
    min_valid_segments: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each frame to a fixed worm length by arc-length resampling."""
    if body_angle.ndim != 2:
        return body_angle, np.zeros(0, dtype=bool)
    if body_angle.shape[0] < body_angle.shape[1]:
        body_angle = body_angle.T

    T, D = body_angle.shape
    D_target = D if n_target_segments is None else int(n_target_segments)
    out = np.full((T, D_target), np.nan, dtype=float)
    valid_mask = np.zeros(T, dtype=bool)

    for t in range(T):
        row = body_angle[t]
        finite = np.isfinite(row)
        run = _longest_finite_run(finite)
        if run is None:
            continue
        a, b = run
        if (b - a) < min_valid_segments:
            continue
        angles_obs = row[a:b]
        x, y = _angles_to_xy(angles_obs)
        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.sqrt(dx * dx + dy * dy)
        s = np.concatenate(([0.0], np.cumsum(seg_len)))
        if len(s) < 2 or s[-1] <= 0:
            continue
        s_new = np.linspace(0.0, s[-1], D_target + 1)
        x_new = np.interp(s_new, s, x)
        y_new = np.interp(s_new, s, y)
        dnx = np.diff(x_new)
        dny = np.diff(y_new)
        out[t] = np.arctan2(dny, dnx)
        valid_mask[t] = True

    return out, valid_mask


def compute_eigenworms(body_angle: np.ndarray, eigvecs: np.ndarray, n_modes: int = 5):
    """Compute eigenworm coefficients using Stephens-style preprocessing."""
    if body_angle.ndim != 2:
        return None, None
    body_angle_norm, valid_mask = normalize_body_angle_fixed_length(body_angle)
    if body_angle_norm.ndim != 2 or body_angle_norm.shape[0] == 0:
        return None, None

    T, D = body_angle_norm.shape
    finite_mask = np.all(np.isfinite(body_angle_norm), axis=1)
    if valid_mask.shape[0] == T:
        finite_mask = finite_mask & valid_mask
    n_valid = np.sum(finite_mask)
    if n_valid == 0:
        return None, None

    X = body_angle_norm[finite_mask]
    mean_per_frame = np.mean(X, axis=1, keepdims=True)
    X = X - mean_per_frame
    X = X - np.mean(X, axis=0, keepdims=True)

    eigvecs_use = _resample_eigen_basis_to_segments(eigvecs, D)
    n_modes = min(n_modes, eigvecs_use.shape[1])
    E = eigvecs_use[:, :n_modes]
    coeffs_valid = X @ E

    coeffs = np.full((T, n_modes), np.nan, dtype=float)
    coeffs[finite_mask] = coeffs_valid

    mode_var = np.var(coeffs_valid, axis=0, ddof=0)
    total_var = np.sum(np.var(X, axis=0, ddof=0))
    var_explained = (mode_var / total_var) * 100 if total_var > 0 else np.zeros(n_modes)

    return coeffs, var_explained


# ── label cleaning ───────────────────────────────────────────────────

def standardize_neuron_name(name: str) -> str:
    """Standardize neuron names — strip whitespace, keep leading zeros."""
    return name.strip()


def _standardize_neuron_name_opto(name: str) -> str:
    """Standardize neuron names for optogenetics data — fix single-digit motor neurons."""
    name = name.strip()
    match = re.match(r"^([A-Z]{2,3})(\d)([LR]?)$", name)
    if match:
        prefix, digit, suffix = match.groups()
        if prefix != "IL":
            name = f"{prefix}{digit.zfill(2)}{suffix}"
    return name


def smart_map_neurons_to_connectome(labels: list, allowed_neurons: set) -> tuple:
    """Map neuron labels to connectome neurons with intelligent recovery."""
    n_neurons = len(labels)
    mapped_labels = []
    keep_mask = np.zeros(n_neurons, dtype=bool)
    used_names = set()

    for i, label in enumerate(labels):
        if label in allowed_neurons and label not in used_names:
            mapped_labels.append(label)
            keep_mask[i] = True
            used_names.add(label)
        else:
            mapped_labels.append(None)

    for i, label in enumerate(labels):
        if keep_mask[i]:
            continue
        recovered_name = None
        if "-alt" in label:
            base_name = re.sub(r"-alt\d*$", "", label)
            if base_name in allowed_neurons and base_name not in used_names:
                recovered_name = base_name
        if recovered_name is None and label:
            if not label.endswith("L") and not label.endswith("R"):
                candidate_L = label + "L"
                if candidate_L in allowed_neurons and candidate_L not in used_names:
                    recovered_name = candidate_L
                else:
                    candidate_R = label + "R"
                    if candidate_R in allowed_neurons and candidate_R not in used_names:
                        recovered_name = candidate_R
        if recovered_name:
            mapped_labels[i] = recovered_name
            keep_mask[i] = True
            used_names.add(recovered_name)

    final_labels = [
        lbl for lbl, keep in zip(mapped_labels, keep_mask) if keep and lbl is not None
    ]
    return final_labels, keep_mask


# ====================================================================
# Behaviour processing  (was process_behaviour.py main logic)
# ====================================================================

def _copy_group(src: h5py.Group, dest: h5py.Group) -> None:
    """Recursively copy all datasets in an HDF5 group."""
    for name, item in src.items():
        if isinstance(item, h5py.Dataset):
            dest.create_dataset(
                name, data=item[()], compression="gzip", compression_opts=4
            )
            for attr_name, attr_val in item.attrs.items():
                dest[name].attrs[attr_name] = attr_val
        elif isinstance(item, h5py.Group):
            dest_sub = dest.require_group(name)
            _copy_group(item, dest_sub)


def process_one_behaviour_file(
    h5_path: str,
    label_lookup: Optional[Dict],
    output_dir: str,
    allowed_neurons: Optional[set] = None,
    overwrite: bool = False,
) -> bool:
    """Process a single H5 file: copy data, add labels, and clean labels."""
    basename = os.path.basename(h5_path)
    out_name = basename.replace("-data.h5", ".h5")
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path) and not overwrite:
        return False

    try:
        with h5py.File(h5_path, "r") as src:
            if "gcamp/trace_array_original" not in src:
                print(f"[skip] {basename}: no trace_array_original")
                return False
            has_behavior = "behavior" in src

            X = np.array(src["gcamp/trace_array_original"][()], dtype=float)
            X = ensure_TN(X)
            T, n_neurons = X.shape

            if label_lookup is not None:
                labels_dict = get_neuron_labels_for_h5(h5_path, label_lookup, n_neurons)
            else:
                labels_dict = {}

            labels = [labels_dict.get(i, str(i)) for i in range(n_neurons)]
            labels_cleaned = [l.replace("?", "") for l in labels]
            labels_std = [standardize_neuron_name(l) for l in labels_cleaned]
            labels_std_full = list(labels_std)

            if allowed_neurons is not None:
                labels_std, keep_mask = smart_map_neurons_to_connectome(
                    labels_std, allowed_neurons
                )
                if not np.any(keep_mask):
                    print(f"[skip] {basename}: no neurons match mask")
                    return False
                deleted_labels = [
                    label
                    for label, keep in zip(labels_std_full, keep_mask)
                    if not keep
                ]
                if deleted_labels:
                    print(
                        f"[filtered] {basename}: removed {len(deleted_labels)} neurons -> "
                        + ", ".join(deleted_labels)
                    )
                X = X[:, keep_mask]
                T, n_neurons = X.shape
            else:
                n_neurons = X.shape[1]

            os.makedirs(output_dir, exist_ok=True)
            with h5py.File(out_path, "w") as dest:
                gcamp_grp = dest.require_group("gcamp")
                gcamp_grp.create_dataset(
                    "trace_array_original",
                    data=X,
                    compression="gzip",
                    compression_opts=4,
                )
                dt = h5py.string_dtype(encoding="utf-8")
                gcamp_grp.create_dataset(
                    "neuron_labels", data=np.array(labels_std, dtype=dt)
                )
                if "behavior" in src:
                    beh_dest = dest.require_group("behavior")
                    for key in src["behavior"].keys():
                        if key not in [
                            "body_angle",
                            "body_angle_absolute",
                            "eigenworms",
                        ] and not key.startswith("a_"):
                            if isinstance(src["behavior"][key], h5py.Dataset):
                                beh_dest.create_dataset(
                                    key,
                                    data=src["behavior"][key][()],
                                    compression="gzip",
                                    compression_opts=4,
                                )
                if "timing" in src:
                    time_dest = dest.require_group("timing")
                    _copy_group(src["timing"], time_dest)
                    if "timestamp_nir" in time_dest.attrs:
                        del time_dest.attrs["timestamp_nir"]

        status = "behavior copied" if has_behavior else "no behavior"
        print(
            f"[processed] {out_name}: {n_neurons} neurons, {T} frames ({status}, no eigenworms)"
        )
        return True

    except Exception as e:
        print(f"[error] {basename}: {e}")
        return False


# ====================================================================
# Optogenetics processing  (was process_optogenetics.py)
# ====================================================================

def _is_valid_neuron_label(label: str) -> bool:
    """Return True if *label* is a real neuron name."""
    if not label:
        return False
    stripped = label.lstrip("-")
    if not stripped:
        return False
    if stripped.isdigit():
        return False
    return True


def process_one_opto_folder(
    folder_path: Path,
    output_dir: Path,
    allowed_neurons: Optional[set] = None,
    overwrite: bool = False,
) -> bool:
    """Process a single pumpprobe folder into H5 format."""
    folder_name = folder_path.name
    out_name = f"{folder_name}.h5"
    out_path = output_dir / out_name

    if out_path.exists() and not overwrite:
        return False

    try:
        neural_data = np.loadtxt(folder_path / "neural_data.txt", delimiter=",")
        nan_mask = np.loadtxt(folder_path / "nan_mask.txt", delimiter=",")

        with open(folder_path / "cell_ids.txt", "r") as f:
            cell_ids = [line.strip().strip('"') for line in f.readlines()]

        stim_cells = np.loadtxt(
            folder_path / "stim_cell_indicies.txt", delimiter=",", dtype=int
        )
        stim_frames = np.loadtxt(
            folder_path / "stim_frame_indicies.txt", delimiter=",", dtype=int
        )

        T_raw, N_raw = neural_data.shape

        labels_cleaned = [
            _standardize_neuron_name_opto(lbl) if lbl else "" for lbl in cell_ids
        ]

        valid_mask = np.array(
            [_is_valid_neuron_label(lbl) for lbl in labels_cleaned], dtype=bool
        )
        n_invalid = int((~valid_mask).sum())
        invalid_labels = [
            f"idx={i} '{labels_cleaned[i]}'"
            for i in range(len(labels_cleaned))
            if not valid_mask[i] and labels_cleaned[i]
        ]
        if invalid_labels:
            print(
                f"  [filter] dropping {len(invalid_labels)} invalid-label "
                f"neurons: {', '.join(invalid_labels)}"
            )
        if n_invalid > len(invalid_labels):
            n_empty = n_invalid - len(invalid_labels)
            print(f"  [filter] dropping {n_empty} unlabeled (empty) neurons")

        old_to_new_label: dict[int, int] = {}
        new_idx = 0
        for old_idx in range(N_raw):
            if valid_mask[old_idx]:
                old_to_new_label[old_idx] = new_idx
                new_idx += 1

        neural_data = neural_data[:, valid_mask]
        nan_mask = nan_mask[:, valid_mask]
        labels_cleaned = [lbl for lbl, v in zip(labels_cleaned, valid_mask) if v]

        stim_pairs_raw = list(zip(stim_frames.tolist(), stim_cells.tolist()))
        stim_pairs = [
            (fr, old_to_new_label[ci])
            for fr, ci in stim_pairs_raw
            if ci >= 0 and ci in old_to_new_label
        ]
        n_stim_dropped = len(stim_pairs_raw) - len(stim_pairs)
        n_neg_stim = int((stim_cells < 0).sum())
        if n_stim_dropped:
            print(
                f"  [filter] dropped {n_stim_dropped} stim events "
                f"({n_neg_stim} negative idx, "
                f"{n_stim_dropped - n_neg_stim} removed-neuron)"
            )

        stim_frames_new = np.array([p[0] for p in stim_pairs], dtype=int)
        stim_cells_new = np.array([p[1] for p in stim_pairs], dtype=int)

        all_raw_stim_frames = set(int(fr) for fr, ci in stim_pairs_raw if 0 <= int(fr))

        T, N = neural_data.shape

        if allowed_neurons is not None:
            labels_std, keep_mask = smart_map_neurons_to_connectome(
                labels_cleaned, allowed_neurons
            )
            if not np.any(keep_mask):
                print(f"[skip] {folder_name}: no neurons match connectome")
                return False
            deleted_neurons = [
                labels_cleaned[i]
                for i in range(len(labels_cleaned))
                if not keep_mask[i] and labels_cleaned[i]
            ]
            if deleted_neurons:
                print(
                    f"  [connectome] removed {len(deleted_neurons)} "
                    f"non-atlas neurons: {', '.join(deleted_neurons)}"
                )

            old_to_new_conn: dict[int, int] = {}
            new_idx = 0
            for old_idx in range(N):
                if keep_mask[old_idx]:
                    old_to_new_conn[old_idx] = new_idx
                    new_idx += 1

            neural_data = neural_data[:, keep_mask]
            nan_mask = nan_mask[:, keep_mask]

            stim_pairs = [
                (fr, old_to_new_conn[ci])
                for fr, ci in zip(stim_frames_new, stim_cells_new)
                if int(ci) in old_to_new_conn
            ]
            stim_frames_new = (
                np.array([p[0] for p in stim_pairs], dtype=int)
                if stim_pairs
                else np.array([], dtype=int)
            )
            stim_cells_new = (
                np.array([p[1] for p in stim_pairs], dtype=int)
                if stim_pairs
                else np.array([], dtype=int)
            )
            T, N = neural_data.shape
        else:
            labels_std = labels_cleaned

        neural_data[nan_mask.astype(bool)] = np.nan

        output_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            gcamp_grp = f.create_group("gcamp")
            gcamp_grp.create_dataset(
                "trace_array_original",
                data=neural_data,
                compression="gzip",
                compression_opts=4,
            )
            dt = h5py.string_dtype(encoding="utf-8")
            gcamp_grp.create_dataset(
                "neuron_labels", data=np.array(labels_std, dtype=dt)
            )

            stim_grp = f.create_group("optogenetics")
            stim_grp.create_dataset("stim_cell_indices", data=stim_cells_new)
            stim_grp.create_dataset("stim_frame_indices", data=stim_frames_new)
            stim_grp.attrs["description"] = "Pumpprobe optogenetic stimulation"
            stim_grp.attrs["n_stim_events"] = len(stim_cells_new)
            stim_grp.attrs["n_unique_stim_cells"] = int(
                len(np.unique(stim_cells_new)) if len(stim_cells_new) else 0
            )
            stim_grp.attrs["n_unique_stim_frames"] = int(
                len(np.unique(stim_frames_new)) if len(stim_frames_new) else 0
            )

            stim_matrix = np.zeros((T, N), dtype=np.uint8)
            for frame_idx, cell_idx in zip(
                stim_frames_new.tolist(), stim_cells_new.tolist()
            ):
                if 0 <= frame_idx < T and 0 <= cell_idx < N:
                    stim_matrix[frame_idx, cell_idx] = 1
            stim_grp.create_dataset("stim_matrix", data=stim_matrix)

            stim_any = np.zeros(T, dtype=np.uint8)
            for fr in all_raw_stim_frames:
                if 0 <= fr < T:
                    stim_any[fr] = 1
            stim_grp.create_dataset("stim_any", data=stim_any)
            stim_grp.create_dataset(
                "stim_matrix_with_any",
                data=np.column_stack([stim_matrix, stim_any]),
            )
            stim_grp.attrs["n_raw_stim_events"] = len(all_raw_stim_frames)
            stim_grp.attrs["n_dropped_stim_events"] = len(all_raw_stim_frames) - int(
                np.sum(np.any(stim_matrix > 0, axis=1))
            )

            f.attrs["source_folder"] = folder_name
            f.attrs["T"] = T
            f.attrs["N"] = N
            f.attrs["n_named_neurons"] = N
            f.attrs["n_stim_events"] = len(stim_cells_new)

        _print_h5_summary(out_path, folder_name)
        return True

    except Exception as e:
        import traceback
        print(f"[error] {folder_name}: {e}")
        traceback.print_exc()
        return False


def _print_h5_summary(h5_path: Path, folder_name: str) -> None:
    """Print a detailed summary of the generated H5 file."""
    with h5py.File(h5_path, "r") as f:
        T = f.attrs["T"]
        N = f.attrs["N"]

        labels = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in f["gcamp/neuron_labels"][:]
        ]
        traces = f["gcamp/trace_array_original"][:]
        nan_frac = np.isnan(traces).mean() * 100

        stim_matrix = f["optogenetics/stim_matrix"][:]
        n_stim_events = int(f["optogenetics"].attrs.get("n_stim_events", 0))
        n_unique_cells = int(f["optogenetics"].attrs.get("n_unique_stim_cells", 0))
        n_unique_frames = int(f["optogenetics"].attrs.get("n_unique_stim_frames", 0))
        stim_per_neuron = stim_matrix.sum(axis=0)
        neurons_stimulated = int((stim_per_neuron > 0).sum())

        print(f"\n  ┌── {folder_name} ──────────────────────────")
        print(f"  │ File: {h5_path.name}")
        print(f"  │")
        print(f"  │ gcamp/trace_array_original : ({T}, {N})  float64")
        print(f"  │   NaN fraction             : {nan_frac:.1f}%")
        print(f"  │ gcamp/neuron_labels        : ({N},)  UTF-8 strings")
        print(f"  │   All labels are valid named neurons (no empty/numeric)")
        print(
            f"  │   Labels: {', '.join(labels[:8])}{'...' if N > 8 else ''}"
        )
        print(f"  │")
        print(f"  │ optogenetics/stim_matrix       : ({T}, {N})  uint8")
        print(f"  │   matrix sum                   : {int(stim_matrix.sum())}")
        print(f"  │   neurons receiving stim        : {neurons_stimulated}/{N}")
        print(f"  │ optogenetics/stim_cell_indices  : ({n_stim_events},)  int")
        print(f"  │ optogenetics/stim_frame_indices : ({n_stim_events},)  int")
        print(f"  │   unique stim cells             : {n_unique_cells}")
        print(f"  │   unique stim frames            : {n_unique_frames}")
        print(f"  │   (negative stim indices excluded during processing)")
        print(f"  │")
        print(f"  │ File attrs:")
        for k in sorted(f.attrs.keys()):
            print(f"  │   {k} = {f.attrs[k]}")
        print(f"  │ optogenetics attrs:")
        for k in sorted(f["optogenetics"].attrs.keys()):
            print(f"  │   {k} = {f['optogenetics'].attrs[k]}")
        print(f"  └{'─' * 50}\n")


# ====================================================================
# Subcommand: behaviour
# ====================================================================

def _cmd_behaviour(args) -> int:
    allowed_neurons = None
    if args.neuron_mask:
        mask_path = Path(args.neuron_mask)
        if not mask_path.exists():
            print(f"[error] Neuron mask file not found: {mask_path}")
            return 1
        neuron_list = np.load(str(mask_path), allow_pickle=True)
        allowed_neurons = {standardize_neuron_name(str(n)) for n in neuron_list}
        print(f"[mask] Loaded {len(allowed_neurons)} allowed neurons from {mask_path}")

    label_lookup = None
    if args.label_csv:
        try:
            df_labels = load_label_table(args.label_csv)
            label_lookup = build_label_lookup(df_labels)
            print(f"[labels] Loaded {len(df_labels)} entries from {args.label_csv}")
        except Exception as e:
            print(f"[warning] Failed to load label CSV: {e}")

    input_dir = Path(args.processed_h5_dir)
    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        print(f"[error] No H5 files found in {input_dir}")
        return 1

    print(f"\nProcessing {len(h5_files)} files...")
    print("=" * 70)

    success_count = 0
    for h5_path in tqdm(h5_files, desc="Processing"):
        if process_one_behaviour_file(
            h5_path=str(h5_path),
            label_lookup=label_lookup,
            output_dir=args.output_dir,
            allowed_neurons=allowed_neurons,
            overwrite=args.overwrite,
        ):
            success_count += 1

    print("=" * 70)
    print(f"\n✓ Successfully processed {success_count}/{len(h5_files)} files")
    print(f"✓ Output directory: {args.output_dir}")
    return 0


# ====================================================================
# Subcommand: optogenetics
# ====================================================================

def _cmd_optogenetics(args) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    allowed_neurons = None
    if args.neuron_mask:
        mask_path = Path(args.neuron_mask)
        allowed_neurons = set(np.load(mask_path))
        print(f"[mask] Loaded {len(allowed_neurons)} allowed neurons from {mask_path}")

    pumpprobe_folders = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("pumpprobe_")
    )
    if not pumpprobe_folders:
        print(f"[error] No pumpprobe folders found in {input_dir}")
        return 1

    print(f"Found {len(pumpprobe_folders)} pumpprobe folders\n")
    print("=" * 70)

    n_processed = 0
    for folder in tqdm(pumpprobe_folders, desc="Processing"):
        if process_one_opto_folder(folder, output_dir, allowed_neurons, args.overwrite):
            n_processed += 1

    print("\n" + "=" * 70)
    print(f"\n✓ Successfully processed {n_processed}/{len(pumpprobe_folders)} folders")
    print(f"✓ Output directory: {output_dir}")
    return 0


# ====================================================================
# CLI
# ====================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Data preprocessing and validation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── behaviour ────────────────────────────────────────────────────
    p = sub.add_parser(
        "behaviour",
        help="Process raw H5 files into used H5 format (neuronal + behaviour).",
    )
    p.add_argument("--processed_h5_dir", required=True,
                   help="Directory containing raw processed H5 files")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output prepared H5 files")
    p.add_argument("--label_csv",
                   help="CSV mapping worm/date and ROI indices to neuron labels")
    p.add_argument("--neuron_mask",
                   help="Path to .npy file with list of allowed neuron names")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_behaviour)

    # ── optogenetics ─────────────────────────────────────────────────
    p = sub.add_parser(
        "optogenetics",
        help="Process pumpprobe optogenetic data to H5 format.",
    )
    p.add_argument("--input_dir", required=True,
                   help="Directory containing pumpprobe_* folders")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for H5 files")
    p.add_argument("--neuron_mask", default=None,
                   help="Path to .npy file with allowed neuron names")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_optogenetics)

    args = parser.parse_args(argv)
    result = args.func(args)
    return result if result is not None else 0


if __name__ == "__main__":
    sys.exit(main())
