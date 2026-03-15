from __future__ import annotations

import numpy as np
import torch
import h5py
import re
from typing import Dict, Any, Optional
from pathlib import Path

from .config import Stage2PTConfig
from . import get_stage2_logger

_io_logger = get_stage2_logger("stage2.io")

__all__ = ["load_data_pt", "save_results_pt"]

_PKG_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PKG_DIR.parent

def _ensure_TN(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {X.shape}")
    if X.shape[0] < 400 and X.shape[1] >= 400 and X.shape[1] / max(X.shape[0], 1) > 2:
        return X.T
    return X


def _resolve_path(name: str | Path) -> Path:
    p = Path(name).expanduser()
    if p.exists():
        return p
    for base in (_PKG_DIR, _SRC_DIR):
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return p
def _match_single_label(s: str, atlas: set[str], used: set[str]) -> Optional[str]:
    if s in atlas and s not in used:
        return s
    base = re.sub(r"-alt\d*$", "", s)
    if base in atlas and base not in used:
        return base
    if not base.endswith(("L", "R")):
        for suffix in ("L", "R"):
            c = base + suffix
            if c in atlas and c not in used:
                return c
    return None


def _recover_labels_to_atlas(labels_in: list[str], atlas_labels: list[str]) -> list[str]:
    atlas_set = {str(x) for x in atlas_labels}
    used: set[str] = set()
    out: list[str] = []
    for lab in labels_in:
        target = _match_single_label(str(lab).strip(), atlas_set, used)
        if target is not None:
            out.append(target)
            used.add(target)
    return out


def _clean_display_labels(labels: list[str], n: int) -> list[str]:
    out: list[str] = []
    for i, lab in enumerate(labels[:n]):
        s = str(lab).strip()
        if not s or s.lower() == "nan":
            s = f"Neuron_{i}"
        elif re.fullmatch(r"\d+", s):
            s = f"ROI_{s}"
        out.append(s)
    out.extend(f"Neuron_{j}" for j in range(len(out), n))
    return out


def _atlas_indices(
    file_labels: Optional[list[str]],
    full_labels: Optional[list[str]],
) -> Optional[np.ndarray]:
    if file_labels is None or full_labels is None:
        return None
    indices = []
    for label in file_labels:
        try:
            indices.append(full_labels.index(label))
        except ValueError:
            pass
    return np.array(indices) if indices else None


def _subset_matrix(
    arr: np.ndarray, N: int,
    file_labels: Optional[list[str]], full_labels: Optional[list[str]],
) -> np.ndarray:
    if arr.shape[0] <= N and arr.shape[1] <= N:
        return arr
    idx = _atlas_indices(file_labels, full_labels)
    if idx is None or len(idx) != N:
        n_matched = 0 if idx is None else len(idx)
        raise ValueError(
            f"Cannot subset matrix {arr.shape} to {N} neurons (matched {n_matched})"
        )
    orig = arr.shape
    arr = arr[np.ix_(idx, idx)]
    _io_logger.info("subset_connectome", original=str(orig), new=str(arr.shape))
    return arr
def _load_mask(
    ds_key: Optional[str], N: int, f: h5py.File,
    file_labels: Optional[list[str]], full_labels: Optional[list[str]],
) -> np.ndarray:
    if ds_key is None:
        mask = np.ones((N, N), dtype=float)
        np.fill_diagonal(mask, 0.0)
        return mask
    if ds_key.endswith(".npy"):
        arr = np.load(str(_resolve_path(ds_key)))
        if arr.shape[0] > N or arr.shape[1] > N:
            arr = _subset_matrix(arr, N, file_labels, full_labels)
    else:
        if ds_key not in f:
            raise KeyError(f"mask dataset '{ds_key}' not found in file")
        arr = np.array(f[ds_key], dtype=float)
    if arr.shape != (N, N):
        raise ValueError(f"mask '{ds_key}' has shape {arr.shape}, expected {(N, N)}")
    return arr


def _load_sign_matrix(
    sign_ds: Optional[str], N: int,
    file_labels: Optional[list[str]], full_labels: Optional[list[str]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if sign_ds is None:
        return None
    path = _resolve_path(sign_ds)
    if not path.exists():
        _io_logger.warning("sign_file_missing", path=str(sign_ds))
        return None
    sign = np.load(str(path)).astype(np.float32)
    if sign.shape == (N, N):
        return torch.tensor(sign, dtype=torch.float32, device=device)
    if sign.shape[0] > N:
        idx = _atlas_indices(file_labels, full_labels)
        if idx is not None and len(idx) == N:
            return torch.tensor(sign[np.ix_(idx, idx)], dtype=torch.float32, device=device)
        n_matched = 0 if idx is None else len(idx)
        _io_logger.warning("sign_subset_failed", matched=n_matched, total=N)
        return None
    _io_logger.warning("sign_shape_mismatch", shape=str(sign.shape), expected=f"({N},{N})")
    return None


def _load_neuron_labels(
    f: h5py.File, N0: int, N: int, keep_mask: np.ndarray,
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    ds = "gcamp/neuron_labels"
    if ds not in f:
        return None, None
    raw = f[ds][:]
    file_labels = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
    if len(file_labels) == N0 and not np.all(keep_mask):
        file_labels = [lab for lab, keep in zip(file_labels, keep_mask) if keep]
    full_labels = None
    names_path = _SRC_DIR / "data/used/masks+motor neurons/neuron_names.npy"
    if names_path.exists():
        full_labels = [str(n) for n in np.load(str(names_path))]
    if full_labels is not None and len(file_labels) != N:
        recovered = _recover_labels_to_atlas(file_labels, full_labels)
        if len(recovered) == N:
            file_labels = recovered
    return file_labels, full_labels


def _map_motor_neurons(
    cfg: Stage2PTConfig, file_labels: Optional[list[str]], N: int,
) -> Optional[list[int]]:
    if not cfg.motor_neurons:
        return None
    if all(isinstance(m, (int, np.integer)) for m in cfg.motor_neurons):
        idx = sorted({int(m) for m in cfg.motor_neurons if 0 <= int(m) < N})
        return idx or None
    if file_labels is None:
        return None
    labels_norm = [l.strip().upper() for l in file_labels[:N]]
    idx = [
        labels_norm.index(str(m).strip().upper())
        for m in cfg.motor_neurons
        if str(m).strip().upper() in labels_norm
    ]
    if idx:
        _io_logger.info("motor_neurons_mapped", mapped=len(idx), total=len(cfg.motor_neurons))
        return idx
    _io_logger.warning("motor_neurons_none_found")
    return None
def _load_stage1(
    f: h5py.File, cfg: Stage2PTConfig, device: torch.device,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor,
           Optional[torch.Tensor], np.ndarray, int, int, int]:
    if cfg.stage1_u_dataset not in f:
        raise KeyError(f"Stage 1 u dataset '{cfg.stage1_u_dataset}' not found")
    u = _ensure_TN(np.array(f[cfg.stage1_u_dataset], dtype=float))
    T, N0 = u.shape

    u_var = None
    var_key = getattr(cfg, "stage1_u_var_dataset", None)
    if var_key and var_key in f:
        u_var = _ensure_TN(np.array(f[var_key], dtype=float))
        if u_var.shape != (T, N0):
            raise ValueError(f"Stage 1 u_var shape {u_var.shape}, expected {(T, N0)}")

    keep = np.any(np.isfinite(u), axis=0)
    if not np.all(keep):
        _io_logger.info("dropping_nan_neurons", dropped=int((~keep).sum()), total=int(N0))
        u = u[:, keep]
        if u_var is not None:
            u_var = u_var[:, keep]
    T, N = u.shape

    if cfg.stage1_params_group not in f:
        raise KeyError(f"Stage 1 params group '{cfg.stage1_params_group}' not found")
    g = f[cfg.stage1_params_group]

    sigma_u = None
    if "sigma_u" in g:
        s = np.array(g["sigma_u"], dtype=float).ravel()
        if s.shape[0] != N0:
            raise ValueError("sigma_u must have length N0")
        s = s[keep] if not np.all(keep) else s
        sigma_u = torch.tensor(s, dtype=torch.float32, device=device)
    if sigma_u is None:
        sigma_u = torch.full((N,), cfg.sigma_u_default, dtype=torch.float32, device=device)

    rho = None
    if "rho" in g:
        r = np.array(g["rho"], dtype=float).ravel()
        if r.shape[0] != N0:
            raise ValueError("rho must have length N0")
        r = r[keep] if not np.all(keep) else r
        if r.shape[0] != N:
            raise ValueError("rho length mismatch after filtering")
        rho = torch.tensor(r, dtype=torch.float32, device=device)

    u_t = torch.tensor(u, dtype=torch.float32, device=device)
    uv_t = None if u_var is None else torch.tensor(u_var, dtype=torch.float32, device=device)
    return u_t, uv_t, sigma_u, rho, keep, T, N0, N


def _resolve_dt(f: h5py.File, cfg: Stage2PTConfig) -> float:
    if cfg.dt is not None and cfg.dt > 0:
        return float(cfg.dt)
    g = f.get(cfg.stage1_params_group)
    for container in (g, f):
        if container is not None and "sample_rate_hz" in container.attrs:
            sr = float(container.attrs["sample_rate_hz"])
            if sr > 0:
                return 1.0 / sr
    return 1.0


def _load_stimulus(
    f: h5py.File, cfg: Stage2PTConfig, T: int, N0: int,
    keep_mask: np.ndarray, device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[str], int]:
    ds = cfg.stim_dataset
    if ds is None:
        candidates = [
            "stimulus/regressor", "stim/regressor",
            "stimulus/external_drive", "external_drive", "regressor",
        ]
        ds = next((c for c in candidates if c in f), None)
    if ds is None or ds not in f:
        return None, None, 0

    stim = np.array(f[ds], dtype=float)
    if stim.ndim == 1:
        stim = stim[:, None]
    elif stim.ndim == 2 and stim.shape[0] < stim.shape[1] and stim.shape[1] == T:
        stim = stim.T
    if stim.shape[0] != T:
        raise ValueError(f"Stimulus T={stim.shape[0]}, expected {T}")
    if stim.ndim == 2 and stim.shape[1] == N0 and not np.all(keep_mask):
        stim = stim[:, keep_mask]
    if not np.all(np.isfinite(stim)):
        n_bad = int(np.sum(~np.isfinite(stim)))
        _io_logger.warning("stimulus_nan_values", n_bad=n_bad)
        stim = np.nan_to_num(stim, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(stim, dtype=torch.float32, device=device), ds, int(stim.shape[1])


def _load_gating(
    f: h5py.File, cfg: Stage2PTConfig, T: int, N: int, device: torch.device,
) -> torch.Tensor:
    if cfg.silencing_dataset is None:
        return torch.ones((T, N), dtype=torch.float32, device=device)
    if cfg.silencing_dataset not in f:
        raise KeyError(f"silencing dataset '{cfg.silencing_dataset}' not found")
    g = np.array(f[cfg.silencing_dataset], dtype=float)
    if g.ndim == 1 and g.shape[0] == N:
        g = np.tile(g.reshape(1, N), (T, 1))
    elif not (g.ndim == 2 and g.shape == (T, N)):
        raise ValueError(f"gating shape {g.shape}, expected ({N},) or ({T},{N})")
    return torch.tensor(g, dtype=torch.float32, device=device)


def _load_behaviour(
    f: h5py.File, cfg: Stage2PTConfig, T: int, N: int,
    device: torch.device, file_labels: Optional[list[str]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"b": None, "b_mask": None, "L_b": 0}
    behavior_ds = cfg.behavior_dataset
    if behavior_ds is None:
        behavior_ds = "behaviour/eigenworms_calc_6"

    if behavior_ds not in f:
        raise KeyError(
            f"behaviour dataset '{behavior_ds}' not found. "
            "Stage 2 now requires precomputed eigenworm amplitudes (e.g. /behaviour/eigenworms_calc_6)."
        )
    b_np = _ensure_TN(np.array(f[behavior_ds], dtype=float))
    if b_np.shape[0] != T:
        raise ValueError(f"behaviour T={b_np.shape[0]}, expected {T}")
    result["behavior_dataset"] = behavior_ds

    result["b_mask"] = torch.tensor(
        np.isfinite(b_np).astype(float), dtype=torch.float32, device=device,
    )
    b_np = np.nan_to_num(b_np, nan=0.0)
    result["b"] = torch.tensor(b_np, dtype=torch.float32, device=device)
    result["L_b"] = int(b_np.shape[1])

    motor_idx = _map_motor_neurons(cfg, file_labels, N)
    if motor_idx is not None:
        result["motor_neurons"] = motor_idx
        cfg.motor_neurons = tuple(motor_idx)
    return result

def load_data_pt(cfg: Stage2PTConfig) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    device = torch.device(cfg.device)

    with h5py.File(cfg.h5_path, "r") as f:
        u, u_var, sigma_u, rho, keep, T, N0, N = _load_stage1(f, cfg, device)
        data["u_stage1"] = u
        data["u_var_stage1"] = u_var
        data["sigma_u"] = sigma_u
        data["rho_stage1"] = rho

        stim, stim_ds, d_ell = _load_stimulus(f, cfg, T, N0, keep, device)
        data["stim"] = stim
        data["stim_dataset"] = stim_ds
        data["d_ell"] = d_ell

        data["dt"] = _resolve_dt(f, cfg)

        file_labels, full_labels = _load_neuron_labels(f, N0, N, keep)
        data["neuron_labels"] = (
            _clean_display_labels(list(file_labels), N)
            if file_labels is not None and len(file_labels) == N
            else [f"Neuron_{i}" for i in range(N)]
        )

        mask_args = (N, f, file_labels, full_labels)
        for key, ds in [("T_e", cfg.T_e_dataset), ("T_sv", cfg.T_sv_dataset),
                        ("T_dcv", cfg.T_dcv_dataset)]:
            data[key] = torch.tensor(
                _load_mask(ds, *mask_args), dtype=torch.float32, device=device,
            )

        # Load neurotransmitter sign matrix (model builds reversals from it)
        sign_t = _load_sign_matrix(
            getattr(cfg, "neurotransmitter_sign_dataset", None), N,
            file_labels, full_labels, device,
        )
        if sign_t is not None:
            data["sign_t"] = sign_t
            data["T_sv"] = data["T_sv"].abs()
            out_mask = data["T_sv"] > 0
            exc_count = ((sign_t > 0) & out_mask).sum(dim=1).float()
            inh_count = ((sign_t < 0) & out_mask).sum(dim=1).float()
            n_exc = int((exc_count > inh_count).sum().item())
            n_inh = int((inh_count > exc_count).sum().item())
            _io_logger.info("nt_sign_loaded", excitatory=n_exc, inhibitory=n_inh)

        data["gating"] = _load_gating(f, cfg, T, N, device)
        data.update(_load_behaviour(f, cfg, T, N, device, file_labels))

    return data


def save_results_pt(
    cfg: Stage2PTConfig,
    u_mean: torch.Tensor,
    params: Dict[str, torch.Tensor],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    with h5py.File(cfg.h5_path, "a") as f:
        if cfg.out_u_mean is not None:
            if cfg.out_u_mean in f:
                del f[cfg.out_u_mean]
            f.create_dataset(cfg.out_u_mean, data=u_mean.detach().cpu().numpy())

        grp = f.require_group(cfg.out_params)
        for k in list(grp.keys()):
            del grp[k]
        for k in list(grp.attrs.keys()):
            del grp.attrs[k]

        for key, val in params.items():
            if key == "b_pred":
                continue
            arr = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.array(val)
            if arr.ndim == 0:
                grp.attrs[key] = float(arr)
            else:
                grp.create_dataset(key, data=arr)

        if diagnostics is not None:
            diag_path = "stage2_pt/diagnostics"
            if diag_path in f:
                del f[diag_path]
            gdiag = f.require_group(diag_path)
            for k, v in diagnostics.items():
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                if isinstance(v, (float, int, np.floating, np.integer)):
                    gdiag.attrs[k] = float(v)
                    continue
                arr = np.asarray(v)
                if arr.ndim == 0:
                    gdiag.attrs[k] = float(arr)
                else:
                    gdiag.create_dataset(k, data=arr)
