"""Stage 1 configuration — EM deconvolution of calcium traces."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Optional, Tuple

import h5py


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Helpers                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _rho_from_tau(tau_sec: float, sr_hz: float) -> float:
    """AR(1) decay coefficient rho = exp(-dt / tau)."""
    return float(exp(-1.0 / (sr_hz * tau_sec)))


def _lambda_from_tau(tau_sec: float, sr_hz: float) -> float:
    """Calcium decay rate lambda_c = 1 - exp(-dt / tau)."""
    return float(1.0 - exp(-1.0 / (sr_hz * tau_sec)))


def _rho_clip(tau_clip: Tuple[float, float], sr_hz: float) -> Tuple[float, float]:
    lo, hi = tau_clip
    return (_rho_from_tau(lo, sr_hz), _rho_from_tau(hi, sr_hz))


def _lambda_clip(tau_clip: Tuple[float, float], sr_hz: float) -> Tuple[float, float]:
    lo, hi = tau_clip
    # Note: larger tau => smaller lambda, so bounds swap
    return (_lambda_from_tau(hi, sr_hz), _lambda_from_tau(lo, sr_hz))


def _infer_dataset_type(h5_path: str) -> str:
    """Peek inside the H5 to decide 'behaviour' / 'optogenetics' / 'unknown'."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "optogenetics" in f:
                return "optogenetics"
            if "behavior" in f or "behaviour" in f:
                return "behaviour"
    except OSError:
        pass
    return "unknown"


# Per-dataset overrides (anything not listed inherits the dataclass default).
_DATASET_OVERRIDES: dict[str, dict] = {
    "behaviour": {
        "calcium_indicator": "gcamp7f",
        "sample_rate_hz":    100.0 / 60.0,
        "tau_c_init_sec":    5.0,
    },
    "optogenetics": {
        "calcium_indicator": "gcamp6s",
        "sample_rate_hz":    2.0,
        "tau_c_init_sec":    2.0,
    },
}
_DATASET_OVERRIDES["unknown"] = dict(_DATASET_OVERRIDES["behaviour"])


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Config dataclass                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class Stage1Config:
    """All parameters for Stage 1 EM calcium deconvolution.

    Dataset-dependent fields (sample_rate_hz, calcium_indicator, tau_c_init_sec)
    are auto-resolved from the H5 file in __post_init__ unless set explicitly.
    """

    # ── 1. DATA I/O ──────────────────────────────────────────────────────
    h5_path: str
    trace_dataset: str = "gcamp/trace_array_original"

    out_u_mean:  str = "stage1/u_mean"
    out_u_var:   str = "stage1/u_var"
    out_c_mean:  str = "stage1/c_mean"
    out_c_var:   str = "stage1/c_var"
    out_params:  str = "stage1/params"

    # ── 2. PREPROCESSING ────────────────────────────────────────────────
    center_traces:  bool  = False
    use_dff:        bool  = True
    f0_method:      str   = "quantile"
    f0_quantile:    float = 0.2
    f0_window_sec:  float = 120.0
    f0_eps:         float = 1e-3

    # ── 3. DATASET & INDICATOR ──────────────────────────────────────────
    # Auto-inferred from h5_path in __post_init__; set explicitly to override.
    dataset_type:      Optional[str]   = None
    calcium_indicator: Optional[str]   = None
    sample_rate_hz:    Optional[float] = None

    # ── 4. TIME CONSTANTS & INITIAL VALUES ──────────────────────────────
    tau_u_init_sec:     float           = 1.0
    tau_c_init_sec:     Optional[float] = None     # dataset-dep. (5s behav., 2s opto.)
    sigma_u_scale_init: float           = 0.7
    sigma_c_init:       float           = 5e-3

    # ── 5. OBSERVATION MODEL ────────────────────────────────────────────
    fix_alpha:          bool  = True
    alpha_value:        float = 1.0
    sigma_y_floor:      float = 0.06
    sigma_y_floor_frac: float = 0.85
    alpha_floor:        float = 1e-6

    # ── 6. PARAMETER SHARING (across neurons) ───────────────────────────
    share_rho:      bool = False
    share_lambda_c: bool = False
    share_sigma_c:  bool = False

    # ── 7. EM ALGORITHM ─────────────────────────────────────────────────
    em_max_iters:  int   = 200
    em_tol_rel_ll: float = 1e-4

    # ── 8. PARAMETER BOUNDS ─────────────────────────────────────────────
    tau_u_clip_sec: Tuple[float, float] = (0.5, 8.0)
    tau_c_clip_sec: Tuple[float, float] = (0.5, 20.0)
    rho_clip:       Optional[Tuple[float, float]] = None    # derived in __post_init__
    lambda_clip:    Optional[Tuple[float, float]] = None    # derived in __post_init__
    eps_var: float = 1e-12

    # ─────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        dtype    = self.dataset_type or _infer_dataset_type(self.h5_path)
        defaults = _DATASET_OVERRIDES.get(dtype, _DATASET_OVERRIDES["unknown"])

        def _fill(attr: str, value: object) -> None:
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, value)

        _fill("dataset_type",      dtype)
        _fill("calcium_indicator",  defaults["calcium_indicator"])
        _fill("sample_rate_hz",     defaults["sample_rate_hz"])
        _fill("tau_c_init_sec",     defaults["tau_c_init_sec"])

        # Derive clip bounds from time-constant limits + sample rate
        sr = self.sample_rate_hz
        _fill("rho_clip",    _rho_clip(self.tau_u_clip_sec, sr))
        _fill("lambda_clip", _lambda_clip(self.tau_c_clip_sec, sr))
