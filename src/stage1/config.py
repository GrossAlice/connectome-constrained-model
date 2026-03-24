"""Stage 1 configuration — EM deconvolution of calcium traces."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Optional, Tuple

import h5py


# ── Helpers ─────────────────────────────────────────────────────────────────

def _rho(tau: float, sr: float) -> float:
    return float(exp(-1.0 / (sr * tau)))

def _lam(tau: float, sr: float) -> float:
    return float(1.0 - exp(-1.0 / (sr * tau)))


def _infer_dataset_type(h5_path: str) -> str:
    try:
        with h5py.File(h5_path, "r") as f:
            if "optogenetics" in f:  return "optogenetics"
            if "behavior" in f or "behaviour" in f:  return "behaviour"
    except OSError:
        pass
    return "unknown"


# (indicator, sample_rate_hz, tau_c_init_sec)
_PRESETS = {
    "behaviour":    ("gcamp7f", 100.0 / 60.0, 5.0),
    "optogenetics": ("gcamp6s", 2.0,          2.0),
}
_PRESETS["unknown"] = _PRESETS["behaviour"]


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage1Config:
    # Data I/O
    h5_path:        str
    trace_dataset:  str = "gcamp/trace_array_original"
    out_u_mean:     str = "stage1/u_mean"
    out_u_var:      str = "stage1/u_var"
    out_c_mean:     str = "stage1/c_mean"
    out_c_var:      str = "stage1/c_var"
    out_params:     str = "stage1/params"

    # Preprocessing
    center_traces:  bool  = False
    use_dff:        bool  = True
    f0_method:      str   = "quantile"
    f0_quantile:    float = 0.2
    f0_window_sec:  float = 120.0
    f0_eps:         float = 1e-3

    # Dataset & indicator (auto-inferred in __post_init__)
    dataset_type:      Optional[str]   = None
    calcium_indicator: Optional[str]   = None
    sample_rate_hz:    Optional[float] = None

    # Time constants & initial values
    tau_u_init_sec:     float           = 1.0
    tau_c_init_sec:     Optional[float] = None
    sigma_u_scale_init: float           = 0.7
    sigma_c_init:       float           = 5e-3

    # Observation model
    fix_alpha:          bool  = True
    alpha_value:        float = 1.0
    sigma_y_floor:      float = 0.06
    sigma_y_floor_frac: float = 0.85
    alpha_floor:        float = 1e-6

    # Parameter sharing (across neurons)
    share_rho:      bool = False
    share_lambda_c: bool = False
    share_sigma_c:  bool = False

    # EM algorithm
    em_max_iters:  int   = 200
    em_tol_rel_ll: float = 1e-4

    # Parameter bounds
    tau_u_clip_sec: Tuple[float, float]           = (0.3, 1.0)
    tau_c_clip_sec: Tuple[float, float]           = (1.5, 20.0)
    rho_clip:       Optional[Tuple[float, float]] = None   # derived
    lambda_clip:    Optional[Tuple[float, float]] = None   # derived
    eps_var:        float                          = 1e-12

    # Runtime / IO
    max_neurons:    Optional[int] = None
    overwrite:      bool          = False
    save_dir:       str           = "./stage1_plots"
    label_csv:      Optional[str] = None
    neuron_mask:    Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        dtype = self.dataset_type or _infer_dataset_type(self.h5_path)
        indicator, sr, tau_c = _PRESETS.get(dtype, _PRESETS["unknown"])

        for attr, val in [
            ("dataset_type",      dtype),
            ("calcium_indicator", indicator),
            ("sample_rate_hz",    sr),
            ("tau_c_init_sec",    tau_c),
        ]:
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, val)

        sr = self.sample_rate_hz
        if self.rho_clip is None:
            lo, hi = self.tau_u_clip_sec
            object.__setattr__(self, "rho_clip", (_rho(lo, sr), _rho(hi, sr)))
        if self.lambda_clip is None:
            lo, hi = self.tau_c_clip_sec
            # larger τ → smaller λ, so bounds swap
            object.__setattr__(self, "lambda_clip", (_lam(hi, sr), _lam(lo, sr)))
