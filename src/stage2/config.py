from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, get_type_hints


_DATA_ROOT = (Path(__file__).resolve().parent / ".." / "data" / "used" / "masks+motor neurons").resolve()


def _load_motor_neurons() -> Tuple[str, ...]:
    p = _DATA_ROOT / "motor_neurons_with_control.txt"
    return tuple(l.strip() for l in p.read_text().splitlines()
                 if l.strip() and not l.startswith("#")) if p.exists() else ()


@dataclass
class DataConfig:
    h5_path: str
    stage1_u_dataset: str = "stage1/u_mean"
    stage1_u_var_dataset: Optional[str] = "stage1/u_var"
    stage1_params_group: str = "stage1/params"
    silencing_dataset: Optional[str] = None
    dt: Optional[float] = 0.6
    T_e_dataset: Optional[str] = str(_DATA_ROOT / "T_e.npy")
    T_sv_dataset: Optional[str] = str(_DATA_ROOT / "T_sv.npy")
    T_dcv_dataset: Optional[str] = str(_DATA_ROOT / "T_dcv.npy")
    neurotransmitter_sign_dataset: Optional[str] = str(_DATA_ROOT / "neurotransmitter_sign.npy")

    behavior_dataset: Optional[str] = "behaviour/eigenworms_stephens"
    motor_neurons: Optional[Tuple[str | int, ...]] = field(default_factory=_load_motor_neurons)

@dataclass
class DynamicsConfig:
    learn_lambda_u: bool = True
    learn_I0: bool = True

    edge_specific_G: bool = True      # False→scalar; True→per-edge (N,N)
    G_init_mode: str = "uniform"      # "uniform"|"log_counts"|"sqrt_counts"

    tau_sv_init: Tuple[float, ...] = (0.5, 0.85, 1.5, 2.5, 5.0)
    a_sv_init: Tuple[float, ...] = (2.5, 1.8, 1.2, 0.8, 0.5)
    fix_tau_sv: bool = True
    fix_a_sv: bool = False
    learn_W_sv: bool = False
    W_sv_init_mode: str = "uniform"   # "uniform"|"log_counts"|"sqrt_counts"
    W_sv_normalize: bool = False

    tau_dcv_init: Tuple[float, ...] = (2.0, 6.0, 10.0, 15.0, 20.0)
    a_dcv_init: Tuple[float, ...] = (0.8, 0.6, 0.45, 0.3, 0.2)
    fix_tau_dcv: bool = True
    fix_a_dcv: bool = False
    learn_W_dcv: bool = False
    per_neuron_amplitudes: bool = True  # (N, r) per-neuron vs shared (r,)

    learn_reversals: bool = False
    reversal_mode: str = "per_neuron"   # "scalar"|"per_neuron"|"per_edge"

    network_init_mode: str = "ols"      # "ols" (global) | "per_neuron_ridge" (per neuron)

    u_next_clip_min: Optional[float] = -10
    u_next_clip_max: Optional[float] = 10

    # Per-neuron process noise: learnable log_sigma_u per neuron.
    # When enabled, the model predicts a distribution u_{t+1} ~ N(mu, sigma^2)
    # and the dynamics loss becomes Gaussian NLL instead of weighted MSE.
    learn_noise: bool = False
    noise_floor: float = 1e-3          # minimum sigma (softplus + floor)
    noise_reg: float = 0.0             # L2 penalty on log_sigma_u (prevents collapse)
    noise_mode: str = "homoscedastic"  # "homoscedastic" | "heteroscedastic"
    # homoscedastic:   sigma_i = softplus(b_i) + floor  (constant per neuron)
    # heteroscedastic: sigma_{t,i} = softplus(w_i * u_{t,i} + b_i) + floor
    #                  (diagonal linear map — noise scales with own activation)
    noise_sigma_source: str = "all"     # "all" | "rollout"
    # "all"     — σ receives gradients from one-step + rollout + LOO NLL
    # "rollout" — σ is detached from the one-step NLL so only rollout
    #              and LOO NLL shape the noise magnitude (wider CIs)

@dataclass
class BehaviorConfig:
    behavior_lag_steps: int = 8
    behavior_weight: float = 0.1
    behavior_decoder_type: str = "linear"  # "linear"|"mlp"
    behavior_decoder_hidden: int = 32
    behavior_decoder_dropout: float = 0.1
    train_behavior_ridge_folds: int = 5
    train_behavior_ridge_log_lambda_min: float = -3.0
    train_behavior_ridge_log_lambda_max: float = 10.0
    train_behavior_ridge_n_grid: int = 50

    # Variance-preserving calibration of behaviour predictions.
    # Scale factor for the noise (std) ratio sd_true / sd_pred:
    #   1.0 = full rescaling (match data variance exactly)
    #   0.0 = no rescaling (only shift mean)
    #   >0  = interpolate: scale = 1 + noise_scale * (sd_true/sd_pred - 1)
    behavior_noise_scale: float = 0.0

    # AR-augmented behaviour decoder: prepend lagged eigenworm values to the
    # feature matrix so the ridge/MLP can exploit autocorrelation.
    #   0 = disabled (current default – purely neural features)
    #   2 = recommended (AR(2) captures the ~8 s body-wave oscillation)
    behavior_ar_lags: int = 0

@dataclass
class StimulusConfig:
    stim_dataset: Optional[str] = None
    stim_diagonal_only: bool = True
    ridge_b: float = 0.1
    b_min: Optional[float] = None
    b_max: Optional[float] = None
    # Learnable causal kernel: convolve delta-pulse stimuli into sustained drive
    stim_kernel_len: int = 0          # 0=off; >0 = kernel length in time-steps
    stim_kernel_tau_init: float = 2.0 # initial exponential decay (seconds)

@dataclass
class TrainConfig:
    num_epochs: int = 100
    learning_rate: float = 0.001
    device: str = "cuda"
    grad_clip_norm: float = 1
    dynamics_l2:    float = 0.0

    rollout_steps: int = 0
    rollout_weight: float = 1
    rollout_starts: int = 8
    warmstart_rollout: bool = True

    interaction_l2: float = 0.0
    ridge_W_sv: float = 0.0
    ridge_W_dcv: float = 0.0
    synaptic_lr_multiplier: float = 5.0
    sigma_u_default:     float = 1.0
    use_u_var_weighting: bool  = False
    u_var_scale:         float = 5.0
    u_var_floor:         float = 1e-6

    # Dynamics-CV: periodic joint ridge-CV re-solve of I0, G, a_sv, a_dcv
    # All parameters are solved in a single per-neuron regression so that
    # gap-junction and synaptic features compete on equal footing.
    dynamics_cv_every: int = 5       # 0=disabled
    dynamics_cv_n_folds: int = 5
    dynamics_cv_log_min: float = -2.0
    dynamics_cv_log_max: float = 6.0
    dynamics_cv_n_grid: int = 50
    dynamics_cv_blend: float = 0.5
    dynamics_cv_warmup: int = 3      # ramp blend over this many injections (1=no ramp)
    dynamics_cv_r2_gate: float = 0.01  # skip neuron if per-neuron fit R² < gate

    # Init-anchored regularization
    lambda_u_reg: float = 0.0
    I0_reg: float = 0.0
    G_reg: float = 0.0
    tau_reg: float = 0.0

    # Network strength floor: one-sided penalty that fires when
    # effective network drive (G_rms * a_sv_rms) drops below a
    # fraction of its initialized value.  0=disabled.
    network_strength_floor: float = 1.0    # penalty weight
    network_strength_target: float = 0.8   # fraction of init to maintain

    # LOO auxiliary loss
    loo_aux_weight: float = 0.0      # 0=disabled
    loo_aux_steps: int = 20
    loo_aux_neurons: int = 4
    loo_aux_starts: int = 1



@dataclass
class EvalConfig:
    eval_loo_subset_size: int = 20
    eval_loo_subset_mode: str = "named"  # "named"|"variance"|"best_onestep"|"random"
    eval_loo_subset_names: Tuple[str, ...] = ("AVAL", "AVAR")
    eval_loo_subset_seed: int = 0
    eval_loo_subset_neurons: Optional[Tuple[int, ...]] = None

    # CV-reg: per-neuron shrinkage α blending AR(1) ↔ full model
    #   û_reg = û_ar1 + α_i * (û_model − û_ar1),  α = 1/(1+reg)
    # α chosen per-neuron by temporal-block k-fold CV on one-step MSE.
    cv_reg_enabled: bool = False     # set False to skip CV-reg entirely
    cv_reg_n_folds: int = 5
    cv_reg_log_min: float = -3.0    # log10(reg) lower bound
    cv_reg_log_max: float = 5.0     # log10(reg) upper bound
    cv_reg_n_grid:  int = 50        # grid points (log-spaced)

    # Stochastic trajectory sampling: when the model has learnable noise,
    # sample this many trajectories per neuron during LOO evaluation.
    # 0 = disabled (deterministic only); 20 gives reasonable CI bands.
    n_sample_trajectories: int = 20

    # Full-brain stochastic free-run: sample K whole-brain trajectories
    # where *all* neurons evolve autonomously with learned process noise.
    # 0 = disabled (deterministic free-run only).
    n_freerun_samples: int = 20

    # Alpha-CV / Backbone-CV diagnostics (re-runs ridge-CV solvers at
    # plot time – can be slow; set False to skip the diagnostic figure)
    include_dynamics_cv_diagnostics: bool = True


@dataclass
class OutputConfig:
    plot_every: int = 10
    out_u_mean: Optional[str] = "stage2_pt/u_mean"
    out_params: str = "stage2_pt/params"
    make_posture_video: bool = True
    posture_video_fps: int = 15
    posture_video_dpi: int = 120
    posture_video_max_frames: int = 200
    posture_video_out: Optional[str] = None


@dataclass
class MultiWormConfig:
    # Shared: G, W_sv, W_dcv, a, tau, E, behaviour decoder
    # Per-worm: lambda_u, I0, b, u_unobs
    multi_worm: bool = False
    h5_paths: Tuple[str, ...] = ()
    common_dt: float = 0.6
    worm_weight_mode: str = "equal"        # "equal"|"by_observed"
    per_worm_G: bool = False
    G_consistency_weight: float = 0.01
    infer_unobserved: bool = True
    u_unobs_lr: float = 0.01
    u_unobs_inner_steps: int = 10
    u_unobs_smoothness: float = 0.01
    sigma_u_unobs_scale: float = 2.0
    atlas_min_worm_count: int = 2          # 0=full 302
    require_stage1: bool = True
    val_frac: float = 0.2
    loo_subset_size: int = 20



@dataclass
class Stage2PTConfig:
    data: DataConfig
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    multi: MultiWormConfig = field(default_factory=MultiWormConfig)

    def _owner(self, name):
        o = _FIELD_OWNER.get(name)
        return object.__getattribute__(self, o) if o else None

    def __getattr__(self, name):
        sub = self._owner(name)
        if sub is not None: return getattr(sub, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        sub = self._owner(name) if name not in type(self).__dataclass_fields__ else None
        if sub is not None: setattr(sub, name, value)
        else: object.__setattr__(self, name, value)


# field name → sub-config attribute name
_FIELD_OWNER = {
    f: sec
    for sec, cls in get_type_hints(Stage2PTConfig).items()
    if sec in Stage2PTConfig.__dataclass_fields__ and dataclasses.is_dataclass(cls)
    for f in cls.__dataclass_fields__
}


def make_config(h5_path: str, **kwargs) -> Stage2PTConfig:
    buckets: dict[str, dict] = {s: {} for s in {*_FIELD_OWNER.values()}}
    buckets["data"]["h5_path"] = h5_path
    for k, v in kwargs.items():
        if (s := _FIELD_OWNER.get(k)): buckets[s][k] = v
    bad = sorted(k for k in kwargs if k not in _FIELD_OWNER)
    if bad: warnings.warn(f"Unknown config keys (ignored): {bad}", stacklevel=2)
    subs = {s: get_type_hints(Stage2PTConfig)[s](**b) for s, b in buckets.items()}
    return Stage2PTConfig(**subs)
