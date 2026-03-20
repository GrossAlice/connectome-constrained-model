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
    learn_lambda_u: bool = True     # init: OLS on AR(1) u(t+1)=(1−λ)u(t)+λI₀
    learn_I0: bool = True           # init: OLS residual (ȳ−(1−λ)ẋ)/λ

    edge_specific_G: bool = False   # False→scalar G; True→per-edge (N,N)
    G_init_mode: str = "uniform"    # "uniform"|"log_counts"|"sqrt_counts" (edge-specific only)

    tau_sv_init: Tuple[float, ...] = (0.5, 0.85, 1.5, 2.5, 5.0)   # time constants (s); fixed when fix_tau_sv
    a_sv_init: Tuple[float, ...] = (2.5, 1.8, 1.2, 0.8, 0.5)      # per-rank amplitudes; OLS-rescaled by init_network_scale
    fix_tau_sv: bool = True         # False → tau_sv becomes trainable
    learn_W_sv: bool = False        # init: 1.0 (uniform) or from synapse counts
    W_sv_init_mode: str = "uniform" # "uniform"|"log_counts"|"sqrt_counts"
    W_sv_normalize: bool = False    # row-normalise so (T_sv·W_sv).sum(1)=1

    tau_dcv_init: Tuple[float, ...] = (2.0, 6.0, 10.0, 15.0, 20.0) # slower basis than SV (s)
    a_dcv_init: Tuple[float, ...] = (0.8, 0.6, 0.45, 0.3, 0.2)    # OLS-rescaled by init_network_scale
    fix_tau_dcv: bool = True
    learn_W_dcv: bool = False       # init: 1.0, same as W_sv

    # init: E_sv sign from NT type, magnitudes = 1st/99th pctile of u; E_dcv = median(I₀)
    learn_reversals: bool = False           # True → E_sv/E_dcv trainable
    reversal_mode: str = "scalar"       # "scalar"|"per_neuron" (N,)|"per_edge" (N,N)

    network_init_mode: str = "ols"  # "ols" (global) | "per_neuron_ridge" (per-neuron ridge, median β)

    u_next_clip_min: Optional[float] = -10
    u_next_clip_max: Optional[float] = 10

@dataclass
class BehaviorConfig:
    behavior_lag_steps: int = 8
    behavior_weight: float = 0.1
    behavior_decoder_type: str = "linear"     # "linear" | "mlp"
    behavior_decoder_hidden: int = 32
    behavior_decoder_dropout: float = 0.1

@dataclass
class StimulusConfig:
    stim_dataset: Optional[str] = None
    stim_diagonal_only: bool = True
    ridge_b: float = 0.1
    b_min: Optional[float] = None
    b_max: Optional[float] = None

@dataclass
class TrainConfig:
    num_epochs: int = 100
    learning_rate: float = 0.001
    device: str = "cuda"
    grad_clip_norm: float = 1.0
    dynamics_l2:    float = 0.0
    dynamics_objective: str = "one_step"

    rollout_steps: int = 0
    rollout_weight: float = 0.1
    rollout_starts: int = 8
    warmstart_rollout: bool = False

    interaction_l2: float = 0.0
    ridge_W_sv: float = 0.0
    ridge_W_dcv: float = 0.0
    synaptic_lr_multiplier: float = 10.0
    sigma_u_default:     float = 1.0
    use_u_var_weighting: bool  = False
    u_var_scale:         float = 5.0
    u_var_floor:         float = 1e-6

    alpha_per_neuron: bool = True
    alpha_cv_every: int = 0         # re-solve every N epochs (0=disabled)
    alpha_cv_n_folds: int = 5
    alpha_cv_log_min: float = -6.0  # log₁₀(λ) grid bounds
    alpha_cv_log_max: float = 10.0
    alpha_cv_n_grid: int = 80

    train_behavior_ridge_folds: int = 5
    train_behavior_ridge_log_lambda_min: float = -3.0
    train_behavior_ridge_log_lambda_max: float = 10.0
    train_behavior_ridge_n_grid: int = 50


@dataclass
class EvalConfig:
    eval_loo_subset_size: int = 50
    eval_loo_subset_mode: str = "motor_best_onestep" # "motor_best_onestep"
    eval_loo_subset_seed: int = 0
    eval_loo_subset_neurons: Optional[Tuple[int, ...]] = None


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
    # Shared params: G, W_sv, W_dcv, a, tau, E, behaviour decoder
    # Per-worm params: lambda_u, I0, b (stimulus), u_unobs
    multi_worm: bool = False
    h5_paths: Tuple[str, ...] = ()
    common_dt: float = 0.6
    worm_weight_mode: str = "equal"             # "equal" (1/W) | "by_observed" (weight ∝ N_obs)
    per_worm_G: bool = False                    # per-worm G scalar vs one shared
    G_consistency_weight: float = 0.01          # λ_gc Σ(G_w−Ģ)²; only when per_worm_G
    infer_unobserved: bool = True               # MAP inference for unobserved neuron trajectories
    u_unobs_lr: float = 0.01                    # Adam LR for u_U free-variables
    u_unobs_inner_steps: int = 10               # gradient steps on u_U per outer epoch
    u_unobs_smoothness: float = 0.01            # temporal smoothness prior on u_U
    sigma_u_unobs_scale: float = 2.0            # inflate σ_u for unobserved neurons
    atlas_min_worm_count: int = 2               # 0=full 302; >0=neuron in ≥K worms
    require_stage1: bool = True                 # skip files without stage1 (False=z-score raw)
    val_frac: float = 0.2                       # temporal hold-out fraction per worm
    leave_worm_out_eval: bool = True            # train on W−1, eval on held-out worm
    loo_subset_size: int = 20                   # max neurons per worm for LOO eval
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
