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
    dt: Optional[float] = None
    T_e_dataset: Optional[str] = str(_DATA_ROOT / "T_e.npy")
    T_sv_dataset: Optional[str] = str(_DATA_ROOT / "T_sv.npy")
    T_dcv_dataset: Optional[str] = str(_DATA_ROOT / "T_dcv.npy")
    neurotransmitter_sign_dataset: Optional[str] = str(_DATA_ROOT / "neurotransmitter_sign.npy")


@dataclass
class DynamicsConfig:
    dynamics_scale: float = 1.0

    # ── Leak / tonic drive ──────────────────────────────────────────
    learn_lambda_u: bool = True
    learn_I0: bool = True

    # ── Gap junctions ──────────────────────────────────────────────
    G_init: float = 0.1
    G_max: Optional[float] = 2.0
    edge_specific_G: bool = False

    # ── SV synapses ────────────────────────────────────────────────
    # a_sv: set to a_sv_init per rank, then scaled by init_network_scale
    # tau_sv: fixed at tau_sv_init (log-spaced temporal basis)
    # W_sv: per-edge weights, init to W_sv_init (softplus domain)
    # E_sv: per-neuron reversal from NT sign × data quantiles
    r_sv: int = 5
    tau_sv_init: Tuple[float, ...] = (0.5, 0.85, 1.5, 2.5, 5.0)
    a_sv_init: Tuple[float, ...] = (0.25, 0.18, 0.12, 0.08, 0.05)
    a_sv_max: Optional[float] = 2.0
    fix_tau_sv: bool = True
    W_sv_init: float = 1.0
    learn_W_sv: bool = False
    E_sv_exc_init: float = 0.1
    E_sv_inh_init: float = -0.1

    # ── DCV synapses ───────────────────────────────────────────────
    # a_dcv: set to a_dcv_init per rank, then scaled by init_network_scale
    # tau_dcv: fixed at tau_dcv_init (slower temporal basis than SV)
    # W_dcv: per-edge weights, init to W_dcv_init (softplus domain)
    # E_dcv: set to median of I₀ baseline
    r_dcv: int = 5
    tau_dcv_init: Tuple[float, ...] = (2.0, 6.0, 10.0, 15.0, 20.0)
    a_dcv_init: Tuple[float, ...] = (0.08, 0.06, 0.045, 0.03, 0.02)
    a_dcv_max: Optional[float] = 2.0
    fix_tau_dcv: bool = True
    W_dcv_init: float = 1.0
    learn_W_dcv: bool = False
    E_dcv_init: float = 0.0

    # ── Reversal potentials ────────────────────────────────────────
    # E_sv_exc/inh: data quantiles (q_hi / q_lo) per NT sign
    # E_dcv: median of OLS-derived I₀ baseline
    learn_reversals: bool = False
    per_neuron_reversals: bool = False
    edge_specific_reversals: bool = False


    # ── Ridge-CV for per-neuron kernel amplitudes ──────────────────
    alpha_per_neuron: bool = True
    alpha_cv_every: int = 5
    alpha_cv_n_folds: int = 5
    alpha_cv_log_lambda_min: float = -6.0
    alpha_cv_log_lambda_max: float = 6.0
    alpha_cv_n_grid: int = 60

    # ── Safety clamps ──────────────────────────────────────────────
    u_next_clip_min: Optional[float] = -6
    u_next_clip_max: Optional[float] = 6

    # ── Physical bounds (reparameterization constraints) ───────────
    lambda_u_lo: float = 0.0
    lambda_u_hi: float = 0.9999
    G_lo: float = 0.0
    a_lo: float = 0.0
    tau_lo: float = 1e-4
    W_lo: float = 0.0


@dataclass
class StimulusConfig:
    stim_dataset: Optional[str] = None
    stim_diagonal_only: bool = True
    ridge_b: float = 0.1
    b_min:              Optional[float] = None
    b_max:              Optional[float] = None


@dataclass
class BehaviorConfig:
    behavior_dataset: Optional[str] = "behaviour/eigenworms_calc_6"
    motor_neurons: Optional[Tuple[str | int, ...]] = field(default_factory=_load_motor_neurons)
    behavior_lag_steps: int = 8
    behavior_weight: float = 0.1
    train_behavior_ridge_folds: int = 5
    train_behavior_ridge_log_lambda_min: float = -3.0
    train_behavior_ridge_log_lambda_max: float = 10.0
    train_behavior_ridge_n_grid: int = 50


@dataclass
class TrainConfig:
    num_epochs: int = 40
    learning_rate: float = 0.001
    device: str = "cuda"
    grad_clip_norm: float = 1.0
    dynamics_l2:    float = 0.0
    dynamics_objective: str = "one_step"

    rollout_steps: int = 5
    rollout_weight: float = 0.0
    rollout_starts: int = 8
    warmstart_rollout: bool = False

    neuron_dropout_frac: float = 0.0
    interaction_l2: float = 0.0
    ridge_W_sv: float = 0.0
    ridge_W_dcv: float = 0.0
    sigma_u_default:     float = 1.0
    use_u_var_weighting: bool  = False
    u_var_scale:         float = 5.0
    u_var_floor:         float = 1e-6


@dataclass
class EvalConfig:
    eval_loo_subset_size: int = 5
    eval_loo_subset_mode: str = "best_onestep"
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
    # ── Mode switch ────────────────────────────────────────────────
    multi_worm: bool = False
    h5_paths: Tuple[str, ...] = ()          # one H5 per worm; ignored when multi_worm=False

    # ── Data alignment ─────────────────────────────────────────────
    common_dt: float = 0.6                  # resample every worm to this dt (seconds)

    # ── Loss weighting ─────────────────────────────────────────────
    #   "equal"       — 1/W per worm  (recommended)
    #   "by_observed" — weight ∝ N_w^obs / Σ N^obs
    worm_weight_mode: str = "equal"

    # ── Parameter sharing ──────────────────────────────────────────
    #   Shared across worms:  G (or per-worm, see below), W_sv, W_dcv,
    #                         a_sv, a_dcv, tau_sv, tau_dcv, E_sv, E_dcv,
    #                         behaviour decoder
    #   Per-worm:             lambda_u, I0, b (stimulus), u_unobs
    per_worm_G: bool = False                # per-worm scalar G vs one shared scalar
    G_consistency_weight: float = 0.01      # λ_gc Σ_w (G_w − Ḡ)²  (only when per_worm_G)

    # ── MAP trajectory inference for unobserved neurons ────────────
    #   Two-level alternating optimisation (EPFL-style):
    #     Step 1 — fix θ,ψ, update u_U    (trajectory inference)
    #     Step 2 — fix u_U, update θ,ψ    (model learning)
    infer_unobserved: bool = True
    u_unobs_lr: float = 0.01               # Adam LR for trajectory free-variables
    u_unobs_inner_steps: int = 10           # gradient steps on u_U per outer epoch
    u_unobs_smoothness: float = 0.01        # temporal smoothness prior weight
    sigma_u_unobs_scale: float = 2.0        # inflate σ_u for unobserved neurons

    # ── Atlas construction ─────────────────────────────────────────
    atlas_min_worm_count: int = 0           # 0 = full 302 atlas; >0 = neuron must appear in ≥ K worms
    require_stage1: bool = True             # True = skip files without stage1; False = z-score raw traces

    # ── Temporal & cross-worm evaluation ───────────────────────────
    val_frac: float = 0.2                   # fraction of time held out per worm
    leave_worm_out_eval: bool = True        # train on W−1, eval on held-out worm


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
