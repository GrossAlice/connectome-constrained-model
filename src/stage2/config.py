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
    lambda_u_lo: float = 0.1           # floor on lambda_u (0 = unconstrained)
    lambda_u_hi: float = 0.9999       # ceiling on lambda_u
    linear_chemical_synapses: bool = False
    chemical_synapse_activation: str = "sigmoid"
    noise_corr_rank: int = 0          # 0=diagonal; >0 = low-rank Σ = diag + V V^T

    edge_specific_G: bool = True       # False→scalar; True→per-edge (N,N)
    G_init_mode: str = "corr_weighted"      # "uniform"|"log_counts"|"sqrt_counts"|"corr_weighted"
    W_init_mode: str = "corr_weighted"       # "uniform"|"corr_weighted" — scale W_sv/W_dcv by |corr(i,j)|

    tau_sv_init: Tuple[float, ...] = (0.5, 5)
    a_sv_init: Tuple[float, ...] = (2.0, 0.8)
    fix_tau_sv: bool = True
    fix_a_sv: bool = False
    learn_W_sv: bool = True

    tau_dcv_init: Tuple[float, ...] = (3.0, 5.0)
    a_dcv_init: Tuple[float, ...] = (0.8, 0.6)
    fix_tau_dcv: bool = True
    fix_a_dcv: bool = False
    learn_W_dcv: bool = True
    per_neuron_amplitudes: bool = True  # (N, r) per-neuron vs shared (r,)

    learn_reversals: bool = False
    reversal_mode: str = "per_neuron"   # "scalar"|"per_neuron"|"per_edge"

    u_next_clip_min: Optional[float] = -10
    u_next_clip_max: Optional[float] = 10
    learn_noise: bool = True
    noise_floor: float = 1e-3          # minimum sigma (softplus + floor)
    noise_reg: float = 0.0             # L2 penalty on log_sigma_u (prevents collapse)
    noise_mode: str = "heteroscedastic"  # "homoscedastic" | "heteroscedastic"

    lowrank_rank: int = 0              # 0=off; >0 captures non-connectome interactions
    graph_poly_order: int = 1          # 1=standard; 2-3 adds higher-order hops
    coupling_dropout: float = 0.0      # 0=off; >0 = dropout probability


    # ── Chemical synapse kernel mode ─────────────────────────────────
    chemical_synapse_mode: str = "iir"  # "iir" (exponential IIR) | "fir" (learnable FIR per-edge)
    fir_kernel_len: int = 5              # FIR kernel length in time-steps (only if mode="fir")
    fir_activation: str = "softplus"    # activation before FIR filter: "identity"|"sigmoid"|"softplus"
    fir_include_reversal: bool = False  # multiply by (E - u_i) driving force in FIR mode

    # ── Lag (FIR self + neighbor on raw u) ───────────────────────────
    lag_order: int = 5                 # 0=off; K>0 adds K self-lag terms
    lag_neighbor: bool = True          # True: also add connectome-sparse neighbor lags
    lag_connectome_mask: str = "all"   # "T_e"|"all" — which connectome to use for neighbor lags
    lag_neighbor_activation: str = "none"  # "none"|"sigmoid"|"softplus"|"tanh" — apply φ(u_j) before neighbor lag weighting
    lag_neighbor_per_type: bool = False  # True: separate lag weights per connectome type (T_e, T_sv, T_dcv)

@dataclass
class BehaviorConfig:
    behavior_lag_steps: int = 8
    behavior_weight: float = 0.0
    behavior_n_modes: int = 6           # 0=use all eigenworm modes from the HDF5
    behavior_weight_cap: float = 0.0   # >0 : L∞ penalty weight on model.b
    behavior_decoder_type: str = "mlp"  # "linear"|"mlp"
    behavior_decoder_hidden: int = 128
    behavior_decoder_n_layers: int = 2
    behavior_decoder_dropout: float = 0.1
    train_behavior_ridge_folds: int = 5
    train_behavior_ridge_log_lambda_min: float = -3.0
    train_behavior_ridge_log_lambda_max: float = 10.0
    train_behavior_ridge_n_grid: int = 50

@dataclass
class StimulusConfig:
    stim_dataset: Optional[str] = None
    stim_diagonal_only: bool = True
    ridge_b: float = 0.1
    b_min: Optional[float] = None
    b_max: Optional[float] = None
    stim_kernel_len: int = 0          # 0=off; >0 = kernel length in time-steps
    stim_kernel_tau_init: float = 2.0 # initial exponential decay (seconds)

@dataclass
class TrainConfig:
    num_epochs: int = 30
    learning_rate: float = 0.001
    device: str = "cuda"
    grad_clip_norm: float = 1
    dynamics_l2:    float = 0.0
    corr_reg_weight: float = 0.0       # penalise low-correlation edges (requires init_corr_reg_mask)
    seed: int = 42                  # global RNG seed for reproducibility (0=unseeded)

    rollout_steps: int = 30
    rollout_weight: float = 0.0
    rollout_starts: int = 8
    warmstart_rollout: bool = True

    input_noise_sigma: float = 0.0   # σ of additive Gaussian noise on u_t during training (0=off)

    rollout_curriculum: bool = False  # enable curriculum scheduling
    rollout_K_start: int = 5          # initial rollout horizon
    rollout_K_end: int = 30           # final rollout horizon
    rollout_curriculum_start_epoch: int = 0    # epoch to begin ramping
    rollout_curriculum_end_epoch: int = 100    # epoch to reach full horizon

    synaptic_lr_multiplier: float = 5.0
    sigma_u_default:     float = 1.0
    use_u_var_weighting: bool  = False
    u_var_scale:         float = 5.0
    u_var_floor:         float = 1e-6


    # Init-anchored regularization
    lambda_u_reg: float = 0.0
    I0_reg: float = 0.0
    tau_reg: float = 0.0
    network_strength_floor: float = 1.0   # weight of one-sided penalty keeping G*a_sv above target
    network_strength_target: float = 0.8  # fraction of init G_rms*a_sv_rms to maintain

    # LOO auxiliary loss
    loo_aux_weight: float = 0.0      # >0 enables differentiable LOO during training
    loo_aux_steps: int = 0
    loo_aux_neurons: int = 0
    loo_aux_starts: int = 8
    cv_folds: int = 3                # k-fold temporal CV (must be ≥ 2)
    parallel_folds: bool = True     # evaluate CV folds in parallel (CUDA streams)
    use_mixed_precision: bool = True  # bfloat16 autocast for forward + loss


@dataclass
class EvalConfig:
    eval_loo_subset_size: int = 30
    eval_loo_subset_mode: str = "variance"  # "named"|"variance"|"best_onestep"|"random"
    eval_loo_subset_names: Tuple[str, ...] = ("AVAL", "AVAR")
    eval_loo_subset_seed: int = 0
    eval_loo_subset_neurons: Optional[Tuple[int, ...]] = None
    eval_free_run_seed_steps: int = 16

    n_freerun_samples: int = 20
    eval_loo_window_size: int = 50
    eval_loo_warmup_steps: int = 40  # teacher-forced burn-in before each LOO window (0=cold)
    skip_cv_loo: bool = False        # skip LOO evaluation during CV (faster dev iterations)
    skip_free_run: bool = False      # skip free-run + decomposition during CV (faster sweeps)


@dataclass
class OutputConfig:
    plot_every: int = 10
    out_u_mean: Optional[str] = "stage2_pt/u_mean"
    out_params: str = "stage2_pt/params"
    skip_final_eval: bool = False      # skip generate_eval_loo_plots after CV
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
    u_unobs_connectome_init: bool = True   # use gap-junction graph to init u_unobs
    u_unobs_init_alpha: float = 1.0        # ridge regularisation for connectome init
    u_unobs_init_smooth_sigma: float = 0.0 # temporal smoothing of init (0 = off)
    u_unobs_warmup_epochs: int = 20        # pre-train u_unobs before main loop (0 = off)
    u_unobs_warmup_lr: float = 0.01        # learning rate for warmup phase
    u_unobs_low_rank: bool = False          # parameterise u_unobs = u_obs @ C
    u_unobs_low_rank_alpha: float = 1.0    # ridge α for low-rank C init
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
