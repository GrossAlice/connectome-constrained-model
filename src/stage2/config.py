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

    # --- Constraint-test-motivated options ---
    # Test 2: Ridge≫MLP ⇒ σ(u) operates in linear regime at calcium-imaging
    # timescale.  Setting True replaces sigmoid with identity (saves params).
    linear_chemical_synapses: bool = False
    # Chemical-synapse activation function φ(u) applied before synaptic pooling.
    # "sigmoid" = default  |  "tanh" = centered  |  "softplus" = non-neg monotone
    # "relu" = sparse  |  "elu" = smooth sparse  |  "swish" = self-gated
    # "shifted_sigmoid" = σ(α(u−β)) with learned slope+threshold  |  "identity"
    # When linear_chemical_synapses=True this is overridden to "identity".
    chemical_synapse_activation: str = "sigmoid"
    # Test 9: residual noise correlations ≈0.05-0.10; low-rank noise captures
    # shared unmodeled inputs (neuromodulation, unobserved neurons).
    noise_corr_rank: int = 0          # 0=diagonal; >0 = low-rank Σ = diag + V V^T

    edge_specific_G: bool = False      # False→scalar; True→per-edge (N,N)
    G_init_mode: str = "uniform"      # "uniform"|"log_counts"|"sqrt_counts"

    tau_sv_init: Tuple[float, ...] = (0.5, 2.5)
    a_sv_init: Tuple[float, ...] = (2.0, 0.8)
    fix_tau_sv: bool = True
    fix_a_sv: bool = False
    learn_W_sv: bool = True

    tau_dcv_init: Tuple[float, ...] = (3.0, 5.0)
    a_dcv_init: Tuple[float, ...] = (0.8, 0.6)
    fix_tau_dcv: bool = True
    fix_a_dcv: bool = False
    learn_W_dcv: bool = True
    per_neuron_amplitudes: bool = False # (N, r) per-neuron vs shared (r,)

    learn_reversals: bool = False
    reversal_mode: str = "per_neuron"   # "scalar"|"per_neuron"|"per_edge"

    u_next_clip_min: Optional[float] = -10
    u_next_clip_max: Optional[float] = 10
    learn_noise: bool = True
    noise_floor: float = 1e-3          # minimum sigma (softplus + floor)
    noise_reg: float = 0.0             # L2 penalty on log_sigma_u (prevents collapse)
    noise_mode: str = "heteroscedastic"  # "homoscedastic" | "heteroscedastic"

    # --- Per-neuron coupling gate (decoder analysis: linear neighbors hurt) ---
    # Learned g_i ∈ [0,1] per neuron scales total coupling (I_gap+I_sv+I_dcv+I_lr).
    # Neurons where neighbors hurt can close their gate (g_i→0).
    coupling_gate: bool = True
    coupling_gate_init: float = 0.0    # logit init (0 → sigmoid=0.5)

    # --- Non-connectome coupling and multi-hop propagation ---
    # Low-rank dense coupling: I_lr = V @ tanh(U @ u)
    lowrank_rank: int = 0              # 0=off; >0 captures non-connectome interactions
    # Graph polynomial: I_gap += α₂ L²u + α₃ L³u  (multi-hop gap junction)
    # Test 5: lag plateau at K≈5-10 → multi-hop helps capture indirect paths.
    graph_poly_order: int = 1          # 1=standard; 2-3 adds higher-order hops

    # --- Coupling dropout (regularise the connectome path) ---
    # During training, randomly zero out I_coupling for each neuron with
    # probability p.  Forces the model to be robust to missing connectome
    # inputs (analogous to DropConnect on the graph edges).
    coupling_dropout: float = 0.0      # 0=off; >0 = dropout probability

    # --- Residual MLP correction (close the gap with unconstrained MLP) ---
    # Adds I_mlp = MLP(u) to the dynamics update.  This captures arbitrary
    # nonlinear interactions that the connectome graph cannot represent
    # (missing edges, neuromodulation, volume transmission, etc.).
    # 0 = off; >0 = hidden dimension of 2-layer MLP.
    residual_mlp_hidden: int = 0
    residual_mlp_layers: int = 2
    residual_mlp_dropout: float = 0.1
    residual_mlp_context: int = 1      # K=1: MLP sees u_t only; K>1: MLP sees [u_t,...,u_{t-K+1}]

    # --- Linear lag terms (AR(K) per neuron + connectome-sparse neighbor lags) ---
    # Adds I_lag = Σ_k α_{k,i} u_{i,t-k}  (self-lags, diagonal, K×N params)
    # and optionally Σ_k G^(k)_{ij} u_{j,t-k}  (neighbor-lags along connectome edges).
    # Matches Ridge baseline's multi-step memory without an MLP.
    lag_order: int = 0                 # 0=off; K>0 adds K self-lag terms
    lag_neighbor: bool = False         # True: also add connectome-sparse neighbor lags

@dataclass
class BehaviorConfig:
    behavior_lag_steps: int = 8
    behavior_weight: float = 0.0
    behavior_n_modes: int = 6           # 0=use all eigenworm modes from the HDF5
    # Test 7: behavior coupling gain ≈0.005-0.01 → cap stimulus weights to
    # prevent over-fitting the weakest term in the dynamics.
    behavior_weight_cap: float = 0.0   # >0 : L∞ penalty weight on model.b
    behavior_decoder_type: str = "mlp"  # "linear"|"mlp"
    behavior_decoder_hidden: int = 128
    behavior_decoder_n_layers: int = 2
    behavior_decoder_dropout: float = 0.1
    #Linear parameters (only used if behavior_decoder_type="linear")
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
    num_epochs: int = 50
    learning_rate: float = 0.001
    device: str = "cuda"
    grad_clip_norm: float = 1
    dynamics_l2:    float = 0.0
    seed: int = 42                  # global RNG seed for reproducibility (0=unseeded)

    rollout_steps: int = 30
    rollout_weight: float = 0.0
    rollout_starts: int = 8
    warmstart_rollout: bool = True

    # Noise injection: add Gaussian noise to teacher-forced inputs
    input_noise_sigma: float = 0.0   # σ of additive Gaussian noise on u_t during training (0=off)

    # Curriculum rollout scheduling: linearly ramp rollout_steps over epochs
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
    coupling_gate_reg: float = 0.0      # L2 on gate logits (→ gate≈0.5)
    network_strength_floor: float = 1.0   # weight of one-sided penalty keeping G*a_sv above target
    network_strength_target: float = 0.8  # fraction of init G_rms*a_sv_rms to maintain

    # LOO auxiliary loss
    loo_aux_weight: float = 0.0      # >0 enables differentiable LOO during training
    loo_aux_steps: int = 0
    loo_aux_neurons: int = 0
    loo_aux_starts: int = 8
    cv_folds: int = 2                # k-fold temporal CV (must be ≥ 2)


@dataclass
class EvalConfig:
    eval_loo_subset_size: int = 30
    eval_loo_subset_mode: str = "variance"  # "named"|"variance"|"best_onestep"|"random"
    eval_loo_subset_names: Tuple[str, ...] = ("AVAL", "AVAR")
    eval_loo_subset_seed: int = 0
    eval_loo_subset_neurons: Optional[Tuple[int, ...]] = None
    eval_free_run_seed_steps: int = 16

    n_sample_trajectories: int = 20
    n_freerun_samples: int = 20
    eval_loo_window_size: int = 50
    eval_loo_warmup_steps: int = 40  # teacher-forced burn-in before each LOO window (0=cold)


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
    # --- Connectome-informed unobserved init (option 4) ---
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
