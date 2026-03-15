from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, get_type_hints


_DATA_ROOT = (Path(__file__).resolve().parent / ".." / "data" / "used" / "masks+motor neurons").resolve()


def _data_file(name: str) -> str:
    return str((_DATA_ROOT / name).resolve())


def _tuple_default(*values: float):
    return field(default_factory=lambda: values)


def _load_motor_neurons() -> Tuple[str, ...]:
    path = _DATA_ROOT / "motor_neurons_with_control.txt"
    if not path.exists():
        return ()
    return tuple(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


@dataclass
class DataConfig:
    h5_path: str
    stage1_u_dataset: str = "stage1/u_mean"
    stage1_u_var_dataset: Optional[str] = "stage1/u_var"
    stage1_params_group: str = "stage1/params"
    silencing_dataset: Optional[str] = None
    dt: Optional[float] = None
    T_e_dataset: Optional[str] = _data_file("T_e.npy")
    T_sv_dataset: Optional[str] = _data_file("T_sv.npy")
    T_dcv_dataset: Optional[str] = _data_file("T_dcv.npy")
    neurotransmitter_sign_dataset: Optional[str] = _data_file("neurotransmitter_sign.npy")


@dataclass
class DynamicsConfig:
    G_init: float = 0.01
    edge_specific_G: bool = False
    G_min: float = 0.0
    G_max:           Optional[float] = 2.0

    r_sv: int = 5
    tau_sv_init: Tuple[float, ...] = _tuple_default(0.5, 0.85, 1.5, 2.5, 5.0)
    a_sv_init: Tuple[float, ...] = _tuple_default(0.25, 0.18, 0.12, 0.08, 0.05)
    fix_tau_sv: bool = True
    W_sv_init: float = 1.0
    learn_W_sv: bool = False
    a_sv_min: float = 0.0
    a_sv_max: Optional[float] = 2.0
    tau_sv_min: float = 1e-4
    tau_sv_max: Optional[float] = None

    E_sv_exc_init: float = 0.1
    E_sv_inh_init: float = -0.1

    r_dcv: int = 5
    tau_dcv_init: Tuple[float, ...] = _tuple_default(2.0, 6.0, 10.0, 15.0, 20.0)
    a_dcv_init: Tuple[float, ...] = _tuple_default(0.08, 0.06, 0.045, 0.03, 0.02)
    fix_tau_dcv: bool = True
    W_dcv_init: float = 1.0
    learn_W_dcv: bool = False
    a_dcv_min: float = 0.0
    a_dcv_max: Optional[float] = 2.0
    tau_dcv_min: float = 1e-4
    tau_dcv_max: Optional[float] = None

    E_dcv_init: float = 0.0

    learn_reversals: bool = False
    init_network_frac: float = 0.1

    learn_lambda_u: bool = True
    lambda_u_warmup_frac: float = 0.2
    lambda_u_min:         float = 0.0
    lambda_u_max:         float = 0.9999

    learn_I0: bool = False
    

    alpha_per_neuron: bool = True
    alpha_cv_every: int = 0
    alpha_cv_n_folds:        int   = 5
    alpha_cv_log_lambda_min: float = -10.0
    alpha_cv_log_lambda_max: float = 10.0
    alpha_cv_n_grid:         int   = 50

    u_next_clip_min: Optional[float] = -35
    u_next_clip_max: Optional[float] = 35


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
    behavior_weight:         float = 0.0
    behavior_decoder_mode:   str   = "e2e"
    fit_all_neuron_baseline: bool  = True

    train_behavior_ridge_folds: int = 5
    train_behavior_ridge_log_lambda_min: float = -3.0
    train_behavior_ridge_log_lambda_max: float = 10.0
    train_behavior_ridge_n_grid:         int   = 50
    train_behavior_ridge_disable:        bool  = False

    e2e_decoder_l2: float = 1e-3


@dataclass
class TrainConfig:
    num_epochs: int = 40
    learning_rate: float = 0.001
    device: str = "cuda"
    grad_clip_norm: float = 1.0
    beta: float = 1.0
    dynamics_l2:    float = 0.0
    dynamics_objective: str = "one_step"

    rollout_steps: int = 15
    rollout_weight: float = 0.5
    rollout_starts: int = 8
    warmstart_rollout: bool = True

    neuron_dropout_frac: float = 0.2
    posthoc_cv_regularize: bool = False
    sigma_u_default:     float = 1.0
    use_u_var_weighting: bool  = False
    u_var_scale:         float = 5.0
    u_var_floor:         float = 1e-6


@dataclass
class EvalConfig:
    eval_loo_subset_size: int = 5
    eval_loo_subset_mode: str = "variance"
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
    posture_video_out:        Optional[str] = None


@dataclass
class Stage2PTConfig:
    """Top-level config with flat access to nested dataclass fields."""

    data: DataConfig
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __getattr__(self, name: str):
        owner = _FLAT_CONFIG_OWNER.get(name)
        if owner is not None:
            return getattr(object.__getattribute__(self, owner), name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute {name!r}")

    def __setattr__(self, name: str, value):
        if name in type(self).__dataclass_fields__:
            object.__setattr__(self, name, value)
            return
        owner = _FLAT_CONFIG_OWNER.get(name)
        if owner is not None:
            setattr(object.__getattribute__(self, owner), name, value)
            return
        object.__setattr__(self, name, value)


_SUBCONFIG_TYPES = {
    name: cls
    for name, cls in get_type_hints(Stage2PTConfig).items()
    if name in Stage2PTConfig.__dataclass_fields__ and dataclasses.is_dataclass(cls)
}

_FLAT_CONFIG_OWNER = {
    field_name: section_name
    for section_name, cls in _SUBCONFIG_TYPES.items()
    for field_name in cls.__dataclass_fields__
}


def make_config(h5_path: str, **kwargs) -> Stage2PTConfig:
    """Build `Stage2PTConfig` from flat keyword arguments."""
    sub_kwargs: dict[str, dict] = {name: {} for name in _SUBCONFIG_TYPES}
    sub_kwargs["data"]["h5_path"] = h5_path

    for key, val in kwargs.items():
        if (owner := _FLAT_CONFIG_OWNER.get(key)) is not None:
            sub_kwargs[owner][key] = val

    bad = sorted(k for k in kwargs if k not in _FLAT_CONFIG_OWNER)
    if bad:
        warnings.warn(f"Unrecognised config keys (ignored): {bad}", stacklevel=2)

    return Stage2PTConfig(
        **{name: cls(**sub_kwargs[name]) for name, cls in _SUBCONFIG_TYPES.items()}
    )
