from __future__ import annotations

import argparse
from pathlib import Path
from .config import Stage2PTConfig, make_config
from . import get_stage2_logger
from .train import train_stage2


def main(argv: list[str] | None = None) -> None:
    cfg_default = make_config("")
    parser = argparse.ArgumentParser(description="Train PyTorch Stage 2 model")
    
    # Input
    parser.add_argument("--h5", "--h5_path", dest="h5_path", type=str, required=True,
                        help="Path to input HDF5 file")
    parser.add_argument("--stage1_u_dataset", type=str, default=cfg_default.stage1_u_dataset,
                        help="HDF5 dataset key for Stage 1 u_mean (default: 'stage1/u_mean')")
    parser.add_argument("--stage1_u_var_dataset", type=str, default=cfg_default.stage1_u_var_dataset,
                        help="HDF5 dataset key for Stage 1 u_var, or 'none' to disable (default: 'stage1/u_var')")
    parser.add_argument("--stage1_params_group", type=str, default=cfg_default.stage1_params_group,
                        help="HDF5 group key for Stage 1 params (default: 'stage1/params')")
    parser.add_argument("--sigma_u_default", type=float, default=cfg_default.sigma_u_default,
                        help="Default observation noise std for dynamics loss (default: 1.0)")
    parser.add_argument("--num_epochs", type=int, default=cfg_default.num_epochs, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=cfg_default.learning_rate, help="Learning rate for Adam optimiser")
    parser.add_argument("--device", type=str, default=cfg_default.device, help="PyTorch device: 'cpu' or 'cuda'")
    parser.add_argument("--dynamics_scale", type=float, default=cfg_default.dynamics_scale, help="Weight on dynamics loss term")
    parser.add_argument("--r_sv", type=int, default=cfg_default.r_sv, help="Number of exponential basis functions for SV kernels")
    parser.add_argument("--r_dcv", type=int, default=cfg_default.r_dcv, help="Number of exponential basis functions for DCV kernels")
    parser.add_argument(
        "--fix_tau_sv",
        dest="fix_tau_sv",
        action="store_true",
        default=None,
        help="Fix SV synaptic time constants tau_sv (do not learn).",
    )
    parser.add_argument(
        "--no_fix_tau_sv",
        dest="fix_tau_sv",
        action="store_false",
        default=None,
        help="Learn SV synaptic time constants tau_sv.",
    )
    parser.add_argument(
        "--fix_tau_dcv",
        dest="fix_tau_dcv",
        action="store_true",
        default=None,
        help="Fix DCV synaptic time constants tau_dcv (do not learn).",
    )
    parser.add_argument(
        "--no_fix_tau_dcv",
        dest="fix_tau_dcv",
        action="store_false",
        default=None,
        help="Learn DCV synaptic time constants tau_dcv.",
    )
    parser.add_argument(
        "--fix_taus",
        action="store_true",
        default=False,
        help="Convenience flag: fix both tau_sv and tau_dcv (do not learn time constants).",
    )
    parser.add_argument("--tau_sv_init", type=float, nargs="*", default=None, help="Initial SV time constants (seconds)")
    parser.add_argument("--a_sv_init", type=float, nargs="*", default=None, help="Initial SV amplitudes")
    parser.add_argument("--tau_dcv_init", type=float, nargs="*", default=None, help="Initial DCV time constants (seconds)")
    parser.add_argument("--a_dcv_init", type=float, nargs="*", default=None, help="Initial DCV amplitudes")

    # Optional per-edge synaptic weights
    parser.add_argument(
        "--learn_W_sv",
        action="store_true",
        default=cfg_default.learn_W_sv,
        help="Learn per-edge SV synaptic conductances W_sv(i,j) >= 0 (sign from E_sv).",
    )
    parser.add_argument(
        "--learn_W_dcv",
        action="store_true",
        default=cfg_default.learn_W_dcv,
        help="Learn per-edge DCV synaptic conductances W_dcv(i,j) >= 0.",
    )
    parser.add_argument(
        "--W_sv_init",
        type=float,
        default=cfg_default.W_sv_init,
        help="Initial value for per-edge SV weights (only used if --learn_W_sv).",
    )
    parser.add_argument(
        "--W_dcv_init",
        type=float,
        default=cfg_default.W_dcv_init,
        help="Initial value for per-edge DCV weights (only used if --learn_W_dcv).",
    )
    parser.add_argument("--G_init", type=float, default=cfg_default.G_init, help="Initial gap junction strength")
    parser.add_argument(
        "--edge_specific_G",
        action="store_true",
        default=cfg_default.edge_specific_G,
        help="Learn per-edge gap junction weights G (N×N) instead of a scalar.",
    )
    parser.add_argument(
        "--learn_I0",
        dest="learn_I0",
        action="store_true",
        default=None,
        help="Learn per-neuron tonic bias current I0.",
    )
    parser.add_argument(
        "--fix_I0",
        dest="learn_I0",
        action="store_false",
        default=None,
        help="Fix I0 at per-neuron baseline (mode of each trace).",
    )

    parser.add_argument(
        "--learn_reversals",
        dest="learn_reversals",
        action="store_true",
        default=None,
        help="Learn reversal potentials E_sv/E_dcv (overrides config if provided).",
    )
    parser.add_argument(
        "--fix_reversals",
        dest="learn_reversals",
        action="store_false",
        default=None,
        help="Fix reversal potentials at init values (overrides config if provided).",
    )
    parser.add_argument("--E_sv_init", type=float, default=None,
                        help="Initial SV excitatory reversal potential (sets E_sv_exc_init)")
    parser.add_argument("--E_dcv_init", type=float, default=None,
                        help="Initial DCV reversal potential (default: 0.0)")
    parser.add_argument(
        "--learn_lambda_u",
        dest="learn_lambda_u",
        action="store_true",
        default=None,
        help="Learn lambda_u during training (init from Stage 1 rho).",
    )
    parser.add_argument(
        "--fix_lambda_u",
        dest="learn_lambda_u",
        action="store_false",
        default=None,
        help="Keep lambda_u fixed at Stage 1 rho values.",
    )
    parser.add_argument(
        "--force_interactions",
        action="store_true",
        help=(
            "Convenience flag to encourage interaction-driven dynamics: "
            "sets --fix_lambda_u. "
            "(Does not change G_init unless you pass it.)"
        ),
    )
    parser.add_argument("--dt", type=float, default=cfg_default.dt, help="Override sampling interval (seconds)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory for diagnostic plots (omit to skip plotting)")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")

    parser.add_argument(
        "--plot_every",
        type=int,
        default=cfg_default.plot_every,
        help=(
            "If >0 and --save_dir is set, save plots every N epochs into "
            "save_dir/epoch_XXXX (default: config value)."
        ),
    )
    parser.add_argument(
        "--eval_loo_subset_size",
        type=int,
        default=cfg_default.eval_loo_subset_size,
        help="Optional number of neurons for LOO in eval_loo plots (0=all).",
    )
    parser.add_argument(
        "--eval_loo_subset_mode",
        type=str,
        default=cfg_default.eval_loo_subset_mode,
        choices=["variance", "random", "worst_onestep", "best_onestep"],
        help="Subset selection mode for eval_loo plots.",
    )
    parser.add_argument(
        "--eval_loo_subset_seed",
        type=int,
        default=cfg_default.eval_loo_subset_seed,
        help="Random seed for random eval_loo subset selection.",
    )
    parser.add_argument(
        "--eval_loo_subset_neurons",
        type=int,
        nargs="*",
        default=None,
        help="Explicit neuron indices for eval_loo LOO (overrides subset size/mode).",
    )
    # dataset keys for connectome masks
    parser.add_argument(
        "--T_e_dataset",
        type=str,
        default=None,
        help="Dataset key for gap junction adjacency mask (T_e).  Overrides the default 'connectome/T_e'",
    )
    parser.add_argument(
        "--T_sv_dataset",
        type=str,
        default=None,
        help="Dataset key for SV synapse mask (T_sv).  Overrides the default 'connectome/T_sv'",
    )
    parser.add_argument(
        "--T_dcv_dataset",
        type=str,
        default=None,
        help="Dataset key for DCV synapse mask (T_dcv).  Overrides the default 'connectome/T_dcv'",
    )
    parser.add_argument(
        "--neurotransmitter_sign_dataset",
        type=str,
        default=None,
        help="Path to neurotransmitter sign matrix (+1=exc, -1=inh). "
             "Used to set reversal potentials E_sv. Set to 'none' to disable.",
    )

    # dataset key for silencing mask
    parser.add_argument(
        "--silencing_dataset",
        type=str,
        default=None,
        help="Dataset key for neuron silencing mask.  Overrides the default of no silencing (all ones).",
    )

    # dataset key for stimulus regressor
    parser.add_argument(
        "--stim_dataset",
        type=str,
        default=None,
        help="Dataset key for stimulus regressor (external drive).  If provided, model learns per-neuron weights.",
    )
    parser.add_argument(
        "--ridge_b",
        type=float,
        default=cfg_default.ridge_b,
        help=f"Ridge regularization strength for stimulus weights (default: {cfg_default.ridge_b}).",
    )
    parser.add_argument(
        "--dynamics_l2",
        type=float,
        default=cfg_default.dynamics_l2,
        help="L2 penalty on trainable dynamics model parameters (default: 0.0).",
    )
    parser.add_argument(
        "--dynamics_objective",
        type=str,
        default=cfg_default.dynamics_objective,
        choices=["one_step", "rollout"],
        help="Dynamics objective: 'one_step' (teacher-forced) or 'rollout' (full trajectory).",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=cfg_default.rollout_steps,
        help="Short-horizon rollout steps for auxiliary loss (0 = disabled).",
    )
    parser.add_argument(
        "--rollout_weight",
        type=float,
        default=cfg_default.rollout_weight,
        help=f"Weight of short-horizon rollout loss (default: {cfg_default.rollout_weight}).",
    )
    parser.add_argument(
        "--rollout_starts",
        type=int,
        default=cfg_default.rollout_starts,
        help="Number of random starting points per epoch for rollout loss (default: 8).",
    )
    parser.add_argument(
        "--alpha_cv_every",
        type=int,
        default=cfg_default.alpha_cv_every,
        help="Re-solve per-neuron kernel amplitudes via ridge-CV every N epochs (0=disabled).",
    )
    parser.add_argument(
        "--neuron_dropout_frac",
        type=float,
        default=cfg_default.neuron_dropout_frac,
        help="Fraction of neurons masked per step for LOO-aligned training (0=disabled).",
    )
    parser.add_argument(
        "--warmstart_rollout",
        type=int,
        default=int(cfg_default.warmstart_rollout),
        help="Warm-start rollout synaptic states from teacher-forced trajectory (0=off, 1=on).",
    )

    parser.add_argument(
        "--stim_diagonal_only",
        dest="stim_diagonal_only",
        action="store_true",
        default=cfg_default.stim_diagonal_only,
        help=(
            "If stim is neuron-specific (T,N), constrain stimulus mapping to be diagonal: "
            "I_stim_i(t)=b_i*stim_i(t). Prevents learning a full (N,N) mixing matrix."
        ),
    )
    parser.add_argument(
        "--no_stim_diagonal_only",
        dest="stim_diagonal_only",
        action="store_false",
        help="Allow full stimulus mixing instead of diagonal-only mapping.",
    )

    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=cfg_default.grad_clip_norm,
        help="Optional gradient clipping max-norm (0 disables).",
    )
    parser.add_argument(
        "--G_max",
        type=float,
        default=cfg_default.G_max,
        help="Upper bound on gap-junction conductances (None = unbounded).",
    )
    parser.add_argument(
        "--a_sv_max",
        type=float,
        default=cfg_default.a_sv_max,
        help="Upper bound on SV kernel amplitudes (None = unbounded).",
    )
    parser.add_argument(
        "--a_dcv_max",
        type=float,
        default=cfg_default.a_dcv_max,
        help="Upper bound on DCV kernel amplitudes (None = unbounded).",
    )
    parser.add_argument("--b_max", type=float, default=cfg_default.b_max, help="Optional max abs clamp for stimulus weights b")

    # variance-weighted loss using Stage 1 u_var
    parser.add_argument(
        "--use_u_var_weighting",
        action="store_true",
        default=cfg_default.use_u_var_weighting,
        help=(
            "Weight dynamics loss by Stage 1 posterior variance u_var when available: "
            "uses denom = sigma_u^2 + u_var_scale * max(u_var, u_var_floor)."
        ),
    )
    parser.add_argument(
        "--no_use_u_var_weighting",
        dest="use_u_var_weighting",
        action="store_false",
        help="Disable variance-weighted dynamics loss.",
    )
    parser.add_argument(
        "--u_var_scale",
        type=float,
        default=cfg_default.u_var_scale,
        help="Scale factor applied to Stage 1 u_var in variance-weighted loss (default: config value).",
    )
    parser.add_argument(
        "--u_var_floor",
        type=float,
        default=cfg_default.u_var_floor,
        help="Floor applied to Stage 1 u_var before using it in the loss (default: config value).",
    )


    # behaviour decoding arguments
    parser.add_argument(
        "--behavior_dataset",
        type=str,
        default=cfg_default.behavior_dataset,
        help="Dataset key for behavioural observations (default: behaviour/eigenworms_calc_6).",
    )
    parser.add_argument(
        "--motor_neurons",
        type=int,
        nargs="*",
        default=None,
        help="Motor neurons used by the behaviour decoder. Provide a space-separated list of indices to override the config/default selection.",
    )
    parser.add_argument(
        "--behavior_lag_steps",
        type=int,
        default=None,
        help=f"Number of time lags for the behaviour decoder input. z_M(t) = [u_M(t), ..., u_M(t-L)]. Default: {cfg_default.behavior_lag_steps}.",
    )
    parser.add_argument(
        "--behavior_weight",
        type=float,
        default=None,
        help=f"Weight of behaviour prediction loss in training (default: {cfg_default.behavior_weight}).",
    )
    parser.add_argument(
        "--make_posture_video",
        dest="make_posture_video",
        action="store_true",
        default=None,
        help="Generate posture comparison MP4 (raw vs Stage 1 decoded vs model-decoded) at end of training.",
    )
    parser.add_argument(
        "--no_make_posture_video",
        dest="make_posture_video",
        action="store_false",
        default=None,
        help="Disable posture comparison MP4 generation.",
    )
    parser.add_argument(
        "--posture_video_fps",
        type=int,
        default=cfg_default.posture_video_fps,
        help="FPS for end-of-training posture comparison video.",
    )
    parser.add_argument(
        "--posture_video_dpi",
        type=int,
        default=cfg_default.posture_video_dpi,
        help="DPI for end-of-training posture comparison video.",
    )
    parser.add_argument(
        "--posture_video_max_frames",
        type=int,
        default=cfg_default.posture_video_max_frames,
        help="Maximum frames in posture video (0 = all).",
    )
    parser.add_argument(
        "--posture_video_out",
        type=str,
        default=None,
        help="Optional output MP4 path for posture comparison video.",
    )
    args = parser.parse_args(argv)

    if getattr(args, "force_interactions", False):
        args.learn_lambda_u = False

    cfg_kwargs = {
        "h5_path": args.h5_path,
        "stage1_u_dataset": args.stage1_u_dataset,
        "stage1_params_group": args.stage1_params_group,
        "sigma_u_default": args.sigma_u_default,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "dynamics_scale": args.dynamics_scale,
        "r_sv": args.r_sv,
        "r_dcv": args.r_dcv,
        "fix_tau_sv": ((cfg_default.fix_tau_sv if getattr(args, "fix_tau_sv", None) is None else bool(args.fix_tau_sv))
                or bool(getattr(args, "fix_taus", False))),
        "fix_tau_dcv": ((cfg_default.fix_tau_dcv if getattr(args, "fix_tau_dcv", None) is None else bool(args.fix_tau_dcv))
                 or bool(getattr(args, "fix_taus", False))),
        "G_init": args.G_init,
        "edge_specific_G": bool(getattr(args, "edge_specific_G", False)),
        "learn_lambda_u": (cfg_default.learn_lambda_u if getattr(args, "learn_lambda_u", None) is None else bool(args.learn_lambda_u)),
        "plot_every": int(getattr(args, "plot_every", cfg_default.plot_every) or 0),
        "eval_loo_subset_size": int(getattr(args, "eval_loo_subset_size", cfg_default.eval_loo_subset_size) or 0),
        "eval_loo_subset_mode": str(getattr(args, "eval_loo_subset_mode", cfg_default.eval_loo_subset_mode)),
        "eval_loo_subset_seed": int(getattr(args, "eval_loo_subset_seed", cfg_default.eval_loo_subset_seed) or 0),
        "use_u_var_weighting": bool(getattr(args, "use_u_var_weighting", cfg_default.use_u_var_weighting)),
        "u_var_scale": float(getattr(args, "u_var_scale", cfg_default.u_var_scale)),
        "u_var_floor": float(getattr(args, "u_var_floor", cfg_default.u_var_floor)),
        "stim_diagonal_only": bool(getattr(args, "stim_diagonal_only", cfg_default.stim_diagonal_only)),
        "grad_clip_norm": float(getattr(args, "grad_clip_norm", cfg_default.grad_clip_norm) or 0.0),
        "b_max": getattr(args, "b_max", None),

        "learn_W_sv": bool(getattr(args, "learn_W_sv", False)),
        "learn_W_dcv": bool(getattr(args, "learn_W_dcv", False)),
        "W_sv_init": float(getattr(args, "W_sv_init", cfg_default.W_sv_init)),
        "W_dcv_init": float(getattr(args, "W_dcv_init", cfg_default.W_dcv_init)),
    }

    # parameter upper bounds
    if getattr(args, "G_max", None) is not None:
        cfg_kwargs["G_max"] = float(args.G_max)
    if getattr(args, "a_sv_max", None) is not None:
        cfg_kwargs["a_sv_max"] = float(args.a_sv_max)
    if getattr(args, "a_dcv_max", None) is not None:
        cfg_kwargs["a_dcv_max"] = float(args.a_dcv_max)

    # stage1_u_var_dataset: support "none" to disable
    _uvar = getattr(args, "stage1_u_var_dataset", cfg_default.stage1_u_var_dataset)
    if _uvar is not None and str(_uvar).lower() == "none":
        cfg_kwargs["stage1_u_var_dataset"] = None
    elif _uvar is not None:
        cfg_kwargs["stage1_u_var_dataset"] = _uvar

    if getattr(args, "learn_reversals", None) is not None:
        cfg_kwargs["learn_reversals"] = bool(args.learn_reversals)
    if getattr(args, "learn_I0", None) is not None:
        cfg_kwargs["learn_I0"] = bool(args.learn_I0)
    if getattr(args, "E_sv_init", None) is not None:
        cfg_kwargs["E_sv_exc_init"] = float(args.E_sv_init)
    if getattr(args, "E_dcv_init", None) is not None:
        cfg_kwargs["E_dcv_init"] = float(args.E_dcv_init)
    # parse optional time constants and amplitudes
    if args.tau_sv_init is not None and len(args.tau_sv_init) > 0:
        cfg_kwargs["tau_sv_init"] = tuple(args.tau_sv_init)
    if args.a_sv_init is not None and len(args.a_sv_init) > 0:
        cfg_kwargs["a_sv_init"] = tuple(args.a_sv_init)
    if args.tau_dcv_init is not None and len(args.tau_dcv_init) > 0:
        cfg_kwargs["tau_dcv_init"] = tuple(args.tau_dcv_init)
    if args.a_dcv_init is not None and len(args.a_dcv_init) > 0:
        cfg_kwargs["a_dcv_init"] = tuple(args.a_dcv_init)
    if args.dt is not None:
        cfg_kwargs["dt"] = args.dt
    # override mask datasets if provided
    if args.T_e_dataset is not None:
        cfg_kwargs["T_e_dataset"] = args.T_e_dataset
    if args.T_sv_dataset is not None:
        cfg_kwargs["T_sv_dataset"] = args.T_sv_dataset
    if args.T_dcv_dataset is not None:
        cfg_kwargs["T_dcv_dataset"] = args.T_dcv_dataset
    sign_arg = getattr(args, "neurotransmitter_sign_dataset", None)
    if sign_arg is not None:
        cfg_kwargs["neurotransmitter_sign_dataset"] = None if sign_arg.lower() == "none" else sign_arg
    # silencing and warm start
    if args.silencing_dataset is not None:
        cfg_kwargs["silencing_dataset"] = args.silencing_dataset
    if args.stim_dataset is not None:
        cfg_kwargs["stim_dataset"] = args.stim_dataset
    if args.ridge_b is not None:
        cfg_kwargs["ridge_b"] = args.ridge_b
    if getattr(args, "dynamics_l2", None) is not None:
        cfg_kwargs["dynamics_l2"] = float(args.dynamics_l2)
    if getattr(args, "dynamics_objective", None) is not None:
        cfg_kwargs["dynamics_objective"] = str(args.dynamics_objective)
    if getattr(args, "rollout_steps", None) is not None:
        cfg_kwargs["rollout_steps"] = int(args.rollout_steps)
    if getattr(args, "rollout_weight", None) is not None:
        cfg_kwargs["rollout_weight"] = float(args.rollout_weight)
    if getattr(args, "rollout_starts", None) is not None:
        cfg_kwargs["rollout_starts"] = int(args.rollout_starts)
    if getattr(args, "alpha_cv_every", None) is not None:
        cfg_kwargs["alpha_cv_every"] = int(args.alpha_cv_every)
    if getattr(args, "neuron_dropout_frac", None) is not None:
        cfg_kwargs["neuron_dropout_frac"] = float(args.neuron_dropout_frac)
    if getattr(args, "warmstart_rollout", None) is not None:
        cfg_kwargs["warmstart_rollout"] = bool(int(args.warmstart_rollout))
    if args.eval_loo_subset_neurons is not None and len(args.eval_loo_subset_neurons) > 0:
        cfg_kwargs["eval_loo_subset_neurons"] = tuple(args.eval_loo_subset_neurons)
    # behaviour
    if args.behavior_dataset is not None:
        cfg_kwargs["behavior_dataset"] = args.behavior_dataset
    if args.motor_neurons is not None:
        cfg_kwargs["motor_neurons"] = tuple(args.motor_neurons)
    if args.behavior_lag_steps is not None:
        cfg_kwargs["behavior_lag_steps"] = args.behavior_lag_steps
    if getattr(args, "behavior_weight", None) is not None:
        cfg_kwargs["behavior_weight"] = float(args.behavior_weight)
    if getattr(args, "make_posture_video", None) is not None:
        cfg_kwargs["make_posture_video"] = bool(args.make_posture_video)
    if getattr(args, "posture_video_fps", None) is not None:
        cfg_kwargs["posture_video_fps"] = int(args.posture_video_fps)
    if getattr(args, "posture_video_dpi", None) is not None:
        cfg_kwargs["posture_video_dpi"] = int(args.posture_video_dpi)
    if getattr(args, "posture_video_max_frames", None) is not None:
        cfg_kwargs["posture_video_max_frames"] = int(args.posture_video_max_frames)
    if getattr(args, "posture_video_out", None):
        cfg_kwargs["posture_video_out"] = str(args.posture_video_out)
    
    # ========================================================================
    # Train
    # ========================================================================
    h5_path = cfg_kwargs.pop("h5_path", args.h5_path)
    cfg = make_config(h5_path, **cfg_kwargs)
    eval_result = train_stage2(cfg, save_dir=args.save_dir, show=args.show)

    if getattr(cfg, "make_posture_video", False) and eval_result is not None:
        _make_posture_video(cfg, eval_result, args.save_dir)


def _make_posture_video(
    cfg: Stage2PTConfig, eval_result: dict, save_dir: str | None,
) -> None:
    try:
        try:
            from scripts.posture_videos import make_posture_compare_video
        except ModuleNotFoundError:
            from scripts.posture_compare_video import make_posture_compare_video

        beh = eval_result.get("beh")
        if beh is None:
            get_stage2_logger().warning("posture_video_skipped", reason="no_behaviour_predictions")
            return
        out_path = str(
            getattr(cfg, "posture_video_out", None)
            or str(Path(save_dir or ".") / "13_posture_comparison.mp4")
        )
        make_posture_compare_video(
            h5_path=str(cfg.h5_path),
            out_path=out_path,
            ew_raw=beh["b_actual"],
            ew_stage1=beh["b_pred_gt"],
            ew_model_cv=beh["b_pred_model"],
            fps=int(getattr(cfg, "posture_video_fps", 15) or 15),
            dpi=int(getattr(cfg, "posture_video_dpi", 120) or 120),
            max_frames=int(getattr(cfg, "posture_video_max_frames", 0) or 0),
        )
    except Exception as e:
        import traceback
        get_stage2_logger().warning("posture_video_failed", error=str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()