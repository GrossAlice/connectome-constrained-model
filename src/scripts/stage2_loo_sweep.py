#!/usr/bin/env python3
"""Stage-2 LOO parameter sweep — test linear models, kernel ablation, tau counts.

Runs stage2 with k-fold temporal CV on 2 worms, sweeping over:
  1. baseline        — current defaults (5τ sv, 5τ dcv, nonlinear, edge G)
  2. linear          — chemical_synapse_activation='identity'
  3. no_kernels      — r_sv=0, r_dcv=0 (gap junctions + AR1 only)
  4. no_kernels_lin  — gap-only + linear
  5. 2tau            — 2 tau_sv, 2 tau_dcv
  6. 2tau_linear     — 2 tau + linear
  7. 1tau            — single tau per channel
  8. 1tau_linear     — 1 tau + linear
  9. scalar_G        — edge_specific_G=False (scalar gap conductance)
 10. no_W_learn      — fix W_sv, W_dcv (only learn G, λ, I0, amplitudes)
 11. hi_loo_aux      — strong loo_aux (weight=2, neurons=60, steps=50)
 12. lo_lambda_floor — lambda_u_lo=0.0 (let neurons be fully persistent)

Usage:
    python -u -m scripts.stage2_loo_sweep \
        --worms 2022-08-02-01 2022-06-14-01 \
        --out output_plots/stage2/loo_sweep_v2 \
        --epochs 30 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
#  Condition definitions                                                        #
# --------------------------------------------------------------------------- #

def _make_conditions() -> OrderedDict:
    """Return OrderedDict[name → dict of config overrides]."""
    C = OrderedDict()

    # 0. Baseline — current best defaults
    C["baseline"] = dict(
        edge_specific_G=True,
        learn_W_sv=True,
        learn_W_dcv=True,
        learn_noise=True,
        noise_mode="heteroscedastic",
        tau_sv_init=(0.5, 0.85, 1.5, 2.5, 5.0),
        a_sv_init=(2.5, 1.8, 1.2, 0.8, 0.5),
        tau_dcv_init=(2.0, 6.0, 10.0, 15.0, 20.0),
        a_dcv_init=(0.8, 0.6, 0.45, 0.3, 0.2),
        rollout_steps=0,
        loo_aux_weight=0.0,
    )

    # 1. Linear chemical synapses
    C["linear"] = {**C["baseline"], "chemical_synapse_activation": "identity"}

    # 2. No kernel dynamics — gap junctions + AR(1) only
    C["no_kernels"] = {
        **C["baseline"],
        "tau_sv_init": (),
        "a_sv_init": (),
        "tau_dcv_init": (),
        "a_dcv_init": (),
        "learn_W_sv": False,
        "learn_W_dcv": False,
    }

    # 3. No kernels + linear
    C["no_kernels_lin"] = {**C["no_kernels"], "chemical_synapse_activation": "identity"}

    # 4. 2-tau model (faster kernels + slow kernel each channel)
    C["2tau"] = {
        **C["baseline"],
        "tau_sv_init": (0.5, 2.5),
        "a_sv_init": (2.5, 0.8),
        "tau_dcv_init": (2.0, 10.0),
        "a_dcv_init": (0.8, 0.3),
    }

    # 5. 2-tau + linear
    C["2tau_linear"] = {**C["2tau"], "chemical_synapse_activation": "identity"}

    # 6. Single tau per channel
    C["1tau"] = {
        **C["baseline"],
        "tau_sv_init": (1.5,),
        "a_sv_init": (1.5,),
        "tau_dcv_init": (6.0,),
        "a_dcv_init": (0.5,),
    }

    # 7. 1-tau + linear
    C["1tau_linear"] = {**C["1tau"], "chemical_synapse_activation": "identity"}

    # 8. Scalar G (instead of edge-specific)
    C["scalar_G"] = {**C["baseline"], "edge_specific_G": False}

    # 9. Fix W (don't learn synaptic weight matrices)
    C["no_W_learn"] = {
        **C["baseline"],
        "learn_W_sv": False,
        "learn_W_dcv": False,
    }

    # 10. Strong LOO auxiliary loss (on baseline)
    C["hi_loo_aux"] = {
        **C["baseline"],
        "loo_aux_weight": 2.0,
        "loo_aux_steps": 50,
        "loo_aux_neurons": 60,
        "loo_aux_starts": 8,
        "rollout_steps": 20,
        "rollout_weight": 0.5,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    }

    # 11. LOO aux + linear + 2tau  (combo of promising ideas)
    C["combo_lin_2tau_loo"] = {
        **C["2tau"],
        "chemical_synapse_activation": "identity",
        "loo_aux_weight": 1.0,
        "loo_aux_steps": 30,
        "loo_aux_neurons": 32,
        "loo_aux_starts": 4,
        "rollout_steps": 10,
        "rollout_weight": 0.5,
        "warmstart_rollout": True,
    }

    # 12. Zero lambda floor (let lambda_u go to ~0 for persistent neurons)
    C["lo_lambda_floor"] = {**C["baseline"], "lambda_u_lo": 0.0}

    return C


# --------------------------------------------------------------------------- #
#  Sweep runner                                                                 #
# --------------------------------------------------------------------------- #

def run_one(
    worm_h5: str,
    cond_name: str,
    overrides: dict,
    *,
    base_out: Path,
    epochs: int,
    device: str,
    cv_folds: int,
    loo_subset: int,
) -> dict:
    """Train + evaluate a single (worm, condition) and return summary metrics."""
    from stage2.config import make_config
    from stage2.train import train_stage2

    tag = f"{Path(worm_h5).stem}_{cond_name}"
    save_dir = base_out / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build config with overrides
    kw = dict(overrides)
    kw["num_epochs"] = epochs
    kw["device"] = device
    kw["cv_folds"] = cv_folds
    kw["eval_loo_subset_size"] = loo_subset
    kw["eval_loo_subset_mode"] = "variance"
    kw["eval_loo_window_size"] = 50
    kw["eval_loo_warmup_steps"] = 40
    kw["learning_rate"] = 0.001
    kw["synaptic_lr_multiplier"] = 5.0
    kw["learn_lambda_u"] = True
    kw["learn_I0"] = True
    kw["per_neuron_amplitudes"] = True
    kw["fix_tau_sv"] = True
    kw["fix_tau_dcv"] = True
    kw["fix_a_sv"] = False
    kw["fix_a_dcv"] = False
    kw["behavior_weight"] = 0.0  # skip behaviour to speed things up
    # Keep noise learning on for all conditions
    kw.setdefault("learn_noise", True)
    kw.setdefault("noise_mode", "heteroscedastic")
    # Skip posture video & free-run stochastic to save time
    kw["make_posture_video"] = False
    kw["n_freerun_samples"] = 0

    cfg = make_config(worm_h5, **kw)
    # Monkey-patch skip_final_eval (not in dataclass but checked by train code)
    cfg.skip_final_eval = True

    t0 = time.time()
    try:
        result = train_stage2(cfg, save_dir=str(save_dir), show=False)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"tag": tag, "error": str(e)}
    elapsed = time.time() - t0

    # Extract key metrics
    summary = {
        "tag": tag,
        "worm": Path(worm_h5).stem,
        "condition": cond_name,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
    }
    if result is not None:
        for k in ("cv_onestep_r2_mean", "cv_onestep_r2_median",
                   "cv_loo_r2_mean", "cv_loo_r2_median",
                   "cv_loo_r2_windowed_mean", "cv_loo_r2_windowed_median"):
            v = result.get(k)
            summary[k] = round(float(v), 4) if v is not None and np.isfinite(v) else None

        # Per-neuron LOO R² distribution
        cv_loo = result.get("cv_loo_r2")
        if cv_loo is not None:
            valid = cv_loo[np.isfinite(cv_loo)]
            summary["loo_n_neurons"] = int(len(valid))
            summary["loo_n_positive"] = int((valid > 0).sum())
            summary["loo_q25"] = round(float(np.percentile(valid, 25)), 4) if len(valid) else None
            summary["loo_q75"] = round(float(np.percentile(valid, 75)), 4) if len(valid) else None

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worms", nargs="+", default=["2022-08-02-01", "2022-06-14-01"])
    parser.add_argument("--out", default="output_plots/stage2/loo_sweep_v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--loo_subset", type=int, default=30,
                        help="Number of neurons for LOO (0=all, 30=top-variance)")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Run only these conditions (default: all)")
    args = parser.parse_args()

    h5_dir = Path("data/used/behaviour+neuronal activity atanas (2023)/2")
    worm_paths = [str(h5_dir / f"{w}.h5") for w in args.worms]
    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    conditions = _make_conditions()
    if args.conditions:
        conditions = OrderedDict(
            (k, v) for k, v in conditions.items() if k in args.conditions
        )

    n_total = len(worm_paths) * len(conditions)
    print(f"\n{'='*70}")
    print(f"Stage-2 LOO Parameter Sweep")
    print(f"  Worms:      {args.worms}")
    print(f"  Conditions: {list(conditions.keys())}")
    print(f"  Total runs: {n_total}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  CV folds:   {args.cv_folds}")
    print(f"  LOO subset: {args.loo_subset}")
    print(f"  Output:     {base_out}")
    print(f"{'='*70}\n")

    all_results = []
    for wi, h5 in enumerate(worm_paths):
        for ci, (cname, overrides) in enumerate(conditions.items()):
            idx = wi * len(conditions) + ci + 1
            print(f"\n{'#'*70}")
            print(f"  [{idx}/{n_total}] worm={args.worms[wi]}  cond={cname}")
            print(f"{'#'*70}\n")
            sys.stdout.flush()

            summary = run_one(
                h5, cname, overrides,
                base_out=base_out,
                epochs=args.epochs,
                device=args.device,
                cv_folds=args.cv_folds,
                loo_subset=args.loo_subset,
            )
            all_results.append(summary)

            # Print running summary table
            _print_summary_table(all_results)

            # Save incremental results
            with open(base_out / "sweep_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    _print_summary_table(all_results)

    # Save final
    with open(base_out / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {base_out / 'sweep_results.json'}")


def _print_summary_table(results: list[dict]):
    """Print a compact summary table."""
    header = (
        f"{'Tag':<35} {'1-step':>7} {'LOO-med':>8} {'LOO-w':>7} "
        f"{'#pos':>5} {'time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['tag']:<35} ERROR: {r['error'][:40]}")
            continue
        onestep = r.get("cv_onestep_r2_median")
        loo_med = r.get("cv_loo_r2_median")
        loo_w = r.get("cv_loo_r2_windowed_median")
        n_pos = r.get("loo_n_positive", "?")
        elapsed = r.get("elapsed_s", 0)
        print(
            f"{r['tag']:<35} "
            f"{onestep:>7.4f} " if onestep is not None else f"{'N/A':>7} ",
            end="",
        )
        print(
            f"{loo_med:>8.4f} " if loo_med is not None else f"{'N/A':>8} ",
            end="",
        )
        print(
            f"{loo_w:>7.4f} " if loo_w is not None else f"{'N/A':>7} ",
            end="",
        )
        print(f"{n_pos:>5} {elapsed:>6.0f}s")


if __name__ == "__main__":
    main()
