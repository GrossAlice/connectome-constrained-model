#!/usr/bin/env python3
"""Stage-2 LOO sweep v3 — coupling regularisation, coupling gate, learned
reversals, per-neuron tau scale.

Motivated by the neural-activity-decoder analysis:
  • Ridge self-only R²=0.84, adding linear neighbours HURTS (Δ=−0.13)
  • Only the Transformer benefits from neighbours (+0.08) → attention/gating
  • LOO ceiling from causal-only ≈ 0.2–0.3; current stage2 ≈ 0.09 (3× headroom)
  • Conductance-based g*(E−u) creates voltage-dependent gating that can be
    exploited by learning reversals (each neuron sets its own coupling point)

Conditions (all use v5 defaults as base):
  0. v5_base         — current v5 defaults (loo_aux=1, rollout=30, 150ep)
  1. int_l2_01       — interaction_l2=0.01 (penalise deviation from AR1)
  2. int_l2_10       — interaction_l2=0.1
  3. ridge_W         — ridge_W_sv=0.1, ridge_W_dcv=0.1 (shrink synaptic weights)
  4. G_reg           — G_reg=0.1 (anchor gap conductances)
  5. gate            — coupling_gate=True (per-neuron gate on coupling)
  6. gate_int_l2     — coupling_gate + interaction_l2=0.1
  7. learn_E         — learn_reversals=True (conductance-based gating via E−u)
  8. gate_E          — coupling_gate + learn_reversals
  9. tau_scale       — per_neuron_tau_scale=True
 10. gate_E_tau      — gate + E + tau_scale (full structural)
 11. kitchen_sink    — gate + E + tau_scale + int_l2=0.1 + ridge_W=0.1

Usage:
    python -u -m scripts.stage2_loo_sweep_v3 \
        --worms 2022-08-02-01 \
        --out output_plots/stage2/loo_sweep_v3 \
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

def _v5_base() -> dict:
    """v5 defaults: the current best configuration."""
    return dict(
        # Dynamics
        linear_chemical_synapses=False,
        edge_specific_G=False,
        learn_W_sv=True,
        learn_W_dcv=True,
        learn_noise=True,
        noise_mode="heteroscedastic",
        noise_corr_rank=0,
        tau_sv_init=(0.5, 0.85, 1.5, 2.5, 5.0),
        a_sv_init=(2.5, 1.8, 1.2, 0.8, 0.5),
        tau_dcv_init=(2.0, 6.0, 10.0, 15.0, 20.0),
        a_dcv_init=(0.8, 0.6, 0.45, 0.3, 0.2),
        # Training
        rollout_steps=0,
        rollout_weight=0,
        rollout_starts=0,
        warmstart_rollout=False,
        loo_aux_weight=0,
        loo_aux_steps=0,
        loo_aux_neurons=0,
        loo_aux_starts=0,
        # Regularisation (all off in v5)
        interaction_l2=0.0,
        ridge_W_sv=0.0,
        ridge_W_dcv=0.0,
        G_reg=0.0,
        # New features (off by default)
        coupling_gate=False,
        per_neuron_tau_scale=False,
        learn_reversals=False,
        coupling_gate_reg=0.0,
    )


def _make_conditions() -> OrderedDict:
    C = OrderedDict()
    B = _v5_base

    # 0. v5 baseline
    C["v5_base"] = B()

    # --- Regularisation sweep ---
    # 1. Interaction L2 (penalise coupling contribution vs AR1)
    C["int_l2_01"] = {**B(), "interaction_l2": 0.01}
    # 2. Stronger
    C["int_l2_10"] = {**B(), "interaction_l2": 0.1}

    # 3. Ridge on synaptic weights (shrink toward zero)
    C["ridge_W"] = {**B(), "ridge_W_sv": 0.1, "ridge_W_dcv": 0.1}

    # 4. Anchor gap junction conductances to init
    C["G_reg"] = {**B(), "G_reg": 0.1}

    # --- Structural: coupling gate ---
    # 5. Per-neuron coupling gate (neurons learn to accept/reject coupling)
    C["gate"] = {**B(), "coupling_gate": True}

    # 6. Gate + interaction L2 (gate selects, L2 shrinks remaining)
    C["gate_int_l2"] = {
        **B(),
        "coupling_gate": True,
        "interaction_l2": 0.1,
    }

    # --- Structural: learned reversals (conductance-based gating) ---
    # 7. Learn E_sv per neuron: g*(E_i − u_i) → coupling shuts off when E≈u
    C["learn_E"] = {**B(), "learn_reversals": True}

    # 8. Gate + learned reversals (two gating mechanisms)
    C["gate_E"] = {
        **B(),
        "coupling_gate": True,
        "learn_reversals": True,
    }

    # --- Structural: per-neuron tau scale ---
    # 9. Per-neuron tau scale: tau_eff[j,r] = tau[r] * exp(s_j)
    #    Each neuron's presynaptic output has its own temporal signature.
    C["tau_scale"] = {**B(), "per_neuron_tau_scale": True}

    # --- Combinations ---
    # 10. Full structural (gate + E + tau_scale)
    C["gate_E_tau"] = {
        **B(),
        "coupling_gate": True,
        "learn_reversals": True,
        "per_neuron_tau_scale": True,
    }

    # 11. Kitchen sink: structural + regularisation
    C["kitchen_sink"] = {
        **B(),
        "coupling_gate": True,
        "learn_reversals": True,
        "per_neuron_tau_scale": True,
        "interaction_l2": 0.1,
        "ridge_W_sv": 0.1,
        "ridge_W_dcv": 0.1,
    }

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
    kw["behavior_weight"] = 0.0  # skip behaviour for speed
    # Noise always on
    kw.setdefault("learn_noise", True)
    kw.setdefault("noise_mode", "heteroscedastic")
    # Skip expensive eval extras
    kw["make_posture_video"] = False
    kw["n_freerun_samples"] = 0
    kw["n_sample_trajectories"] = 0

    cfg = make_config(worm_h5, **kw)
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

        # Coupling gate statistics (if present)
        gate_vals = result.get("coupling_gate_values")
        if gate_vals is not None:
            summary["gate_mean"] = round(float(gate_vals.mean()), 4)
            summary["gate_min"] = round(float(gate_vals.min()), 4)
            summary["gate_max"] = round(float(gate_vals.max()), 4)
            summary["gate_std"] = round(float(gate_vals.std()), 4)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worms", nargs="+", default=["2022-08-02-01"])
    parser.add_argument("--out", default="output_plots/stage2/loo_sweep_v3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cv_folds", type=int, default=2)
    parser.add_argument("--loo_subset", type=int, default=30,
                        help="Number of neurons for LOO (0=all)")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Run only these conditions (default: all)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of conditions to run in parallel "
                             "(each gets its own process; default=1)")
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
    print(f"\n{'='*72}")
    print(f"Stage-2 LOO Sweep v3 — coupling gate / reversals / tau-scale / reg")
    print(f"  Worms:      {args.worms}")
    print(f"  Conditions: {list(conditions.keys())}")
    print(f"  Total runs: {n_total}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  CV folds:   {args.cv_folds}")
    print(f"  LOO subset: {args.loo_subset}")
    print(f"  Parallel:   {args.parallel}")
    print(f"  Output:     {base_out}")
    print(f"{'='*72}\n")

    all_results = []

    if args.parallel > 1:
        # ---- parallel execution via multiprocessing ----
        import multiprocessing as mp
        from functools import partial

        jobs = []
        for wi, h5 in enumerate(worm_paths):
            for ci, (cname, overrides) in enumerate(conditions.items()):
                jobs.append((h5, cname, overrides, args.worms[wi]))

        def _run_job(job_args):
            h5, cname, overrides, worm_name = job_args
            print(f"  [parallel] starting worm={worm_name} cond={cname}")
            sys.stdout.flush()
            return run_one(
                h5, cname, overrides,
                base_out=base_out,
                epochs=args.epochs,
                device=args.device,
                cv_folds=args.cv_folds,
                loo_subset=args.loo_subset,
            )

        # Use fork: main process hasn't done CUDA work yet, so fork is safe
        # and avoids pickling issues that spawn has with __main__ functions.
        ctx = mp.get_context("fork")
        with ctx.Pool(args.parallel) as pool:
            for summary in pool.imap_unordered(_run_job, jobs):
                all_results.append(summary)
                _print_summary_table(all_results)
                with open(base_out / "sweep_results.json", "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

    else:
        # ---- sequential execution (original) ----
        for wi, h5 in enumerate(worm_paths):
            for ci, (cname, overrides) in enumerate(conditions.items()):
                idx = wi * len(conditions) + ci + 1
                print(f"\n{'#'*72}")
                print(f"  [{idx}/{n_total}] worm={args.worms[wi]}  cond={cname}")
                print(f"{'#'*72}\n")
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

                _print_summary_table(all_results)

                with open(base_out / "sweep_results.json", "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

    # Final
    print(f"\n{'='*72}")
    print(f"FINAL SUMMARY — v3 (coupling gate / reversals / tau-scale / reg)")
    print(f"{'='*72}")
    _print_summary_table(all_results)

    with open(base_out / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {base_out / 'sweep_results.json'}")


def _print_summary_table(results: list[dict]):
    header = (
        f"{'Condition':<18} {'1-step':>7} {'LOO-med':>8} {'LOO-w':>7} "
        f"{'#pos':>5} {'gate':>6} {'time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r.get('condition','?'):<18} ERROR: {r['error'][:45]}")
            continue
        cond = r.get("condition", "?")
        onestep = r.get("cv_onestep_r2_median")
        loo_med = r.get("cv_loo_r2_median")
        loo_w = r.get("cv_loo_r2_windowed_median")
        n_pos = r.get("loo_n_positive", "?")
        gate_m = r.get("gate_mean")
        elapsed = r.get("elapsed_s", 0)
        parts = [f"{cond:<18}"]
        parts.append(f"{onestep:>7.4f}" if onestep is not None else f"{'N/A':>7}")
        parts.append(f"{loo_med:>8.4f}" if loo_med is not None else f"{'N/A':>8}")
        parts.append(f"{loo_w:>7.4f}" if loo_w is not None else f"{'N/A':>7}")
        parts.append(f"{n_pos:>5}")
        parts.append(f"{gate_m:>6.3f}" if gate_m is not None else f"{'—':>6}")
        parts.append(f"{elapsed:>5.0f}s")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
