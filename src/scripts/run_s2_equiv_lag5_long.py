#!/usr/bin/env python
"""
Stage2 ODE equiv of U3_full_lag5 — LONGER training.

Previous run (60 epochs): LOO = 0.3996
Target (Ridge U3):        LOO = 0.4802

This script runs 200 epochs to see if more training closes the gap.
Also tries a small grid of dynamics_l2 since Ridge uses per-neuron α.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.chdir(SRC)

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = SRC / "output_plots" / "stage2" / "union_ridge_ode"

CONDITIONS = {
    # more epochs, same l2
    "S2_200ep": dict(num_epochs=200, dynamics_l2=1e-3),
    # more epochs, lower l2
    "S2_200ep_l2e4": dict(num_epochs=200, dynamics_l2=1e-4),
    # more epochs, higher l2
    "S2_200ep_l2e2": dict(num_epochs=200, dynamics_l2=1e-2),
}


def run_one(name: str, num_epochs: int, dynamics_l2: float):
    save_dir = SAVE_ROOT / name
    summary_path = save_dir / "summary.json"
    if summary_path.exists():
        prev = json.loads(summary_path.read_text())
        print(f"\n[SKIP] {name} already done: LOO={prev.get('cv_loo_r2_mean','?')}")
        return prev

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[{name}]  epochs={num_epochs}  l2={dynamics_l2}")
    print(f"{'='*60}")

    cfg = make_config(
        H5,
        lag_order            = 5,
        lag_neighbor         = True,
        lag_connectome_mask  = "all",
        lag_neighbor_per_type = False,
        lag_neighbor_activation = "none",
        use_gap_junctions    = True,
        use_sv_synapses      = True,
        use_dcv_synapses     = True,
        chemical_synapse_mode = "iir",
        chemical_synapse_activation = "sigmoid",
        edge_specific_G      = True,
        per_neuron_amplitudes = True,
        G_init_mode          = "corr_weighted",
        W_init_mode          = "corr_weighted",
        graph_poly_order     = 1,
        learn_reversals      = False,
        synapse_lag_taps     = 0,
        num_epochs           = num_epochs,
        cv_folds             = 3,
        dynamics_l2          = dynamics_l2,
        seed                 = 42,
        synaptic_lr_multiplier = 5.0,
        rollout_steps        = 0,
        rollout_weight       = 0.0,
        rollout_starts       = 0,
        skip_free_run        = True,
    )

    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=str(save_dir))
    elapsed = (time.time() - t0) / 60

    summary = {
        "condition": name,
        "elapsed_min": round(elapsed, 1),
        "num_epochs": num_epochs,
        "dynamics_l2": dynamics_l2,
    }
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_loo_r2_mean", "cv_loo_r2_median", "best_fold_idx"]:
        if k in results:
            summary[k] = results[k]

    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    all_results = {}
    for name, kw in CONDITIONS.items():
        try:
            r = run_one(name, **kw)
            all_results[name] = r
        except Exception:
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": "failed"}

    # ── summary table ─────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"[SUMMARY]  S2_equiv long-training sweep")
    print(f"{'='*60}")
    print(f"{'Condition':<22s}  {'epochs':>6s}  {'l2':>8s}  {'1step':>8s}  {'LOO':>8s}  {'LOO med':>8s}")
    print(f"{'-'*22}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    # include original 60ep result
    orig_path = SAVE_ROOT / "S2_equiv_lag5" / "summary.json"
    if orig_path.exists():
        orig = json.loads(orig_path.read_text())
        print(f"{'S2_equiv_lag5 (ref)':<22s}  {'60':>6s}  {'1e-3':>8s}  "
              f"{orig.get('cv_onestep_r2_mean',0):.4f}  "
              f"{orig.get('cv_loo_r2_mean',0):.4f}  "
              f"{orig.get('cv_loo_r2_median',0):.4f}")

    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<22s}  {'ERROR':>6s}")
            continue
        print(f"{name:<22s}  {r.get('num_epochs','?'):>6}  {r.get('dynamics_l2','?'):>8}  "
              f"{r.get('cv_onestep_r2_mean',0):.4f}  "
              f"{r.get('cv_loo_r2_mean',0):.4f}  "
              f"{r.get('cv_loo_r2_median',0):.4f}")

    print(f"\nReference:  U3_full_lag5 (Ridge): LOO = 0.4802")
    print(f"{'='*60}")

    (SAVE_ROOT / "long_training_results.json").write_text(
        json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
