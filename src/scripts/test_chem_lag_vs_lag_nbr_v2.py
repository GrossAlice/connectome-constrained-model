#!/usr/bin/env python
"""Sweep v2: companion to test_chem_lag_vs_lag_nbr.

C0 = A0 baseline but lag_order=2 (minimal self-lag)
C1 = chem_lag (SV only), keep lag_nbr for T_e + T_dcv, softplus activation
C2 = chem_lag (SV+DCV) + all lag_nbr + learnable reversal potential
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.chdir(SRC)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

BASE_DIR = SRC / "output_plots" / "stage2" / "chem_lag_vs_lag_nbr_v2"
BASE_DIR.mkdir(parents=True, exist_ok=True)

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

# ── Shared training config (matches repro_A0_baseline) ──
COMMON = dict(
    edge_specific_G=True,
    per_neuron_amplitudes=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    graph_poly_order=1,
    synapse_lag_taps=0,
    # Training
    num_epochs=60,
    cv_folds=3,
    dynamics_l2=1e-3,
    coupling_dropout=0.0,
    input_noise_sigma=0.0,
    seed=42,
    rollout_steps=10,
    rollout_weight=0.3,
    rollout_starts=8,
    synaptic_lr_multiplier=5.0,
    skip_free_run=True,
)

CONDITIONS = {
    "C0_A0_lag2": dict(
        # A0 baseline but with self-lag order = 2 (minimal AR)
        chemical_synapse_mode="iir",
        chemical_synapse_activation="sigmoid",
        lag_order=2,
        lag_neighbor=True,
        lag_connectome_mask="all",
        lag_neighbor_activation="none",
        lag_neighbor_per_type=True,
        learn_reversals=False,
    ),
    "C1_chem_lag_sv_softplus": dict(
        # chem_lag replaces SV only (with softplus activation),
        # keep lag_nbr for T_e and T_dcv (exclude SV from lag_nbr).
        # IIR is off entirely → DCV is handled by lag_nbr _lag_G_dcv.
        chemical_synapse_mode="lag",
        chemical_synapse_activation="softplus",
        lag_order=10,
        lag_neighbor=True,
        lag_exclude_types=("sv",),      # keep T_e + T_dcv in lag_nbr
        lag_connectome_mask="all",
        lag_neighbor_activation="none",
        lag_neighbor_per_type=True,
        learn_reversals=False,
    ),
    "C2_chem_lag_reversal": dict(
        # chem_lag (SV+DCV) + full lag_nbr + learnable reversal potential.
        # Smart init: per_neuron E_sv initialized via sign_t.
        chemical_synapse_mode="lag",
        chemical_synapse_activation="sigmoid",
        chem_lag_include_reversal=True,
        lag_order=10,
        lag_neighbor=True,
        lag_connectome_mask="all",
        lag_neighbor_activation="none",
        lag_neighbor_per_type=True,
        learn_reversals=True,
        reversal_mode="per_neuron",
    ),
}

from stage2.config import make_config
from stage2.train import train_stage2_cv

all_results = {}
for name, overrides in CONDITIONS.items():
    save_dir = BASE_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    kw = {**COMMON, **overrides}
    cfg = make_config(H5, **kw)
    cfg.output.out_u_mean = None

    print(f"\n{'='*60}")
    print(f"  CONDITION: {name}")
    for k in sorted(overrides):
        print(f"    {k} = {overrides[k]}")
    print(f"{'='*60}\n")

    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=str(save_dir))
    elapsed = (time.time() - t0) / 60

    summary = {"condition": name, "elapsed_min": round(elapsed, 1)}
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_fr_r2_mean", "cv_fr_r2_median", "best_fold_idx",
              "cv_loo_r2_mean", "cv_loo_r2_median"]:
        if k in results:
            summary[k] = results[k]

    (save_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    all_results[name] = summary
    print(f"\n>>> {name}: LOO={summary.get('cv_loo_r2_mean','?'):.4f}  "
          f"medLOO={summary.get('cv_loo_r2_median','?'):.4f}  "
          f"1step={summary.get('cv_onestep_r2_mean','?'):.4f}  "
          f"({elapsed:.1f} min)")

# ── Final comparison ──
print(f"\n{'='*70}")
print(f"{'Condition':<30s} {'1-step R²':>10s} {'LOO R²':>10s} {'med LOO':>10s}")
print(f"{'-'*70}")
for name, s in all_results.items():
    print(f"{name:<30s} {s.get('cv_onestep_r2_mean',0):10.4f} "
          f"{s.get('cv_loo_r2_mean',0):10.4f} {s.get('cv_loo_r2_median',0):10.4f}")
print(f"{'='*70}")

(BASE_DIR / "all_results.json").write_text(
    json.dumps(all_results, indent=2, default=str))
print(f"\nResults saved to {BASE_DIR}")
