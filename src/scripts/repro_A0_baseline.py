#!/usr/bin/env python
"""Exact reproduction of A0_baseline from sweep_lag_ablation (LOO=0.4705)."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.chdir(SRC)

SAVE_DIR = SRC / "output_plots" / "stage2" / "repro_A0_baseline"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

# ── Exact A0_baseline config (no overrides) ──
CFG_KW = dict(
    lag_order=10,
    lag_neighbor=True,
    lag_connectome_mask="all",
    lag_neighbor_activation="none",
    lag_neighbor_per_type=True,
    # D02 architecture
    chemical_synapse_mode="iir",
    chemical_synapse_activation="sigmoid",
    edge_specific_G=True,
    per_neuron_amplitudes=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    graph_poly_order=1,
    learn_reversals=False,
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

from stage2.config import make_config
from stage2.train import train_stage2_cv

cfg = make_config(H5, **CFG_KW)
cfg.output.out_u_mean = None

save = str(SAVE_DIR)
t0 = time.time()
results = train_stage2_cv(cfg, save_dir=save)
elapsed = (time.time() - t0) / 60

summary = {"condition": "repro_A0_baseline", "elapsed_min": round(elapsed, 1)}
for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
          "cv_fr_r2_mean", "cv_fr_r2_median", "best_fold_idx",
          "cv_loo_r2_mean", "cv_loo_r2_median"]:
    if k in results:
        summary[k] = results[k]

(SAVE_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\n=== RESULT: LOO={summary.get('cv_loo_r2_mean','?'):.4f}  "
      f"medLOO={summary.get('cv_loo_r2_median','?'):.4f}  "
      f"1step={summary.get('cv_onestep_r2_mean','?'):.4f}  "
      f"({elapsed:.1f} min) ===")
