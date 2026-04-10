#!/usr/bin/env python
"""
Stage2 ODE — exact parameter equivalent of U3_full_lag5 (union-ridge).

U3_full_lag5 (Ridge) feature set per neuron i:
  u_i(t)                                 → β_self   (1 feature)
  u_i(t-1), …, u_i(t-5)                 → β_lag    (5 features)
  u_j(t)   for j ∈ N(i) = union mask     → β_nbr    (|N(i)| features)
  u_j(t-k) for j ∈ N(i), k=1…5          → β_nbr_lag (5·|N(i)| features)
  intercept                               → b_i

Stage2 ODE mapping:
  u_i(t)           →  λ_i · u_i(t)     [leak; λ via sigmoid ∈ (0,1)]
  u_i(t-1..t-5)   →  I_lag self        [α_ik, free sign]  ← MATCHES
  u_j(t) union     →  I_gap + I_sv + I_dcv at t=0
                       (but: I_gap uses T_e Laplacian with G∈[0,2];
                        I_sv/I_dcv use sigmoid activation, IIR,
                        softplus W ≥ 0 — all EXTRA constraints)
  u_j(t-k) union   →  I_lag neighbor   [G_ij_k, free sign, union mask]  ← MATCHES
  intercept        →  I_0              [free]  ← MATCHES

Irreducible structural extras in Stage2 vs Ridge:
  • IIR chemical synapses (I_sv, I_dcv) — can't be fully disabled without
    losing t=0 neighbor coupling on T_sv/T_dcv edges
  • Gap junction Laplacian structure: G_ij·(u_j−u_i), not free β_ij·u_j
  • Softplus/sigmoid weight constraints (W≥0, G∈[0,2])
  • Shared dynamics_l2 (not per-neuron RidgeCV α)
  • SGD training (not closed-form)

Parameters explicitly matched to U3:
  lag_order=5               same lag depth
  lag_neighbor=True         include neighbor lag
  lag_connectome_mask="all" union T_e∪T_sv∪T_dcv
  lag_neighbor_per_type=False    single union mask (not per-type)
  lag_neighbor_activation="none" linear (no φ before weighting)
  rollout_steps=0           no multi-step rollout (U3 is 1-step only)
  rollout_weight=0.0        ↑
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
SAVE = SRC / "output_plots" / "stage2" / "union_ridge_ode" / "S2_equiv_lag5"

cfg = make_config(
    H5,
    # ── I_lag: exact match to U3 ──────────────────────────────────
    lag_order            = 5,          # K=5 lag taps (t-1 … t-5)
    lag_neighbor         = True,       # include neighbor lags
    lag_connectome_mask  = "all",      # union mask: T_e ∪ T_sv ∪ T_dcv
    lag_neighbor_per_type = False,     # single union set of weights (NOT per-type)
    lag_neighbor_activation = "none",  # linear (no sigmoid/softplus on u_j)

    # ── Biophysical channels (provide t=0 neighbor coupling) ──────
    # These are structurally extra vs Ridge; kept because U3 *does*
    # have u_j(t) features — these are the Stage2 equivalent.
    use_gap_junctions    = True,       # I_gap on T_e edges
    use_sv_synapses      = True,       # I_sv  on T_sv edges (IIR + sigmoid)
    use_dcv_synapses     = True,       # I_dcv on T_dcv edges (IIR + sigmoid)
    chemical_synapse_mode = "iir",     # exponential IIR filter
    chemical_synapse_activation = "sigmoid",  # φ(u_j) = sigmoid
    edge_specific_G      = True,      # per-edge gap conductance G_ij
    per_neuron_amplitudes = True,      # per-neuron a_sv, a_dcv
    G_init_mode          = "corr_weighted",
    W_init_mode          = "corr_weighted",
    graph_poly_order     = 1,         # standard Laplacian, no higher-order hops
    learn_reversals      = False,
    synapse_lag_taps     = 0,

    # ── Training ──────────────────────────────────────────────────
    num_epochs           = 60,
    cv_folds             = 3,         # 3-fold temporal CV (same as U3)
    dynamics_l2          = 1e-3,      # shared L2 (vs per-neuron α in Ridge)
    seed                 = 42,
    synaptic_lr_multiplier = 5.0,

    # ── NO rollout (U3 is 1-step-ahead only) ─────────────────────
    rollout_steps        = 0,
    rollout_weight       = 0.0,
    rollout_starts       = 0,

    skip_free_run        = True,
)

SAVE.mkdir(parents=True, exist_ok=True)
t0 = time.time()
results = train_stage2_cv(cfg, save_dir=str(SAVE))
elapsed = (time.time() - t0) / 60

summary = {
    "condition": "S2_equiv_lag5",
    "elapsed_min": round(elapsed, 1),
}
for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
          "cv_loo_r2_mean", "cv_loo_r2_median", "best_fold_idx"]:
    if k in results:
        summary[k] = results[k]

(SAVE / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\n{'='*60}")
print(f"S2_equiv_lag5:  1step={summary.get('cv_onestep_r2_mean','?'):.4f}  "
      f"LOO={summary.get('cv_loo_r2_mean','?'):.4f}  "
      f"med={summary.get('cv_loo_r2_median','?'):.4f}  "
      f"({elapsed:.1f} min)")
print()
print("Reference baselines:")
print(f"  U3_full_lag5 (Ridge, union, free sign):  LOO = 0.4802")
print(f"  D02 (Stage2 ODE, per_type, rollout):     LOO = 0.4200")
print(f"  Conn-Ridge (K=5, Adam, retrain_loo.py):  LOO = 0.4750")
print(f"{'='*60}")
