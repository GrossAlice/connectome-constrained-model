#!/usr/bin/env python
"""Quick comparison: IIR vs lag chemical synapse mode.

Full config (20 epochs, 2 folds, with LOO + free-run).

Conditions:
  T0  IIR baseline   (current defaults)
  T1  lag mode       (lag-style FIR for chemical synapses, lag_neighbor for T_e only)
  T2  lag mode       (same, but disable separate lag_neighbor entirely)
  T3  lag mode       (lag + sigmoid activation on lag neighbors, no separate IIR)
"""
import json, os, sys, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/test_chem_lag_mode")

_COMMON = dict(
    # Light training
    num_epochs=20,
    cv_folds=2,
    parallel_folds=True,
    rollout_steps=10,
    rollout_weight=0.3,
    rollout_starts=8,
    input_noise_sigma=0.02,
    dynamics_l2=1e-3,
    coupling_dropout=0.05,
    # Standard features
    edge_specific_G=True,
    per_neuron_amplitudes=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    graph_poly_order=1,
    lag_order=5,
    seed=42,
)

CONDITIONS = {
    "T0_iir_baseline": dict(
        chemical_synapse_mode="iir",
        lag_neighbor=True,
        lag_neighbor_per_type=True,
        lag_connectome_mask="all",
        lag_neighbor_activation="none",
    ),
    "T1_lag_lagNbr_Te_only": dict(
        chemical_synapse_mode="lag",
        chem_lag_kernel_len=0,   # use lag_order=5
        lag_neighbor=True,
        lag_neighbor_per_type=True,
        lag_connectome_mask="T_e",      # lag neighbor only on gap junctions
        lag_neighbor_activation="none",
    ),
    "T2_lag_no_lagNbr": dict(
        chemical_synapse_mode="lag",
        chem_lag_kernel_len=0,
        lag_neighbor=False,             # no separate lag neighbor at all
    ),
    "T3_lag_lagNbr_Te_sigmoid": dict(
        chemical_synapse_mode="lag",
        chem_lag_kernel_len=0,
        lag_neighbor=True,
        lag_neighbor_per_type=True,
        lag_connectome_mask="T_e",
        lag_neighbor_activation="sigmoid",
    ),
    "T4_iir_no_lagNbr": dict(
        chemical_synapse_mode="iir",
        lag_neighbor=False,             # IIR I_sv, no lag neighbor (ablation control for T0)
    ),
}

results = {}
for label, overrides in CONDITIONS.items():
    out = OUT_ROOT / label
    if out.exists():
        # Check if already completed
        summary_f = out / "summary.json"
        if summary_f.exists():
            print(f"\n{'='*60}\n  SKIP {label} (already done)\n{'='*60}")
            results[label] = json.loads(summary_f.read_text())
            continue
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    kw = {**_COMMON, **overrides}
    cfg = make_config(H5, **kw)
    cfg.output.out_u_mean = None        # don't write u_mean to HDF5

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  chemical_synapse_mode = {cfg.chemical_synapse_mode}")
    print(f"  lag_neighbor = {cfg.lag_neighbor}")
    print(f"{'='*60}\n")

    t0 = time.time()
    summary = train_stage2(cfg, save_dir=str(out))
    elapsed = time.time() - t0

    summary["elapsed_s"] = round(elapsed, 1)
    results[label] = summary
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  ✓ {label} done in {elapsed:.0f}s")

# ── Print comparison table ──────────────────────────────────────────
print("\n" + "="*80)
print(f"{'Condition':<30} {'1step_mean':>10} {'1step_med':>10} {'time_s':>8}")
print("-"*80)
for label, d in results.items():
    r1 = d.get("cv_onestep_r2_mean", "?")
    r1m = d.get("cv_onestep_r2_median", "?")
    t = d.get("elapsed_s", "?")
    r1s = f"{r1:.4f}" if isinstance(r1, float) else str(r1)
    r1ms = f"{r1m:.4f}" if isinstance(r1m, float) else str(r1m)
    ts = f"{t:.0f}" if isinstance(t, (int, float)) else str(t)
    print(f"{label:<30} {r1s:>10} {r1ms:>10} {ts:>8}")
print("="*80)

# Save all
(OUT_ROOT / "all_results.json").write_text(
    json.dumps(results, indent=2, default=str))
print(f"\nResults saved to {OUT_ROOT}/all_results.json")
