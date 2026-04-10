#!/usr/bin/env python
"""Ablate T_e from I_lag_nbr to test whether it's justified.

D0  = baseline   : all lag_nbr types active  (e, sv, dcv)  [reference]
D1  = drop T_e   : lag_exclude_types=("e",)  — keep SV+DCV lags
D2  = T_e only   : lag_exclude_types=("sv","dcv")  — keep only T_e lag
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.chdir(SRC)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

BASE_DIR = SRC / "output_plots" / "stage2" / "lag_nbr_Te_ablation"
BASE_DIR.mkdir(parents=True, exist_ok=True)

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

# ── Shared training config (matches repro_A0_baseline) ──
COMMON = dict(
    lag_order=10,
    chemical_synapse_mode="iir",
    chemical_synapse_activation="sigmoid",
    edge_specific_G=True,
    per_neuron_amplitudes=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    graph_poly_order=1,
    learn_reversals=False,
    synapse_lag_taps=0,
    lag_neighbor=True,
    lag_connectome_mask="all",
    lag_neighbor_activation="none",
    lag_neighbor_per_type=True,
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
    "D0_baseline_all_nbr": dict(
        # Full A0 baseline: lag_nbr for e + sv + dcv
        lag_exclude_types=(),
    ),
    "D1_drop_Te_lag": dict(
        # Drop only T_e from lag_nbr, keep SV+DCV
        lag_exclude_types=("e",),
    ),
    "D2_Te_lag_only": dict(
        # Keep only T_e lag, drop SV+DCV lags
        lag_exclude_types=("sv", "dcv"),
    ),
}

from stage2.config import make_config
from stage2.train import train_stage2_cv

all_results = {}
for name, overrides in CONDITIONS.items():
    out_dir = BASE_DIR / name
    if (out_dir / "summary.json").exists():
        print(f"[skip] {name} — already done")
        d = json.loads((out_dir / "summary.json").read_text())
        all_results[name] = d
        continue

    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    cfg_kw = {**COMMON, **overrides, "save_dir": str(out_dir), "h5_path": H5}
    cfg = make_config(**cfg_kw)

    # Save run config
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = {**COMMON, **overrides, "condition": name}
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, default=str))

    t0 = time.time()
    results = train_stage2_cv(cfg)
    elapsed = (time.time() - t0) / 60.0

    summary = {
        "condition": name,
        "elapsed_min": round(elapsed, 1),
        "cv_onestep_r2_mean": results.get("cv_onestep_r2_mean"),
        "cv_onestep_r2_median": results.get("cv_onestep_r2_median"),
        "cv_loo_r2_mean": results.get("cv_loo_r2_mean"),
        "cv_loo_r2_median": results.get("cv_loo_r2_median"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    all_results[name] = summary
    print(f"\n>>> {name}: 1step={summary['cv_onestep_r2_mean']:.4f}  "
          f"LOO={summary['cv_loo_r2_mean']:.4f}  "
          f"med={summary['cv_loo_r2_median']:.4f}  ({elapsed:.1f} min)")

# ── Final comparison ─────────────────────────────────────
print("\n" + "="*70)
print("  T_e ABLATION COMPARISON")
print("="*70)
for name, s in all_results.items():
    print(f"  {name:<30s}  1step={s['cv_onestep_r2_mean']:.4f}  "
          f"LOO={s['cv_loo_r2_mean']:.4f}  med={s['cv_loo_r2_median']:.4f}")

(BASE_DIR / "all_results.json").write_text(
    json.dumps(all_results, indent=2, default=str))
print(f"\nDone. Results in {BASE_DIR}")
