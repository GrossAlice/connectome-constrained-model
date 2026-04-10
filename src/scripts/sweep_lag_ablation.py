#!/usr/bin/env python
"""Ablation sweep: exclude each I_lag component one at a time.

Baseline = F03_lag10_l2 (lag_order=10, dynamics_l2=1e-3, lag_neighbor_per_type=True).
Conditions:
  A0  baseline      – all I_lag components active
  A1  no_self       – exclude self-lag (α_k)
  A2  no_e          – exclude gap-junction neighbor lag (G_e)
  A3  no_sv         – exclude SV chemical neighbor lag (G_sv)
  A4  no_dcv        – exclude DCV chemical neighbor lag (G_dcv)
  A5  no_nbr        – exclude all three neighbor types (keep self only)
  A6  self_only_e   – exclude self + sv + dcv (gap-junction neighbor only)
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.chdir(SRC)

OUT_ROOT = SRC / "output_plots" / "stage2" / "sweep_lag_ablation"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Baseline config: F03_lag10_l2 ──────────────────────────────────────
BASE = dict(
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

CONDITIONS = {
    "A0_baseline":      dict(),                                   # all active
    "A1_no_self":       dict(lag_exclude_types=("self",)),        # drop self-lag
    "A2_no_e":          dict(lag_exclude_types=("e",)),           # drop gap neighbor
    "A3_no_sv":         dict(lag_exclude_types=("sv",)),          # drop SV neighbor
    "A4_no_dcv":        dict(lag_exclude_types=("dcv",)),         # drop DCV neighbor
    "A5_no_nbr":        dict(lag_exclude_types=("e","sv","dcv")), # self-only
    "A6_nbr_e_only":    dict(lag_exclude_types=("self","sv","dcv")), # gap-nbr only
}

# ── Runner ─────────────────────────────────────────────────────────────
H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

def run_one(name: str, overrides: dict) -> dict:
    from stage2.config import make_config
    from stage2.train import train_stage2_cv
    save_dir = str(OUT_ROOT / name)
    os.makedirs(save_dir, exist_ok=True)
    cfg_kw = {**BASE, **overrides}
    cfg = make_config(H5, **cfg_kw)
    cfg.output.out_u_mean = None
    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=save_dir)
    elapsed = (time.time() - t0) / 60
    diff = {k: v for k, v in overrides.items() if k not in BASE or BASE[k] != v}
    summary = {"condition": name, "overrides": diff, "elapsed_min": round(elapsed, 1)}
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_fr_r2_mean", "cv_fr_r2_median", "best_fold_idx",
              "cv_loo_r2_mean", "cv_loo_r2_median"]:
        if k in results:
            summary[k] = results[k]
    (Path(save_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary

# ── Main loop ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}
    log_path = OUT_ROOT / "run.log"
    for name, ovr in CONDITIONS.items():
        if (OUT_ROOT / name / "summary.json").exists():
            prev = json.loads((OUT_ROOT / name / "summary.json").read_text())
            results[name] = prev
            print(f"[SKIP] {name} already done: LOO={prev.get('cv_loo_r2_mean','?')}")
            continue
        print(f"\n{'='*60}\n  {name}  overrides={ovr}\n{'='*60}", flush=True)
        try:
            s = run_one(name, ovr)
            results[name] = s
            msg = (f"[DONE] {name}: 1step={s.get('cv_onestep_r2_mean','?'):.4f}  "
                   f"LOO={s.get('cv_loo_r2_mean','?'):.4f}  "
                   f"medLOO={s.get('cv_loo_r2_median','?'):.4f}  "
                   f"({s['elapsed_min']:.1f} min)")
            print(msg, flush=True)
        except Exception as e:
            traceback.print_exc()
            results[name] = {"error": str(e)}
    # Save aggregate
    (OUT_ROOT / "all_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print("\n\n=== FINAL SUMMARY ===")
    for n, r in results.items():
        if "error" in r:
            print(f"  {n}: ERROR {r['error']}")
        else:
            print(f"  {n}: 1step={r.get('cv_onestep_r2_mean','?'):.4f}  "
                  f"LOO={r.get('cv_loo_r2_mean','?'):.4f}  "
                  f"medLOO={r.get('cv_loo_r2_median','?'):.4f}")
    print("DONE")
