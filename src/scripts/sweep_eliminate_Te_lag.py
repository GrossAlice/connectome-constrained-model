#!/usr/bin/env python
"""Sweep: options to eliminate I_lag,nbr,e while preserving its useful effects.

All conditions use lag_exclude_types=("e",) to drop the T_e lag.
Based on run_300ep_eval25 settings but 60 epochs.

O0_ref         : baseline reference (no exclusion, lag_order=10) — control
O4_just_drop   : lag_exclude_types=("e",) only  — Option 4/6 (simplest)
O1_poly2       : graph_poly_order=2             — Option 1 (multi-hop spatial)
O1b_poly3      : graph_poly_order=3             — Option 1 variant
O2_delayed_lap : gap_lag_order=2                — Option 2 (temporal, L·u(t-2))
O3_ar2_poly2   : lag_order=2 + graph_poly_order=2 — Option 3 (recommended combo)
"""

import sys, os, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from stage2.config import make_config
from stage2.train  import train_stage2_cv

ROOT = pathlib.Path("output_plots/stage2/sweep_eliminate_Te_lag")
H5   = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

CONDITIONS = {
    # ── Control: full baseline (no exclusion) ──
    "O0_ref": dict(
        lag_exclude_types=(),
    ),
    # ── Option 4/6: just drop T_e lag ──
    "O4_just_drop": dict(
        lag_exclude_types=("e",),
    ),
    # ── Option 1: multi-hop Laplacian (spatial) ──
    "O1_poly2": dict(
        lag_exclude_types=("e",),
        graph_poly_order=2,
    ),
    "O1b_poly3": dict(
        lag_exclude_types=("e",),
        graph_poly_order=3,
    ),
    # ── Option 2: delayed Laplacian (temporal) ──
    "O2_delayed_lap": dict(
        lag_exclude_types=("e",),
        gap_lag_order=2,
    ),
    # ── Option 3: AR(2) self + poly2 (recommended) ──
    "O3_ar2_poly2": dict(
        lag_exclude_types=("e",),
        lag_order=2,
        graph_poly_order=2,
    ),
}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, overrides in CONDITIONS.items():
        save_dir = ROOT / name
        save_dir.mkdir(parents=True, exist_ok=True)

        cfg = make_config(H5, num_epochs=60, eval_interval=20, **overrides)
        cfg.output.out_u_mean = None

        print(f"\n{'='*60}")
        print(f"=== {name} ===")
        print(f"  overrides: {overrides}")
        print(f"{'='*60}")

        t0 = time.time()
        results = train_stage2_cv(cfg, save_dir=str(save_dir))
        elapsed = (time.time() - t0) / 60

        summary = {"elapsed_min": round(elapsed, 1)}
        for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
                   "cv_fr_r2_mean", "cv_fr_r2_median", "best_fold_idx",
                   "cv_loo_r2_mean", "cv_loo_r2_median"]:
            if k in results:
                summary[k] = results[k]

        (save_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str))

        loo  = summary.get("cv_loo_r2_mean", "?")
        one  = summary.get("cv_onestep_r2_mean", "?")
        print(f"  => 1-step R²={one}  LOO R²={loo}  ({elapsed:.1f} min)")
        all_results[name] = summary

    (ROOT / "all_results.json").write_text(
        json.dumps(all_results, indent=2, default=str))

    print(f"\n{'='*60}")
    print("SUMMARY — Options to eliminate I_lag,nbr,e")
    print(f"{'='*60}")
    for name, s in all_results.items():
        print(f"  {name:20s}  1step={s.get('cv_onestep_r2_mean','?'):.4f}"
              f"  LOO={s.get('cv_loo_r2_mean','?'):.4f}"
              f"  med={s.get('cv_loo_r2_median','?'):.4f}")

if __name__ == "__main__":
    main()
