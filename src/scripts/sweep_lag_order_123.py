#!/usr/bin/env python
"""Sweep lag_order ∈ {1, 2, 3} with lag_exclude_types=("e",), 60 epochs.
Based on run_300ep_eval25 settings."""

import sys, os, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from stage2.config import make_config
from stage2.train  import train_stage2_cv

ROOT = pathlib.Path("output_plots/stage2/sweep_lag_order_123")
H5   = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

CONDITIONS = {
    "L1_lag1": dict(lag_order=1),
    "L2_lag2": dict(lag_order=2),
    "L3_lag3": dict(lag_order=3),
}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, overrides in CONDITIONS.items():
        save_dir = ROOT / name
        save_dir.mkdir(parents=True, exist_ok=True)

        cfg = make_config(
            H5,
            num_epochs=60,
            eval_interval=20,
            lag_exclude_types=("e",),
            **overrides,
        )
        cfg.output.out_u_mean = None

        print(f"\n{'='*60}")
        print(f"=== {name}  |  lag_order={overrides['lag_order']}  |  drop-Te  |  60ep ===")
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

    # aggregate
    (ROOT / "all_results.json").write_text(
        json.dumps(all_results, indent=2, default=str))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, s in all_results.items():
        print(f"  {name:20s}  1step={s.get('cv_onestep_r2_mean','?'):.4f}"
              f"  LOO={s.get('cv_loo_r2_mean','?'):.4f}"
              f"  med={s.get('cv_loo_r2_median','?'):.4f}")

if __name__ == "__main__":
    main()
