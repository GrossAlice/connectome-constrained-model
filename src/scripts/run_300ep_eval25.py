#!/usr/bin/env python
"""Run default config (A0_baseline settings) for 300 epochs with LOO eval every 25."""

import sys, os, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from stage2.config import make_config
from stage2.train  import train_stage2_cv

SAVE_DIR = pathlib.Path("output_plots/stage2/run_300ep_eval25")
H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

def main():
    cfg = make_config(
        H5,
        num_epochs=300,
        eval_interval=25,
    )
    cfg.output.out_u_mean = None

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 300-epoch run  |  eval every 25 epochs ===")
    print(f"Save dir: {SAVE_DIR}")

    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=str(SAVE_DIR))
    elapsed = (time.time() - t0) / 60
    print(f"\n=== DONE  ({elapsed:.1f} min) ===")

    # save results
    summary = {"elapsed_min": round(elapsed, 1)}
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_fr_r2_mean", "cv_fr_r2_median", "best_fold_idx",
              "cv_loo_r2_mean", "cv_loo_r2_median"]:
        if k in results:
            summary[k] = results[k]

    (SAVE_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    loo = summary.get("cv_loo_r2_mean", "?")
    one = summary.get("cv_onestep_r2_mean", "?")
    print(f"1-step R²: {one}")
    print(f"LOO   R²:  {loo}")

if __name__ == "__main__":
    main()
