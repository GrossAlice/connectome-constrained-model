#!/usr/bin/env python
"""Run Stage2 with correlation-weighted init + regularisation.

Three conditions:
  A) corr_weighted init only  (G + W init, no reg)
  B) corr_weighted init + corr_reg_weight=0.01
  C) baseline (uniform init, no corr_reg) for comparison

All use edge_specific_G=True, per_neuron_amplitudes=True, 30 epochs, 3-fold CV.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/run1")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

COMMON = dict(
    num_epochs=30,
    cv_folds=3,
    seed=42
)

CONDITIONS = {
    "C0_baseline": dict(

        corr_reg_weight=0.0,
    )
}

log = OUT_ROOT / "run.log"

def tee(msg):
    print(msg, flush=True)
    with open(log, "a") as f:
        f.write(msg + "\n")

tee(f"=== Correlation-weighted sweep  ({len(CONDITIONS)} conditions) ===")
tee(f"Common: epochs={COMMON['num_epochs']}, folds={COMMON['cv_folds']}, "
    f"edge_specific_G=True, pn_amp=True")

for name, overrides in CONDITIONS.items():
    save_dir = OUT_ROOT / name
    summary_path = save_dir / "summary.json"

    if summary_path.exists():
        tee(f"\n>>> SKIP {name} (summary.json exists)")
        continue

    tee(f"\n{'='*60}")
    tee(f">>> {name}")
    tee(f"    overrides: {overrides}")
    tee(f"{'='*60}")

    kw = {**COMMON, **overrides}
    cfg = make_config(H5, **kw)
    t0 = time.time()

    try:
        result = train_stage2(cfg, save_dir=str(save_dir), show=False)
        elapsed = time.time() - t0
        tee(f">>> {name} DONE in {elapsed/60:.1f} min")

        # Save summary
        summary = {"condition": name, "overrides": overrides,
                   "elapsed_min": round(elapsed / 60, 1)}
        if result and isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (int, float, str, bool)):
                    summary[k] = v
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        tee(f">>> Saved {summary_path}")

    except Exception as e:
        elapsed = time.time() - t0
        tee(f">>> {name} FAILED after {elapsed/60:.1f} min: {e}")
        import traceback
        tee(traceback.format_exc())

tee(f"\n=== All conditions complete ===")
