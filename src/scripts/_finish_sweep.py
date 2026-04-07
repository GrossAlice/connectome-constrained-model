#!/usr/bin/env python
"""
Finish the LOO improvement sweep by training only the poly2 variant
(which was interrupted) and assembling all results + final plot.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import h5py, numpy as np

import matplotlib
matplotlib.use("Agg")

from stage2.config import make_config
from stage2.train import train_stage2_cv

# Re-use the plotting function from the sweep script
from scripts.sweep_loo_improvements import plot_comparison

# ── paths ──────────────────────────────────────────────────────────
H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE = Path("output_plots/stage2/loo_improvement_sweep")
BASELINE_JSON = Path("output_plots/stage2/model_distributions/results.json")
STAGE2_NPZ = Path("output_plots/stage2/default_config_run_v2/cv_onestep.npz")
EPOCHS = 30
DEVICE = "cuda"

# ── load data ──────────────────────────────────────────────────────
with h5py.File(H5, "r") as f:
    u = f["stage1/u_mean"][:]
T, N = u.shape
print(f"  {Path(H5).stem}  T={T}  N={N}")

var = np.var(u, axis=0)
loo_neurons = sorted(np.argsort(var)[::-1][:30].tolist())

all_results = {}
baseline_order = []
stage2_order = []

# ── baselines ──────────────────────────────────────────────────────
with open(BASELINE_JSON) as f:
    prev = json.load(f)
for bname in ["EN", "MLP", "Conn-Ridge"]:
    if bname in prev:
        all_results[bname] = {
            "onestep_r2": np.array(prev[bname]["onestep_r2"]),
            "loo_r2_w": np.array(prev[bname]["loo_r2_w"]),
        }
        baseline_order.append(bname)
print(f"  Loaded baselines: {baseline_order}")

# ── Stage2 baseline ───────────────────────────────────────────────
s2 = np.load(STAGE2_NPZ, allow_pickle=True)
s2_os = s2["cv_r2"]
s2_loo = s2.get("cv_loo_r2_windowed", np.full(N, np.nan))
all_results["Stage2"] = {"onestep_r2": s2_os, "loo_r2_w": s2_loo[loo_neurons]}
stage2_order.append("Stage2")
print(f"  Stage2 baseline: 1step={np.nanmean(s2_os):.4f}  "
      f"LOO_w={np.nanmean(s2_loo[loo_neurons]):.4f}")

# ── Load already-completed variants from their cv_onestep.npz ────
for label, subdir in [("+ loo_aux", "loo_aux"),
                       ("+ lowrank", "lowrank"),
                       ("+ loo+lr", "loolr")]:
    npz = SAVE / subdir / "cv_onestep.npz"
    if not npz.exists():
        print(f"  WARNING: {npz} missing — skipping {label}")
        continue
    d = np.load(npz, allow_pickle=True)
    os_r2 = d["cv_r2"]
    loo_r2 = d.get("cv_loo_r2_windowed", np.full(N, np.nan))
    full_label = f"Stage2 {label}"
    all_results[full_label] = {"onestep_r2": os_r2, "loo_r2_w": loo_r2[loo_neurons]}
    stage2_order.append(full_label)
    print(f"  Loaded {full_label}: 1step={np.nanmean(os_r2):.4f}  "
          f"LOO_w={np.nanmean(loo_r2[loo_neurons]):.4f}")

# ── Train poly2 (the interrupted variant) ────────────────────────
print(f"\n{'─'*60}")
print(f"  Training: Stage2 + poly2  ({EPOCHS} epochs)")
print(f"{'─'*60}")

common = dict(
    device=DEVICE, num_epochs=EPOCHS, learning_rate=1e-3,
    cv_folds=2, eval_loo_subset_size=30, eval_loo_subset_mode="variance",
    skip_final_eval=True,
)
cfg = make_config(H5, **common, graph_poly_order=2)
vdir = str(SAVE / "poly2")
t0 = time.time()
res = train_stage2_cv(cfg, save_dir=vdir, show=False)
elapsed = time.time() - t0

os_r2 = res.get("cv_onestep_r2", np.full(N, np.nan))
loo_r2 = res.get("cv_loo_r2_windowed", np.full(N, np.nan))
full_label = "Stage2 + poly2"
all_results[full_label] = {"onestep_r2": os_r2, "loo_r2_w": loo_r2[loo_neurons]}
stage2_order.append(full_label)
print(f"  → {full_label}  1step={np.nanmean(os_r2):.4f}  "
      f"LOO_w={np.nanmean(loo_r2[loo_neurons]):.4f}  ({elapsed:.0f}s)")

# ── Summary table ────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  {'Model':25s}  {'1step mean':>10s}  {'LOO_w mean':>10s}  "
      f"{'LOO_w med':>10s}")
print(f"  {'─'*60}")
for name in baseline_order + stage2_order:
    r = all_results[name]
    os_v = r["onestep_r2"]
    loo_v = r["loo_r2_w"]
    print(f"  {name:25s}  {np.nanmean(os_v):10.4f}  "
          f"{np.nanmean(loo_v):10.4f}  {np.nanmedian(loo_v):10.4f}")
print(f"{'═'*65}")

# ── Save JSON ────────────────────────────────────────────────────
json_out = {}
for name, r in all_results.items():
    json_out[name] = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in r.items()
    }
json_out["_meta"] = {
    "worm": Path(H5).stem, "h5": H5, "epochs": EPOCHS,
    "loo_neurons": loo_neurons, "n_loo": len(loo_neurons),
    "baseline_order": baseline_order, "stage2_order": stage2_order,
}
with open(SAVE / "sweep_results.json", "w") as f:
    json.dump(json_out, f, indent=2)
print(f"  Saved {SAVE / 'sweep_results.json'}")

# ── Plot ─────────────────────────────────────────────────────────
plot_comparison(
    all_results, SAVE / "loo_sweep_comparison.png",
    Path(H5).stem, len(loo_neurons), stage2_order, baseline_order,
)
