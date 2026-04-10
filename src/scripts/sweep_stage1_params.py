#!/usr/bin/env python
"""Sweep Stage1 EM parameters on RIFL neuron and produce a comparison grid."""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.config import Stage1Config
from stage1.em import fit_stage1_all_neurons
from stage1.run_stage1 import (
    load_traces_and_regressor,
    _load_neuron_labels,
    _apply_neuron_mask,
)

# ── Settings ────────────────────────────────────────────────────────────────
H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01 (copy).h5"
MASK = "/tmp/rifl_mask.npy"
SAVE_DIR = "output_plots/stage1/rifl-sweep"

# Parameter grid — 3 axes
TAU_U_CLIPS = [
    (0.3, 1.0),
    (0.6, 2.0),
    (1.0, 3.0),
    (1.5, 5.0),
]

SIGMA_U_SCALES = [0.3, 0.7, 1.0]

SIGMA_Y_FLOOR_FRACS = [0.85, 0.95]

# ── Run sweep ───────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)

# Build all conditions
conditions = []
for tau_clip, su_scale, sy_frac in itertools.product(
    TAU_U_CLIPS, SIGMA_U_SCALES, SIGMA_Y_FLOOR_FRACS
):
    label = f"tau_u={tau_clip}  σu_s={su_scale}  σy_f={sy_frac}"
    conditions.append(dict(
        tau_u_clip_sec=tau_clip,
        sigma_u_scale_init=su_scale,
        sigma_y_floor_frac=sy_frac,
        label=label,
    ))

print(f"Running {len(conditions)} conditions on RIFL neuron...\n")

# Load traces once
base_cfg = Stage1Config(h5_path=H5, neuron_mask=MASK, overwrite=True)
X_raw = load_traces_and_regressor(base_cfg)
labels = _load_neuron_labels(base_cfg, n_neurons=X_raw.shape[1])
X, labels = _apply_neuron_mask(base_cfg, X_raw, labels)
T = X.shape[0]
dt = 1.0 / base_cfg.sample_rate_hz
time_axis = np.arange(T) * dt

results = []
for i, cond in enumerate(conditions):
    t0 = time.time()
    cfg = Stage1Config(
        h5_path=H5,
        neuron_mask=MASK,
        overwrite=True,
        save_dir=SAVE_DIR,
        tau_u_clip_sec=cond["tau_u_clip_sec"],
        sigma_u_scale_init=cond["sigma_u_scale_init"],
        sigma_y_floor_frac=cond["sigma_y_floor_frac"],
    )
    try:
        out = fit_stage1_all_neurons(X, cfg)
        u_mean = np.asarray(out["u_mean"])[:, 0]
        c_mean = np.asarray(out["c_mean"])[:, 0]
        rho = float(out["rho"][0])
        sigma_u = float(out["sigma_u"][0])
        sigma_y = float(out["sigma_y"][0])
        lam = float(out["lambda_c"])
        ll = float(out["ll_history"][-1])
        n_iter = len(out["ll_history"])
        elapsed = time.time() - t0

        rec = dict(
            idx=i,
            label=cond["label"],
            tau_u_clip=cond["tau_u_clip_sec"],
            sigma_u_scale=cond["sigma_u_scale_init"],
            sigma_y_frac=cond["sigma_y_floor_frac"],
            rho=rho,
            tau_u=round(-dt / np.log(max(rho, 1e-9)), 3),
            lambda_c=lam,
            sigma_u_fit=sigma_u,
            sigma_y_fit=sigma_y,
            ll=ll,
            n_iter=n_iter,
            u_min=float(np.nanmin(u_mean)),
            u_max=float(np.nanmax(u_mean)),
            u_neg_frac=float(np.nanmean(u_mean < 0)),
            elapsed=round(elapsed, 1),
            u_mean=u_mean,
            c_mean=c_mean,
        )
        results.append(rec)
        print(f"[{i+1:2d}/{len(conditions)}] {cond['label']}")
        print(f"    rho={rho:.4f} tau_u={rec['tau_u']:.2f}s  σu={sigma_u:.3f}  σy={sigma_y:.3f}"
              f"  ll={ll:.1f}  u_range=[{rec['u_min']:.1f}, {rec['u_max']:.1f}]"
              f"  neg%={rec['u_neg_frac']:.0%}  ({elapsed:.1f}s)\n")
    except Exception as e:
        print(f"[{i+1:2d}/{len(conditions)}] {cond['label']}  FAILED: {e}\n")

# ── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'#':>3}  {'tau_u_clip':>14}  {'σu_scale':>8}  {'σy_frac':>7}  "
      f"{'rho':>6}  {'tau_u':>6}  {'σu_fit':>7}  {'σy_fit':>7}  {'LL':>9}  "
      f"{'u_min':>6}  {'u_max':>6}  {'neg%':>5}")
print("-" * 100)
for r in sorted(results, key=lambda x: -x["ll"]):
    print(f"{r['idx']:3d}  {str(r['tau_u_clip']):>14}  {r['sigma_u_scale']:>8.2f}  "
          f"{r['sigma_y_frac']:>7.2f}  {r['rho']:>6.4f}  {r['tau_u']:>6.2f}  "
          f"{r['sigma_u_fit']:>7.3f}  {r['sigma_y_fit']:>7.3f}  {r['ll']:>9.1f}  "
          f"{r['u_min']:>6.1f}  {r['u_max']:>6.1f}  {r['u_neg_frac']:>5.0%}")

# ── Comparison plot — grid of u traces ──────────────────────────────────────
n = len(results)
ncols = min(4, n)
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                         sharex=True, sharey=True, squeeze=False)

# Pick a zoom window around the biggest calcium event
best_ll_idx = max(range(n), key=lambda i: results[i]["ll"])
peak_t = int(np.nanargmax(results[best_ll_idx]["c_mean"]))
hw = int(40 / dt)  # ~40 seconds window
t_lo = max(0, peak_t - hw)
t_hi = min(T, peak_t + hw)
t_slice = slice(t_lo, t_hi)

# Sort by LL for display
sorted_results = sorted(results, key=lambda x: -x["ll"])

for k, r in enumerate(sorted_results):
    row, col = divmod(k, ncols)
    ax = axes[row][col]
    ax.plot(time_axis[t_slice], r["c_mean"][t_slice], color="steelblue",
            alpha=0.6, linewidth=1, label="c (calcium)")
    ax.plot(time_axis[t_slice], r["u_mean"][t_slice], color="darkorange",
            linewidth=1.2, label="u (drive)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(
        f"τ_u_clip={r['tau_u_clip']}, σu_s={r['sigma_u_scale']}, σy_f={r['sigma_y_frac']}\n"
        f"ρ={r['rho']:.3f} τ_u={r['tau_u']:.2f}s  LL={r['ll']:.0f}  u∈[{r['u_min']:.1f},{r['u_max']:.1f}]",
        fontsize=7,
    )
    ax.tick_params(labelsize=7)

# Remove empty axes
for k in range(len(sorted_results), nrows * ncols):
    row, col = divmod(k, ncols)
    axes[row][col].set_visible(False)

axes[0][0].legend(fontsize=6, loc="upper right")
fig.suptitle("Stage1 EM Sweep — RIFL neuron (zoomed on peak)", fontsize=11, y=1.01)
fig.tight_layout()
out_path = os.path.join(SAVE_DIR, "sweep_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n[plot] Saved comparison grid -> {out_path}")

# ── Full-trace comparison for top-5 ────────────────────────────────────────
fig2, axes2 = plt.subplots(5, 1, figsize=(14, 12), sharex=True, sharey=True)
top5 = sorted_results[:5]
for k, r in enumerate(top5):
    ax = axes2[k]
    ax.plot(time_axis, r["c_mean"], color="steelblue", alpha=0.5, linewidth=0.8, label="c")
    ax.plot(time_axis, r["u_mean"], color="darkorange", linewidth=0.9, label="u")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_ylabel("a.u.", fontsize=8)
    ax.set_title(
        f"τ_u_clip={r['tau_u_clip']}, σu_s={r['sigma_u_scale']}, σy_f={r['sigma_y_frac']}  |  "
        f"ρ={r['rho']:.3f}  LL={r['ll']:.0f}  u∈[{r['u_min']:.1f},{r['u_max']:.1f}]",
        fontsize=8,
    )
    if k == 0:
        ax.legend(fontsize=7, loc="upper right")
axes2[-1].set_xlabel("Time (s)", fontsize=9)
fig2.suptitle("Stage1 EM Sweep — RIFL — Top 5 by LL (full trace)", fontsize=11)
fig2.tight_layout()
out_path2 = os.path.join(SAVE_DIR, "sweep_top5_full.png")
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"[plot] Saved top-5 full traces -> {out_path2}")

# ── Save JSON summary (without arrays) ─────────────────────────────────────
summary = []
for r in sorted_results:
    s = {k: v for k, v in r.items() if k not in ("u_mean", "c_mean")}
    s["tau_u_clip"] = list(s["tau_u_clip"])
    summary.append(s)

with open(os.path.join(SAVE_DIR, "sweep_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[json] Saved results -> {os.path.join(SAVE_DIR, 'sweep_results.json')}")

print("\nDone!")
