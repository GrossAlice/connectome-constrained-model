#!/usr/bin/env python
"""Overlay latent-drive u(t) from several Stage1 parameter sets on one plot."""
from __future__ import annotations

import os, sys, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.config import Stage1Config
from stage1.em import fit_stage1_all_neurons
from stage1.run_stage1 import load_traces_and_regressor, _load_neuron_labels, _apply_neuron_mask

# ── Data ────────────────────────────────────────────────────────────────────
H5   = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01 (copy).h5"
MASK = "/tmp/rifl_mask.npy"
SAVE = "output_plots/stage1/rifl-sweep"
os.makedirs(SAVE, exist_ok=True)

# ── Parameter grid — named conditions ───────────────────────────────────────
CONDITIONS = [
    dict(name="A: default (0.3,1) σu0.7 σyf0.85",
         tau_u_clip_sec=(0.3, 1.0), sigma_u_scale_init=0.7, sigma_y_floor_frac=0.85),
    dict(name="B: (0.6,2) σu0.7 σyf0.85",
         tau_u_clip_sec=(0.6, 2.0), sigma_u_scale_init=0.7, sigma_y_floor_frac=0.85),
    dict(name="C: (1,3) σu0.7 σyf0.85",
         tau_u_clip_sec=(1.0, 3.0), sigma_u_scale_init=0.7, sigma_y_floor_frac=0.85),
    dict(name="D: (1,3) σu0.3 σyf0.85",
         tau_u_clip_sec=(1.0, 3.0), sigma_u_scale_init=0.3, sigma_y_floor_frac=0.85),
    dict(name="E: (1,3) σu0.3 σyf0.95",
         tau_u_clip_sec=(1.0, 3.0), sigma_u_scale_init=0.3, sigma_y_floor_frac=0.95),
    dict(name="F: (1.5,5) σu0.3 σyf0.95",
         tau_u_clip_sec=(1.5, 5.0), sigma_u_scale_init=0.3, sigma_y_floor_frac=0.95),
    dict(name="G: (2,5) σu0.3 σyf0.95",
         tau_u_clip_sec=(2.0, 5.0), sigma_u_scale_init=0.3, sigma_y_floor_frac=0.95),
    dict(name="H: (0.3,1) σu0.3 σyf0.95",
         tau_u_clip_sec=(0.3, 1.0), sigma_u_scale_init=0.3, sigma_y_floor_frac=0.95),
]

# ── Load traces once ────────────────────────────────────────────────────────
base_cfg = Stage1Config(h5_path=H5, neuron_mask=MASK, overwrite=True)
X_raw    = load_traces_and_regressor(base_cfg)
labels   = _load_neuron_labels(base_cfg, n_neurons=X_raw.shape[1])
X, labels = _apply_neuron_mask(base_cfg, X_raw, labels)
T  = X.shape[0]
dt = 1.0 / base_cfg.sample_rate_hz
t  = np.arange(T) * dt

print(f"Loaded {T} frames, dt={dt:.3f}s, neuron: {labels}\n")

# ── Fit all conditions ──────────────────────────────────────────────────────
results = []
for i, cond in enumerate(CONDITIONS):
    t0 = time.time()
    cfg = Stage1Config(
        h5_path=H5, neuron_mask=MASK, overwrite=True, save_dir=SAVE,
        tau_u_clip_sec=cond["tau_u_clip_sec"],
        sigma_u_scale_init=cond["sigma_u_scale_init"],
        sigma_y_floor_frac=cond["sigma_y_floor_frac"],
    )
    out = fit_stage1_all_neurons(X, cfg)
    u  = np.asarray(out["u_mean"])[:, 0]
    uv = np.asarray(out["u_var"])[:, 0]
    c  = np.asarray(out["c_mean"])[:, 0]
    rho    = float(out["rho"][0])
    sig_u  = float(out["sigma_u"][0])
    sig_y  = float(out["sigma_y"][0])
    ll     = float(out["ll_hist"][-1])
    tau_u  = round(-dt / np.log(max(rho, 1e-9)), 3)
    elapsed = time.time() - t0

    results.append(dict(
        name=cond["name"], u=u, u_var=uv, c=c,
        rho=rho, tau_u=tau_u, sig_u=sig_u, sig_y=sig_y, ll=ll,
        u_min=float(np.nanmin(u)), u_max=float(np.nanmax(u)),
    ))
    print(f"[{i+1}/{len(CONDITIONS)}] {cond['name']}")
    print(f"   ρ={rho:.4f}  τ_u={tau_u:.2f}s  σu={sig_u:.3f}  σy={sig_y:.3f}"
          f"  LL={ll:.1f}  u∈[{np.nanmin(u):.1f},{np.nanmax(u):.1f}]  ({elapsed:.1f}s)\n")

# ── Pick zoom window around biggest calcium event ───────────────────────────
c_ref  = results[0]["c"]
peak_t = int(np.nanargmax(c_ref))
hw     = int(30 / dt)            # ±30 s
t_lo, t_hi = max(0, peak_t - hw), min(T, peak_t + hw)
sl = slice(t_lo, t_hi)

# also get the observed fluorescence for the bottom panel
y_obs = X[sl, 0]

# ── Colour palette ──────────────────────────────────────────────────────────
cmap   = plt.cm.get_cmap("tab10", len(results))
colors = [cmap(i) for i in range(len(results))]

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Multi-panel overlay (like the attached image)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.2, 1.2]})

# --- Panel 1: Latent drive u(t) overlay ---
ax = axes[0]
for k, r in enumerate(results):
    ax.plot(t[sl], r["u"][sl], color=colors[k], linewidth=1.2, alpha=0.85,
            label=f'{r["name"]}  (τ_u={r["tau_u"]:.2f}s, LL={r["ll"]:.0f})')
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_ylabel("Latent drive  $u_i(t)$", fontsize=11, color="navy")
ax.set_title("Stage 1 LGSSM — Latent drive overlay (RIFL neuron)", fontsize=13)
ax.legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.8)
ax.tick_params(labelsize=9)

# --- Panel 2: Calcium c(t) (same for all, but overlay anyway) ---
ax = axes[1]
for k, r in enumerate(results):
    ax.plot(t[sl], r["c"][sl], color=colors[k], linewidth=1, alpha=0.7)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_ylabel("Calcium  $c_i(t)$", fontsize=11, color="darkorange")
ax.tick_params(labelsize=9)

# --- Panel 3: Observed fluorescence ---
ax = axes[2]
ax.plot(t[sl], y_obs, color="green", linewidth=1, alpha=0.8, label="observed ΔF/F")
ax.set_ylabel("Fluorescence  $y_i(t)$", fontsize=11, color="green")
ax.set_xlabel("Time (seconds)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.tick_params(labelsize=9)

fig.tight_layout()
p1 = os.path.join(SAVE, "overlay_zoom.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
print(f"\n[plot] {p1}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Full recording overlay (just u)
# ═══════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(16, 5))
for k, r in enumerate(results):
    ax2.plot(t, r["u"], color=colors[k], linewidth=0.7, alpha=0.75,
             label=f'{r["name"]}')
ax2.axhline(0, color="gray", lw=0.4, ls="--")
ax2.set_xlabel("Time (seconds)", fontsize=11)
ax2.set_ylabel("Latent drive  $u_i(t)$", fontsize=11)
ax2.set_title("Stage 1 — Latent drive overlay (full recording, RIFL)", fontsize=13)
ax2.legend(fontsize=6.5, loc="upper left", ncol=2, framealpha=0.8)
ax2.tick_params(labelsize=9)
fig2.tight_layout()
p2 = os.path.join(SAVE, "overlay_full.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print(f"[plot] {p2}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — u with ±1 std band, one subplot per condition (zoomed)
# ═══════════════════════════════════════════════════════════════════════════
n = len(results)
ncols = 2
nrows = (n + ncols - 1) // ncols
fig3, axes3 = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows),
                           sharex=True, sharey=True, squeeze=False)
for k, r in enumerate(results):
    row, col = divmod(k, ncols)
    ax = axes3[row][col]
    u_sl  = r["u"][sl]
    sd_sl = np.sqrt(np.maximum(r["u_var"][sl], 0))
    ax.fill_between(t[sl], u_sl - sd_sl, u_sl + sd_sl, color=colors[k], alpha=0.2)
    ax.plot(t[sl], u_sl, color=colors[k], linewidth=1.2)
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_title(f'{r["name"]}\nρ={r["rho"]:.3f}  τ_u={r["tau_u"]:.2f}s  '
                 f'σu={r["sig_u"]:.2f}  LL={r["ll"]:.0f}  '
                 f'u∈[{r["u_min"]:.1f},{r["u_max"]:.1f}]', fontsize=8)
    ax.tick_params(labelsize=7)
for k in range(n, nrows * ncols):
    axes3[k // ncols][k % ncols].set_visible(False)
fig3.suptitle("Latent drive ± 1 std  (RIFL, zoomed)", fontsize=12, y=1.01)
fig3.tight_layout()
p3 = os.path.join(SAVE, "overlay_individual.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"[plot] {p3}")

# ── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"{'#':>2} {'Name':<42} {'ρ':>6} {'τ_u':>5} {'σu':>6} {'σy':>6} "
      f"{'LL':>9} {'u_min':>6} {'u_max':>6}")
print("-" * 110)
for k, r in enumerate(sorted(results, key=lambda x: -x["ll"])):
    print(f"{k:2d} {r['name']:<42} {r['rho']:6.4f} {r['tau_u']:5.2f} "
          f"{r['sig_u']:6.3f} {r['sig_y']:6.3f} {r['ll']:9.1f} "
          f"{r['u_min']:6.1f} {r['u_max']:6.1f}")

# ── Save JSON ───────────────────────────────────────────────────────────────
summary = [{k: v for k, v in r.items() if k not in ("u", "u_var", "c")} for r in results]
with open(os.path.join(SAVE, "overlay_results.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone!  Plots in {SAVE}/")
