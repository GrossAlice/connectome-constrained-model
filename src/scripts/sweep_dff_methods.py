#!/usr/bin/env python
"""Compare Stage1 LGSSM output across different ΔF/F₀ methods.

Uses condition G parameters: tau_u_clip_sec=(2,5), sigma_u_scale_init=0.3,
sigma_y_floor_frac=0.95.  Compares:
  1. Global 20th-percentile (current default)
  2. Rolling 20th-percentile, 120s window
  3. Rolling 20th-percentile, 60s window
  4. No ΔF/F at all (raw fluorescence → let EM learn α,β)
"""
from __future__ import annotations

import os, sys, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.config import Stage1Config
from stage1.em import fit_stage1_all_neurons
from stage1.io_h5 import load_traces_and_regressor

# ── Data ────────────────────────────────────────────────────────────────────
H5   = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01 (copy).h5"
MASK = "/tmp/rifl_mask.npy"
SAVE = "output_plots/stage1/rifl-dff-methods"
os.makedirs(SAVE, exist_ok=True)

# ── Shared Stage1 params (condition G) ──────────────────────────────────────
SHARED = dict(
    tau_u_clip_sec=(2.0, 5.0),
    sigma_u_scale_init=0.3,
    sigma_y_floor_frac=0.95,
)

# ── DFF method conditions ───────────────────────────────────────────────────
CONDITIONS = [
    dict(name="1: global pct20 (current default)",
         use_dff=True, f0_method="quantile", f0_quantile=0.2, f0_window_sec=120.0),
    dict(name="2: rolling pct20, 120s window",
         use_dff=True, f0_method="rolling_quantile", f0_quantile=0.2, f0_window_sec=120.0),
    dict(name="3: rolling pct20, 60s window",
         use_dff=True, f0_method="rolling_quantile", f0_quantile=0.2, f0_window_sec=60.0),
    dict(name="4: no ΔF/F (raw fluorescence)",
         use_dff=False, f0_method="quantile", f0_quantile=0.2, f0_window_sec=120.0),
]

# ── Run each condition ──────────────────────────────────────────────────────
results = []
for i, cond in enumerate(CONDITIONS):
    t0 = time.time()
    cfg = Stage1Config(
        h5_path=H5, neuron_mask=MASK, overwrite=True, save_dir=SAVE,
        use_dff=cond["use_dff"],
        f0_method=cond["f0_method"],
        f0_quantile=cond["f0_quantile"],
        f0_window_sec=cond["f0_window_sec"],
        **SHARED,
    )
    # Load traces (each condition may preprocess differently)
    X = load_traces_and_regressor(cfg)

    # Apply neuron mask
    from stage1.run_stage1 import _load_neuron_labels, _apply_neuron_mask
    labels = _load_neuron_labels(cfg, n_neurons=X.shape[1])
    X, labels = _apply_neuron_mask(cfg, X, labels)
    T = X.shape[0]
    dt = 1.0 / cfg.sample_rate_hz
    t = np.arange(T) * dt

    # Store the observed trace for plotting
    y_obs = X[:, 0].copy()

    # Fit
    out = fit_stage1_all_neurons(X, cfg)
    u  = np.asarray(out["u_mean"])[:, 0]
    uv = np.asarray(out["u_var"])[:, 0]
    c  = np.asarray(out["c_mean"])[:, 0]
    rho    = float(out["rho"][0])
    sig_u  = float(out["sigma_u"][0])
    sig_y  = float(out["sigma_y"][0])
    alpha  = float(out["alpha"][0])
    beta   = float(out["beta"][0])
    ll     = float(out["ll_hist"][-1])
    tau_u  = round(-dt / np.log(max(rho, 1e-9)), 3)
    elapsed = time.time() - t0

    results.append(dict(
        name=cond["name"], u=u, u_var=uv, c=c, y_obs=y_obs, t=t,
        rho=rho, tau_u=tau_u, sig_u=sig_u, sig_y=sig_y,
        alpha=alpha, beta=beta, ll=ll,
        u_min=float(np.nanmin(u)), u_max=float(np.nanmax(u)),
        y_min=float(np.nanmin(y_obs)), y_max=float(np.nanmax(y_obs)),
    ))
    print(f"[{i+1}/{len(CONDITIONS)}] {cond['name']}")
    print(f"   ρ={rho:.4f}  τ_u={tau_u:.2f}s  σu={sig_u:.3f}  σy={sig_y:.3f}"
          f"  α={alpha:.3f}  β={beta:.3f}")
    print(f"   LL={ll:.1f}  u∈[{np.nanmin(u):.1f},{np.nanmax(u):.1f}]"
          f"  y∈[{np.nanmin(y_obs):.3f},{np.nanmax(y_obs):.3f}]  ({elapsed:.1f}s)\n")

# ── Zoom window ─────────────────────────────────────────────────────────────
c_ref = results[0]["c"]
peak_t = int(np.nanargmax(c_ref))
hw = int(30 / dt)
t_lo, t_hi = max(0, peak_t - hw), min(T, peak_t + hw)
sl = slice(t_lo, t_hi)

# ── Colours ─────────────────────────────────────────────────────────────────
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 4-row plot: one per condition (zoomed)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(results), 3, figsize=(16, 3.5 * len(results)),
                         squeeze=False)
for k, r in enumerate(results):
    t_arr = r["t"]
    # Panel 1: observed y(t)
    ax = axes[k][0]
    ax.plot(t_arr[sl], r["y_obs"][sl], color=colors[k], lw=1)
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_ylabel(f"y(t)", fontsize=9)
    ax.set_title(f'{r["name"]}\ny∈[{r["y_min"]:.3f}, {r["y_max"]:.3f}]', fontsize=8)
    ax.tick_params(labelsize=7)

    # Panel 2: latent drive u(t)
    ax = axes[k][1]
    sd = np.sqrt(np.maximum(r["u_var"][sl], 0))
    ax.fill_between(t_arr[sl], r["u"][sl] - sd, r["u"][sl] + sd,
                    color=colors[k], alpha=0.2)
    ax.plot(t_arr[sl], r["u"][sl], color=colors[k], lw=1.2)
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_ylabel("u(t)", fontsize=9)
    ax.set_title(f'ρ={r["rho"]:.3f}  τ_u={r["tau_u"]:.2f}s  σu={r["sig_u"]:.2f}'
                 f'  u∈[{r["u_min"]:.1f},{r["u_max"]:.1f}]', fontsize=8)
    ax.tick_params(labelsize=7)

    # Panel 3: calcium c(t)
    ax = axes[k][2]
    ax.plot(t_arr[sl], r["c"][sl], color=colors[k], lw=1)
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_ylabel("c(t)", fontsize=9)
    ax.set_title(f'α={r["alpha"]:.3f}  β={r["beta"]:.3f}  σy={r["sig_y"]:.3f}', fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes[-1]:
    ax.set_xlabel("Time (seconds)", fontsize=9)

fig.suptitle("ΔF/F₀ method comparison — RIFL neuron (condition G params)", fontsize=12, y=1.01)
fig.tight_layout()
p1 = os.path.join(SAVE, "dff_methods_zoom.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
print(f"[plot] {p1}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Overlay of u(t) from all methods (zoomed)
# ═══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2.5, 1]})

ax = axes2[0]
for k, r in enumerate(results):
    ax.plot(r["t"][sl], r["u"][sl], color=colors[k], lw=1.3, alpha=0.85,
            label=f'{r["name"]}  (u∈[{r["u_min"]:.1f},{r["u_max"]:.1f}])')
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_ylabel("Latent drive  u(t)", fontsize=11)
ax.set_title("Latent drive overlay — ΔF/F₀ method comparison (RIFL, cond G)", fontsize=12)
ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
ax.tick_params(labelsize=9)

ax = axes2[1]
for k, r in enumerate(results):
    ax.plot(r["t"][sl], r["y_obs"][sl], color=colors[k], lw=1, alpha=0.7,
            label=r["name"].split(":")[0] + " y(t)")
ax.axhline(0, color="gray", lw=0.4, ls="--")
ax.set_ylabel("Observed trace  y(t)", fontsize=11)
ax.set_xlabel("Time (seconds)", fontsize=11)
ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.8)
ax.tick_params(labelsize=9)

fig2.tight_layout()
p2 = os.path.join(SAVE, "dff_methods_overlay.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print(f"[plot] {p2}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Full recording u(t) overlay
# ═══════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(16, 4.5))
for k, r in enumerate(results):
    ax3.plot(r["t"], r["u"], color=colors[k], lw=0.6, alpha=0.75,
             label=r["name"])
ax3.axhline(0, color="gray", lw=0.4, ls="--")
ax3.set_xlabel("Time (seconds)", fontsize=11)
ax3.set_ylabel("Latent drive  u(t)", fontsize=11)
ax3.set_title("Full recording — ΔF/F₀ method comparison (RIFL, cond G)", fontsize=12)
ax3.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.8)
ax3.tick_params(labelsize=9)
fig3.tight_layout()
p3 = os.path.join(SAVE, "dff_methods_full.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"[plot] {p3}")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'#':>2} {'Name':<40} {'ρ':>6} {'τ_u':>5} {'σu':>6} {'σy':>6} "
      f"{'α':>7} {'β':>7} {'LL':>9} {'u_min':>6} {'u_max':>6}")
print("-" * 100)
for k, r in enumerate(results):
    print(f"{k:2d} {r['name']:<40} {r['rho']:6.4f} {r['tau_u']:5.2f} "
          f"{r['sig_u']:6.3f} {r['sig_y']:6.3f} {r['alpha']:7.3f} {r['beta']:7.3f} "
          f"{r['ll']:9.1f} {r['u_min']:6.1f} {r['u_max']:6.1f}")

# Save JSON
summary = [{k: v for k, v in r.items() if k not in ("u", "u_var", "c", "y_obs", "t")}
           for r in results]
with open(os.path.join(SAVE, "dff_results.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone!  Plots in {SAVE}/")
