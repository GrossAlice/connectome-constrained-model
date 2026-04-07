#!/usr/bin/env python3
"""Quick plot for baseline transformer CV results on a single worm."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1]  # path to eval_results.json
with open(RESULTS) as f:
    res = json.load(f)

from pathlib import Path
out_dir = Path(RESULTS).parent
worm_id = res["worm_id"]

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# ── 1. One-step R² histogram ──
r2_os = np.array(res["onestep"]["r2"])
ax = axes[0, 0]
ax.hist(r2_os, bins=25, color="steelblue", edgecolor="k", lw=0.3)
ax.axvline(np.nanmean(r2_os), color="red", ls="--", lw=1.2,
           label=f"mean={np.nanmean(r2_os):.3f}")
ax.set_title(f"One-step R²  (N={len(r2_os)} neurons)")
ax.set_xlabel("R²"); ax.set_ylabel("# neurons"); ax.legend(fontsize=7)

# ── 2. LOO R² histogram ──
r2_loo = np.array(res["loo"]["r2"])
finite_loo = r2_loo[np.isfinite(r2_loo)]
ax = axes[0, 1]
ax.hist(finite_loo, bins=15, color="darkorange", edgecolor="k", lw=0.3)
ax.axvline(np.nanmean(finite_loo), color="red", ls="--", lw=1.2,
           label=f"mean={np.nanmean(finite_loo):.3f}")
ax.set_title(f"LOO R²  ({len(finite_loo)} neurons)")
ax.set_xlabel("R²"); ax.legend(fontsize=7)

# ── 3. LOO windowed R² histogram ──
r2_loow = np.array(res["loo_windowed"]["r2"])
finite_loow = r2_loow[np.isfinite(r2_loow)]
w = res["loo_windowed"].get("window_size", 50)
ax = axes[0, 2]
ax.hist(finite_loow, bins=15, color="goldenrod", edgecolor="k", lw=0.3)
ax.axvline(np.nanmean(finite_loow), color="red", ls="--", lw=1.2,
           label=f"mean={np.nanmean(finite_loow):.3f}")
ax.set_title(f"LOO-windowed R²  (w={w}, {len(finite_loow)} neurons)")
ax.set_xlabel("R²"); ax.legend(fontsize=7)

# ── 4. Free-run R² histogram ──
r2_fr = np.array(res["free_run"]["r2"])
finite_fr = r2_fr[np.isfinite(r2_fr)]
ax = axes[1, 0]
ax.hist(finite_fr, bins=15, color="seagreen", edgecolor="k", lw=0.3)
ax.axvline(np.nanmean(finite_fr), color="red", ls="--", lw=1.2,
           label=f"mean={np.nanmean(finite_fr):.3f}")
ax.set_title(f"Free-run R²  ({res['free_run']['mode']}, {len(finite_fr)} neurons)")
ax.set_xlabel("R²"); ax.legend(fontsize=7)

# ── 5. Behaviour R² (ridge: model vs GT) ──
ax = axes[1, 1]
beh_r = res.get("behaviour_ridge", {})
beh_m = res.get("behaviour_mlp", {})
modes = np.arange(6)
w_ = 0.2
ax.bar(modes - w_, beh_r.get("r2_gt", []), w_, label="Ridge (GT)", color="cornflowerblue")
ax.bar(modes,      beh_r.get("r2_model", []), w_, label="Ridge (Model→beh)", color="salmon")
ax.bar(modes + w_, beh_m.get("r2_model", []), w_, label="MLP (Model→beh)", color="mediumpurple")
direct = res.get("behaviour_direct", {}).get("r2", [])
if direct:
    ax.scatter(modes, direct, marker="*", s=60, color="black", zorder=5, label="Direct (model)")
ax.set_xticks(modes); ax.set_xlabel("Eigenworm mode"); ax.set_ylabel("R²")
ax.set_title("Behaviour R²")
ax.legend(fontsize=6, ncol=2)

# ── 6. Summary bar ──
ax = axes[1, 2]
metrics = {
    "1-step": np.nanmean(r2_os),
    "LOO": np.nanmean(finite_loo),
    f"LOO-w{w}": np.nanmean(finite_loow),
    "Free-run": np.nanmean(finite_fr),
    "Beh-Direct": res.get("behaviour_direct", {}).get("r2_mean", 0),
    "Beh-Ridge": beh_r.get("r2_model_mean", 0),
}
colors = ["steelblue", "darkorange", "goldenrod", "seagreen", "black", "salmon"]
bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, edgecolor="k", lw=0.3)
for b, v in zip(bars, metrics.values()):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
ax.set_ylabel("Mean R²"); ax.set_title("Summary"); ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=30)

fig.suptitle(f"Baseline Transformer — {worm_id}  (5-fold CV, d=128, L=2, H=4, K=16)", fontsize=12, y=1.01)
fig.tight_layout()
save_path = out_dir / "summary_plots.png"
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")
