#!/usr/bin/env python3
"""
Summary plot for connectome-constrained neural decoder batch results.

Compares four predictor-set × self-history conditions:
  • conn+self   – connectome neighbours + own history (AR)
  • conn_only   – connectome neighbours only
  • all+self    – all neurons + own history
  • all_only    – all neurons only

Three decoders each: Ridge, PCA-Ridge, MLP.

Reads output_plots/connectome_mlp_batch/<worm>_all/T_e/results.json
"""
from __future__ import annotations

import json, os, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── gather data ──────────────────────────────────────────────────────────
BASE = Path("output_plots/connectome_mlp_batch")
OUT  = BASE / "summary"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob(str(BASE / "*_all" / "T_e" / "results.json")))
assert files, "No results.json found!"

CONDITIONS = ["conn+self", "conn_only", "all+self", "all_only"]
DECODERS   = ["ridge", "pca_ridge", "mlp"]
DECODER_LABELS = {"ridge": "Ridge", "pca_ridge": "PCA-Ridge", "mlp": "MLP"}
COND_LABELS = {
    "conn+self": "Connectome + Self",
    "conn_only": "Connectome Only",
    "all+self":  "All Neurons + Self",
    "all_only":  "All Neurons Only",
}

worms = []
# {cond -> {decoder -> [r2_list across worms]}}
r2_data   = {c: {d: [] for d in DECODERS} for c in CONDITIONS}
corr_data = {c: {d: [] for d in DECODERS} for c in CONDITIONS}
# per-neuron data for scatter / violin
per_neuron = {c: {d: [] for d in DECODERS} for c in CONDITIONS}  # list of arrays
n_partners_all = []  # list of arrays

for f in files:
    worm = Path(f).parent.parent.name.replace("_all", "")
    worms.append(worm)
    with open(f) as fp:
        raw = json.load(fp)
    # results are nested under a context-length key (e.g. "5")
    ctx_key = list(raw.keys())[0]
    data = raw[ctx_key]

    for c in CONDITIONS:
        m = data[c]
        for d in DECODERS:
            r2_data[c][d].append(m[f"r2_mean_{d}"])
            corr_data[c][d].append(m[f"corr_mean_{d}"])
            per_neuron[c][d].append(np.array(m[f"r2_per_neuron_{d}"]))
    # n_partners is the same across conditions; grab from first available
    if "n_partners" in data[CONDITIONS[0]]:
        n_partners_all.append(np.array(data[CONDITIONS[0]]["n_partners"]))

n_worms = len(worms)
print(f"Loaded {n_worms} worms: {worms}")

# ── colours ──────────────────────────────────────────────────────────────
COND_COLORS = {
    "conn+self": "#2166ac",
    "conn_only": "#92c5de",
    "all+self":  "#b2182b",
    "all_only":  "#f4a582",
}

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Bar chart: mean R² per condition × decoder, averaged over worms
# ═════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, metric, data_dict, ylabel in [
    (axes[0], "R²", r2_data, "Mean R² (across neurons)"),
    (axes[1], "Correlation", corr_data, "Mean Pearson r"),
]:
    x = np.arange(len(DECODERS))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    for i, c in enumerate(CONDITIONS):
        means = [np.mean(data_dict[c][d]) for d in DECODERS]
        sems  = [np.std(data_dict[c][d]) / np.sqrt(n_worms) for d in DECODERS]
        ax.bar(x + offsets[i], means, width * 0.9, yerr=sems,
               color=COND_COLORS[c], label=COND_LABELS[c],
               capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([DECODER_LABELS[d] for d in DECODERS], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{metric} by Condition × Decoder (n={n_worms} worms)", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "bar_summary.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT / 'bar_summary.png'}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Per-worm paired comparison: conn+self vs all+self (Ridge)
# ═════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
comparisons = [
    ("conn+self", "all+self", "Connectome+Self vs All+Self"),
    ("conn+self", "all_only", "Connectome+Self vs All-Only"),
    ("conn+self", "conn_only", "Connectome+Self vs Connectome-Only"),
]
for ax, (c1, c2, title) in zip(axes, comparisons):
    dec = "ridge"
    y1 = np.array(r2_data[c1][dec])
    y2 = np.array(r2_data[c2][dec])
    for w in range(n_worms):
        ax.plot([0, 1], [y1[w], y2[w]], "o-", color="grey", alpha=0.5, markersize=5)
    ax.plot([0, 1], [y1.mean(), y2.mean()], "s-", color="black", markersize=10,
            linewidth=2.5, zorder=5, label="Mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([COND_LABELS[c1], COND_LABELS[c2]], fontsize=10)
    ax.set_ylabel("Mean R² (Ridge)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.3, 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    # add p-value from paired t-test
    from scipy.stats import ttest_rel, wilcoxon
    t, p = ttest_rel(y1, y2)
    diff = y1 - y2
    ax.text(0.5, 0.05, f"Δ = {diff.mean():.3f} ± {diff.std():.3f}\np = {p:.2e} (paired t)",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

fig.suptitle("Ridge R²: Paired Comparison Across Worms", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "paired_comparison.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT / 'paired_comparison.png'}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3 – The key question: does connectome constraint help?
#   Scatter: conn+self R² vs all+self R² per neuron (pooled across worms)
# ═════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, dec in zip(axes, DECODERS):
    cs_r2 = np.concatenate(per_neuron["conn+self"][dec])
    as_r2 = np.concatenate(per_neuron["all+self"][dec])
    ax.scatter(as_r2, cs_r2, s=4, alpha=0.15, color=COND_COLORS["conn+self"])
    lim = (-0.5, 1.0)
    ax.plot(lim, lim, "--", color="grey", linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("R² (All Neurons + Self)", fontsize=11)
    ax.set_ylabel("R² (Connectome + Self)", fontsize=11)
    ax.set_title(f"{DECODER_LABELS[dec]} — per neuron (n={len(cs_r2)})", fontsize=12)
    ax.set_aspect("equal")
    above = np.mean(cs_r2 > as_r2)
    ax.text(0.05, 0.92, f"{above*100:.1f}% above diagonal",
            transform=ax.transAxes, fontsize=10, color=COND_COLORS["conn+self"],
            fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Connectome-Constrained vs Unconstrained: Per-Neuron R²", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "scatter_conn_vs_all.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT / 'scatter_conn_vs_all.png'}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Self-history is key: conn+self vs conn_only violin
# ═════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))
violin_data = []
labels = []
colors = []
for c in CONDITIONS:
    vals = np.concatenate(per_neuron[c]["ridge"])
    vals = vals[vals != None].astype(float)
    vals = vals[np.isfinite(vals)]
    violin_data.append(vals)
    labels.append(COND_LABELS[c])
    colors.append(COND_COLORS[c])

parts = ax.violinplot(violin_data, positions=range(len(CONDITIONS)),
                      showmedians=True, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
parts["cmedians"].set_color("black")

# add mean markers
for i, v in enumerate(violin_data):
    ax.scatter(i, np.mean(v), color="black", s=60, zorder=5, marker="D")
    ax.text(i + 0.15, np.mean(v), f"{np.mean(v):.3f}", fontsize=9, va="center")

ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("R² per neuron (Ridge)", fontsize=12)
ax.set_title(f"Per-Neuron R² Distribution — Ridge (pooled, n={len(violin_data[0])} neurons)", fontsize=13)
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "violin_conditions.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT / 'violin_conditions.png'}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 5 – R² vs number of connectome partners
# ═════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
np_all = np.concatenate(n_partners_all).astype(float)
cs_r2_ridge = np.concatenate(per_neuron["conn+self"]["ridge"]).astype(float)
co_r2_ridge = np.concatenate(per_neuron["conn_only"]["ridge"]).astype(float)
# filter valid
valid = np.isfinite(np_all) & np.isfinite(cs_r2_ridge) & np.isfinite(co_r2_ridge)
np_all = np_all[valid].astype(int)
cs_r2_ridge = cs_r2_ridge[valid]
co_r2_ridge = co_r2_ridge[valid]

for ax, (r2_vals, label, color) in zip(axes, [
    (cs_r2_ridge, "Connectome + Self", COND_COLORS["conn+self"]),
    (co_r2_ridge, "Connectome Only",   COND_COLORS["conn_only"]),
]):
    ax.scatter(np_all, r2_vals, s=8, alpha=0.2, color=color)
    # bin means
    unique_np = sorted(set(np_all))
    bin_means = [np.mean(r2_vals[np_all == n]) for n in unique_np]
    bin_sems  = [np.std(r2_vals[np_all == n]) / max(1, np.sqrt(np.sum(np_all == n)))
                 for n in unique_np]
    ax.errorbar(unique_np, bin_means, yerr=bin_sems, fmt="o-", color="black",
                markersize=6, capsize=3, linewidth=1.5, zorder=5, label="Bin mean ± SEM")
    ax.set_xlabel("# Presynaptic Partners (connectome)", fontsize=11)
    ax.set_ylabel("R² (Ridge)", fontsize=11)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Prediction Quality vs Connectome In-Degree", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "r2_vs_partners.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT / 'r2_vs_partners.png'}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Compact heatmap: worm × condition (Ridge R²)
# ═════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
mat = np.zeros((n_worms, len(CONDITIONS)))
for j, c in enumerate(CONDITIONS):
    for i in range(n_worms):
        mat[i, j] = r2_data[c]["ridge"][i]

im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=10, rotation=20, ha="right")
ax.set_yticks(range(n_worms))
ax.set_yticklabels(worms, fontsize=9)
for i in range(n_worms):
    for j in range(len(CONDITIONS)):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if mat[i,j] > 0.5 else "black")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Mean R² (Ridge)", fontsize=11)
ax.set_title("Ridge R² Across Worms × Conditions", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "heatmap_worm_condition.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT / 'heatmap_worm_condition.png'}")

# ═════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY STATISTICS (Ridge R², mean ± std across worms)")
print("="*70)
for c in CONDITIONS:
    vals = r2_data[c]["ridge"]
    print(f"  {COND_LABELS[c]:<25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

print("\n" + "-"*70)
print("KEY COMPARISONS")
print("-"*70)
cs = np.array(r2_data["conn+self"]["ridge"])
co = np.array(r2_data["conn_only"]["ridge"])
als = np.array(r2_data["all+self"]["ridge"])
alo = np.array(r2_data["all_only"]["ridge"])

from scipy.stats import ttest_rel
for name, a, b in [
    ("conn+self vs all+self", cs, als),
    ("conn+self vs all_only", cs, alo),
    ("conn+self vs conn_only", cs, co),
    ("all+self vs all_only", als, alo),
]:
    diff = a - b
    t, p = ttest_rel(a, b)
    print(f"  {name:<30s}: Δ = {diff.mean():+.4f} ± {diff.std():.4f},  p = {p:.2e}")

print(f"\n  Self-history boost (conn):  {(cs-co).mean():.4f}")
print(f"  Self-history boost (all):   {(als-alo).mean():.4f}")
print(f"  Connectome advantage:       {(cs-als).mean():.4f}")
print(f"  (conn+self beats all+self in {np.sum(cs > als)}/{n_worms} worms)")

# fraction of neurons where conn+self > all+self
cs_per = np.concatenate(per_neuron["conn+self"]["ridge"])
as_per = np.concatenate(per_neuron["all+self"]["ridge"])
frac = np.mean(cs_per > as_per)
print(f"\n  Per-neuron: conn+self > all+self in {frac*100:.1f}% of neurons")
print("="*70)
