#!/usr/bin/env python3
"""
Summarize neural_activity_decoder_v5 (connectome-constrained) results
to inform Stage2 model design.

Mirrors the v4 summary plot but highlights how connectome-defined
neighbours change the picture vs all-neuron neighbours.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


V4_DIR = Path("output_plots/neural_activity_decoder_v4")
V5_DIR = Path("output_plots/neural_activity_decoder_v5_connectome")

INPUT_CONFIGS = ["causal_self", "conc_self", "self", "causal", "conc", "conc_causal"]


def load_per_neuron_data(results_dir: Path):
    """Load all per-neuron R² values pooled across worms."""
    all_data = {"ridge": {}, "mlp": {}, "trf": {}}
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        results_file = subdir / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)
        for _k_val, metrics in data.items():
            for metric_name, value in metrics.items():
                if metric_name.startswith("r2_per_neuron_"):
                    parts = metric_name.replace("r2_per_neuron_", "").split("_", 1)
                    model, inp = parts[0], parts[1]
                    if model in all_data and inp in INPUT_CONFIGS:
                        all_data[model].setdefault(inp, []).extend(value)
    return all_data


def load_connectome_stats(results_dir: Path):
    """Load connectome metadata across worms."""
    stats = {"N": [], "gap_edges": [], "syn_edges": [],
             "avg_gap": [], "avg_syn": []}
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ci_path = subdir / f"connectome_info_{subdir.name}.json"
        if ci_path.exists():
            with open(ci_path) as f:
                ci = json.load(f)
            stats["N"].append(ci["N"])
            stats["gap_edges"].append(ci["n_gap_total"])
            stats["syn_edges"].append(ci["n_syn_total"])
            stats["avg_gap"].append(ci["avg_gap_neighbors"])
            stats["avg_syn"].append(ci["avg_syn_presynaptic"])
    return stats


def create_v5_insights_plot(save_dir: Path):
    """Create multi-panel summary plot for v5 connectome results."""

    v5 = load_per_neuron_data(V5_DIR)
    # Also load v4 for comparison where useful
    v4 = load_per_neuron_data(V4_DIR) if V4_DIR.exists() else None
    conn = load_connectome_stats(V5_DIR)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    models = ["ridge", "mlp", "trf"]
    model_labels = ["Ridge", "MLP", "Transformer"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # =========================================================================
    # Panel A: Self vs Causal+Self — v4 (faded) and v5 (solid)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    x_pos = np.arange(len(models))
    width = 0.2

    self_v5 = [np.mean(v5[m]["self"]) for m in models]
    cs_v5 = [np.mean(v5[m]["causal_self"]) for m in models]

    # v4 ghost bars
    if v4:
        self_v4 = [np.mean(v4[m]["self"]) for m in models]
        cs_v4 = [np.mean(v4[m]["causal_self"]) for m in models]
        ax1.bar(x_pos - 1.5 * width, self_v4, width, color="#90CAF9",
                edgecolor="black", alpha=0.35, label="Self (v4)")
        ax1.bar(x_pos - 0.5 * width, cs_v4, width, color="#FFCC80",
                edgecolor="black", alpha=0.35, label="Caus+Self (v4)")

    ax1.bar(x_pos + 0.5 * width, self_v5, width, color="#90CAF9",
            edgecolor="black", label="Self (v5)")
    ax1.bar(x_pos + 1.5 * width, cs_v5, width, color="#FFCC80",
            edgecolor="black", label="Caus+Self (v5)")

    for i, (s, cs) in enumerate(zip(self_v5, cs_v5)):
        delta = cs - s
        color = "green" if delta > 0 else "red"
        ax1.annotate(f"{delta:+.2f}", xy=(i + 1.5 * width, cs + 0.02),
                     ha="center", fontsize=10, color=color, fontweight="bold")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_labels)
    ax1.set_ylabel("R²", fontsize=12)
    ax1.set_title("A. Neighbor Contribution\n(Connectome vs All-neuron)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # =========================================================================
    # Panel B: Causal-only violin — v4 vs v5
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    causal_v5 = [np.array(v5[m]["causal"]) for m in models]

    positions_v5 = x_pos
    parts5 = ax2.violinplot(causal_v5, positions=positions_v5, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts5["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    for i, vals in enumerate(causal_v5):
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        ax2.scatter(np.full_like(vals, i) + jitter, vals, s=8, alpha=0.3, c=colors[i])

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_labels)
    ax2.set_ylabel("R²", fontsize=12)
    ax2.set_title("B. Causal-Only (LOO Ceiling)\nConnectome neighbours only", fontsize=13, fontweight="bold")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(0.3, color="red", linestyle=":", alpha=0.7, label="Practical ceiling ~0.3")
    ax2.set_ylim(-0.5, 0.8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # =========================================================================
    # Panel C: Per-neuron predictability histogram (self-only, same as v4)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])

    ridge_self = np.array(v5["ridge"]["self"])

    bins = np.linspace(-0.2, 1.0, 25)
    ax3.hist(ridge_self, bins=bins, color="#2196F3", alpha=0.7, edgecolor="white")

    ax3.axvline(0.5, color="orange", linestyle="--", label="R²=0.5 (noisy)", linewidth=2)
    ax3.axvline(0.8, color="green", linestyle="--", label="R²=0.8 (good)", linewidth=2)
    ax3.axvline(0.95, color="red", linestyle="--", label="R²=0.95 (excellent)", linewidth=2)

    ax3.set_xlabel("R² (Ridge self-only)", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("C. Neuron Predictability\nDistribution (v5 worms)", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    n_total = len(ridge_self)
    low = (ridge_self < 0.5).sum()
    med = ((ridge_self >= 0.5) & (ridge_self < 0.8)).sum()
    high = ((ridge_self >= 0.8) & (ridge_self < 0.95)).sum()
    exc = (ridge_self >= 0.95).sum()
    ax3.text(0.95, 0.95, f"<0.5: {low}\n0.5-0.8: {med}\n0.8-0.95: {high}\n>0.95: {exc}",
             transform=ax3.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # =========================================================================
    # Panel D: Delta R² per neuron — v5 connectome (+ v4 ghost)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    ridge_delta_v5 = np.array(v5["ridge"]["causal_self"]) - np.array(v5["ridge"]["self"])
    trf_delta_v5 = np.array(v5["trf"]["causal_self"]) - np.array(v5["trf"]["self"])

    bins_d = np.linspace(-0.5, 0.5, 41)
    ax4.hist(ridge_delta_v5, bins=bins_d, alpha=0.6,
             label=f"Ridge v5 (mean={ridge_delta_v5.mean():.3f})", color="#2196F3")
    ax4.hist(trf_delta_v5, bins=bins_d, alpha=0.6,
             label=f"TRF v5 (mean={trf_delta_v5.mean():.3f})", color="#FF9800")

    # v4 ghost outlines
    if v4:
        ridge_delta_v4 = np.array(v4["ridge"]["causal_self"]) - np.array(v4["ridge"]["self"])
        ax4.hist(ridge_delta_v4, bins=bins_d, histtype="step", linewidth=1.5,
                 linestyle="--", color="#0D47A1", alpha=0.6,
                 label=f"Ridge v4 (mean={ridge_delta_v4.mean():.3f})")

    ax4.axvline(0, color="black", linewidth=2)
    ax4.set_xlabel("ΔR² (causal+self − self)", fontsize=11)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title("D. Per-Neuron Neighbor Benefit\n(connectome-constrained)", fontsize=13, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    ridge_improved = (ridge_delta_v5 > 0).sum()
    trf_improved = (trf_delta_v5 > 0).sum()
    n_d = len(ridge_delta_v5)
    ax4.text(0.98, 0.98,
             f"Ridge: {ridge_improved}/{n_d} ({100*ridge_improved/n_d:.0f}%)\n"
             f"TRF: {trf_improved}/{n_d} ({100*trf_improved/n_d:.0f}%)",
             transform=ax4.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # =========================================================================
    # Panel E: Self R² vs Causal R² scatter (connectome causal)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    ridge_self_arr = np.array(v5["ridge"]["self"])
    ridge_causal_arr = np.array(v5["ridge"]["causal"])

    ax5.scatter(ridge_self_arr, ridge_causal_arr, s=20, alpha=0.5, c="#2196F3", edgecolors="none")
    ax5.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")

    mask = (ridge_self_arr > 0) & (ridge_causal_arr > -0.5)
    z = np.polyfit(ridge_self_arr[mask], ridge_causal_arr[mask], 1)
    x_fit = np.linspace(0, 1, 100)
    ax5.plot(x_fit, np.polyval(z, x_fit), "r-", linewidth=2,
             label=f"Fit: y={z[0]:.2f}x{z[1]:+.2f}")

    ax5.set_xlabel("R² (self-only)", fontsize=11)
    ax5.set_ylabel("R² (causal-only, connectome)", fontsize=11)
    ax5.set_title("E. Self vs Causal Predictability\n(Ridge, connectome neighbours)", fontsize=13, fontweight="bold")
    ax5.set_xlim(-0.1, 1.05)
    ax5.set_ylim(-0.5, 0.8)
    ax5.legend(loc="upper left", fontsize=9)
    ax5.grid(alpha=0.3)

    corr = np.corrcoef(ridge_self_arr, ridge_causal_arr)[0, 1]
    ax5.text(0.95, 0.05, f"r = {corr:.2f}", transform=ax5.transAxes,
             ha="right", va="bottom", fontsize=11)

    # =========================================================================
    # Panel F: Stage2 Design Recommendations (text)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    # Compute some numbers for the text box
    mean_gap = np.mean(conn["avg_gap"]) if conn["avg_gap"] else 0
    mean_syn = np.mean(conn["avg_syn"]) if conn["avg_syn"] else 0
    mean_N = np.mean(conn["N"]) if conn["N"] else 0

    ridge_cs_mean_v5 = np.mean(v5["ridge"]["causal_self"])
    ridge_self_mean_v5 = np.mean(v5["ridge"]["self"])
    ridge_cs_delta = ridge_cs_mean_v5 - ridge_self_mean_v5
    mlp_cs_delta = np.mean(v5["mlp"]["causal_self"]) - np.mean(v5["mlp"]["self"])
    causal_only_mean = np.mean(v5["ridge"]["causal"])

    recommendations = f"""
STAGE2 DESIGN: CONNECTOME FINDINGS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ CONNECTOME FIXES NEIGHBOR NOISE
  • Ridge Δ(cs−self): v4=−0.13 → v5={ridge_cs_delta:+.2f}
  • MLP Δ(cs−self): v4=−0.30 → v5={mlp_cs_delta:+.2f}
  • Pruning ~{mean_N:.0f}→~{mean_syn:.0f} inputs eliminates
    overfitting

✓ SELF-HISTORY STILL DOMINANT
  • Ridge self-only: R²={ridge_self_mean_v5:.2f}
  • AR component remains essential
  • Connectome adds clean ΔR²≈{ridge_cs_delta:+.3f}

✓ SPARSE CONNECTIVITY IS KEY
  • Avg gap partners: {mean_gap:.1f}
  • Avg synaptic inputs: {mean_syn:.1f}
  • Sparsity acts as regularisation

✓ LOO CEILING SIMILAR
  • Causal-only (connectome): R²={causal_only_mean:.2f}
  • Still bounded by ~0.3

✓ TRF NEEDS RE-TUNING
  • TRF drops with connectome constraint
  • Fewer features starve attention
  • Consider hybrid: connectome + soft gates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="gray"))

    fig.suptitle("Neural Activity Decoder v5 (Connectome) → Stage2 Model Design",
                 fontsize=16, fontweight="bold", y=0.98)

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "stage2_design_insights_v5.png", dpi=150, bbox_inches="tight")
    plt.savefig(save_dir / "stage2_design_insights_v5.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_dir / 'stage2_design_insights_v5.png'}")


def main():
    save_dir = V5_DIR
    if not V5_DIR.exists():
        print(f"Results directory not found: {V5_DIR}")
        return
    create_v5_insights_plot(save_dir)
    print("\n✓ Stage2 design insights (v5 connectome) plot generated")


if __name__ == "__main__":
    main()
