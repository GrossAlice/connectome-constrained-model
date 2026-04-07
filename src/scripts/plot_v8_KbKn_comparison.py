#!/usr/bin/env python3
"""
Compare v8_KbKn results between two runs:
  • OLD  = v8_KbKn  (parsed from log_*_motor.txt files)
  • NEW  = v8_KbKn_current  (loaded from results_*_motor_KbKn.json)

Generates:
  1. Bar chart: mean R² and Corr per model, old vs new  (one panel per K)
  2. Scatter plot: old vs new R² per worm  (best models)
  3. Delta violin: (new − old) distribution across worms
  4. Lag-sweep comparison: R² vs K for each model × condition
"""

import json, re, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════════════

MODEL_KEYS = {
    "Ridge n+b FR":  "ridge_nb_fr",
    "Ridge n FR":    "ridge_n_fr",
    "Ridge b 1s":    "ridge_b_1s",
    "MLP n+b FR":    "mlp_nb_fr",
    "MLP n FR":      "mlp_n_fr",
    "MLP b 1s":      "mlp_b_1s",
    "TRF n+b FR":    "trf_large_nb_fr",
    "TRF n FR":      "trf_large_n_fr",
    "TRF b 1s":      "trf_large_b_1s",
}

DISPLAY_NAMES = {v: k for k, v in MODEL_KEYS.items()}
DISPLAY_NAMES["trf_large_b_1s"] = "TRF b (1-step)"
DISPLAY_NAMES["mlp_b_1s"]       = "MLP b (1-step)"
DISPLAY_NAMES["ridge_b_1s"]     = "Ridge b (1-step)"


def parse_log_file(path: Path) -> dict:
    """Parse a log_*_motor.txt file → {K_str: {model_key: {r2_mean, corr_mean}}}."""
    text = path.read_text()
    result = {}
    current_K = None

    # Match K_n=X, K_b=X header
    k_pattern = re.compile(r"K_n=(\d+),\s*K_b=(\d+)")
    # Match model result line:  "    Ridge n+b FR (reclamp=10)..." or "    Ridge n FR (direct)..." etc.
    # followed by "      R²=0.350  corr=0.597"
    model_pattern = re.compile(r"^\s+(Ridge|MLP|TRF)\s+(n\+b FR|n FR|b 1s)")
    r2_pattern = re.compile(r"R²=([\-\d.]+)\s+corr=([\-\d.]+)")

    lines = text.splitlines()
    pending_model = None

    for line in lines:
        km = k_pattern.search(line)
        if km:
            current_K = km.group(1)  # K_n value (same as K_b)
            if current_K not in result:
                result[current_K] = {}
            continue

        mm = model_pattern.search(line)
        if mm and current_K is not None:
            model_name = f"{mm.group(1)} {mm.group(2)}"
            pending_model = MODEL_KEYS.get(model_name)
            continue

        rm = r2_pattern.search(line)
        if rm and pending_model is not None and current_K is not None:
            r2_val = float(rm.group(1))
            corr_val = float(rm.group(2))
            result[current_K][pending_model] = {
                "r2_mean": r2_val,
                "corr_mean": corr_val,
            }
            pending_model = None

    return result


def load_old_results(log_dir: Path) -> dict:
    """Load all old log results → {worm_id: {K_str: {model_key: ...}}}."""
    results = {}
    for f in sorted(log_dir.glob("log_*_motor.txt")):
        worm_id = f.stem.replace("log_", "").replace("_motor", "")
        results[worm_id] = parse_log_file(f)
    return results


def load_new_results(json_dir: Path) -> dict:
    """Load all new JSON results → {worm_id: {K_str: {model_key: ...}}}."""
    results = {}
    for f in sorted(json_dir.glob("results_*_motor_KbKn.json")):
        worm_id = f.stem.replace("results_", "").replace("_motor_KbKn", "")
        with open(f) as fp:
            results[worm_id] = json.load(fp)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODELS = [
    "trf_large_nb_fr", "trf_large_n_fr", "trf_large_b_1s",
    "mlp_nb_fr", "mlp_n_fr", "mlp_b_1s",
    "ridge_nb_fr", "ridge_n_fr", "ridge_b_1s",
]

MODEL_COLORS = {
    "trf_large_nb_fr": "#E57373", "trf_large_n_fr": "#EF9A9A", "trf_large_b_1s": "#FFCDD2",
    "mlp_nb_fr": "#64B5F6", "mlp_n_fr": "#90CAF9", "mlp_b_1s": "#BBDEFB",
    "ridge_nb_fr": "#81C784", "ridge_n_fr": "#A5D6A7", "ridge_b_1s": "#C8E6C9",
}

ARCH_COLORS = {"TRF": "#E57373", "MLP": "#64B5F6", "Ridge": "#81C784"}
COND_STYLES = {
    "nb_fr": ("-", "o"),
    "n_fr":  ("-", "s"),
    "b_1s":  ("--", "^"),
}


def _common_worms(old, new):
    return sorted(set(old.keys()) & set(new.keys()))


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Grouped bar chart: old vs new, per model, per K
# ══════════════════════════════════════════════════════════════════════════════

def plot_bar_comparison(old, new, K_values, out_dir):
    """Side-by-side bar: old (hatched) vs new (solid) for R² and Corr."""
    worms = _common_worms(old, new)
    n_worms = len(worms)

    for K in K_values:
        Ks = str(K)
        old_r2 = {m: [] for m in ALL_MODELS}
        new_r2 = {m: [] for m in ALL_MODELS}
        old_corr = {m: [] for m in ALL_MODELS}
        new_corr = {m: [] for m in ALL_MODELS}

        for w in worms:
            od = old[w].get(Ks, {})
            nd = new[w].get(Ks, {})
            for m in ALL_MODELS:
                if m in od:
                    old_r2[m].append(od[m]["r2_mean"])
                    old_corr[m].append(od[m]["corr_mean"])
                if m in nd:
                    new_r2[m].append(nd[m]["r2_mean"])
                    new_corr[m].append(nd[m]["corr_mean"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        x = np.arange(len(ALL_MODELS))
        w_bar = 0.35

        for ax, old_d, new_d, ylabel, title in [
            (ax1, old_r2, new_r2, "Mean R²", "R²"),
            (ax2, old_corr, new_corr, "Mean Correlation", "Correlation"),
        ]:
            old_means = [np.mean(old_d[m]) if old_d[m] else 0 for m in ALL_MODELS]
            old_stds  = [np.std(old_d[m])  if old_d[m] else 0 for m in ALL_MODELS]
            new_means = [np.mean(new_d[m]) if new_d[m] else 0 for m in ALL_MODELS]
            new_stds  = [np.std(new_d[m])  if new_d[m] else 0 for m in ALL_MODELS]

            colors = [MODEL_COLORS[m] for m in ALL_MODELS]

            bars_old = ax.bar(x - w_bar / 2, old_means, w_bar, yerr=old_stds,
                              color=colors, alpha=0.45, hatch="//", edgecolor="gray",
                              label="Old (v8_KbKn)", capsize=2)
            bars_new = ax.bar(x + w_bar / 2, new_means, w_bar, yerr=new_stds,
                              color=colors, alpha=0.9, edgecolor="black", linewidth=0.5,
                              label="New (v8_KbKn_current)", capsize=2)

            ax.set_xticks(x)
            ax.set_xticklabels([DISPLAY_NAMES[m] for m in ALL_MODELS],
                               rotation=40, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=9)

        fig.suptitle(f"Old vs New — K={K}, Motor neurons, {n_worms} worms",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = out_dir / f"compare_bar_K{K}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Scatter: old R² vs new R² per worm (one dot per worm)
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter_old_vs_new(old, new, K_values, out_dir):
    """Scatter plot of old vs new R²/Corr per worm for key models."""
    worms = _common_worms(old, new)

    key_models = ["trf_large_nb_fr", "mlp_nb_fr", "ridge_nb_fr",
                   "trf_large_b_1s", "mlp_b_1s", "ridge_b_1s"]

    for K in K_values:
        Ks = str(K)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        for col, m in enumerate(key_models):
            row = 0 if col < 3 else 1
            c = col if col < 3 else col - 3
            ax = axes[row, c] if col < 3 else axes[1, col - 3]

            old_vals, new_vals = [], []
            for w in worms:
                ov = old[w].get(Ks, {}).get(m, {}).get("r2_mean")
                nv = new[w].get(Ks, {}).get(m, {}).get("r2_mean")
                if ov is not None and nv is not None:
                    old_vals.append(ov)
                    new_vals.append(nv)

            color = MODEL_COLORS[m]
            ax.scatter(old_vals, new_vals, c=color, s=40, alpha=0.7, edgecolor="black", linewidth=0.5)
            lo = min(min(old_vals, default=0), min(new_vals, default=0)) - 0.05
            hi = max(max(old_vals, default=1), max(new_vals, default=1)) + 0.05
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel("Old R²")
            ax.set_ylabel("New R²")
            ax.set_title(DISPLAY_NAMES[m], fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3)

            # Annotate mean delta
            if old_vals:
                delta = np.mean(new_vals) - np.mean(old_vals)
                sign = "+" if delta >= 0 else ""
                ax.text(0.05, 0.92, f"Δ={sign}{delta:.3f}",
                        transform=ax.transAxes, fontsize=9,
                        color="green" if delta >= 0 else "red",
                        fontweight="bold")

        fig.suptitle(f"Per-worm R² — Old vs New  (K={K}, motor neurons, {len(worms)} worms)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = out_dir / f"compare_scatter_K{K}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Delta violin/box: (new − old) across worms, per model
# ══════════════════════════════════════════════════════════════════════════════

def plot_delta_violin(old, new, K_values, out_dir):
    """Violin + strip plot showing ΔR² (new-old) and ΔCorr per model."""
    worms = _common_worms(old, new)

    for K in K_values:
        Ks = str(K)

        delta_r2 = {m: [] for m in ALL_MODELS}
        delta_corr = {m: [] for m in ALL_MODELS}

        for w in worms:
            od = old[w].get(Ks, {})
            nd = new[w].get(Ks, {})
            for m in ALL_MODELS:
                if m in od and m in nd:
                    delta_r2[m].append(nd[m]["r2_mean"] - od[m]["r2_mean"])
                    delta_corr[m].append(nd[m]["corr_mean"] - od[m]["corr_mean"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        for ax, data, ylabel, title in [
            (ax1, delta_r2, "Δ R² (new − old)", "ΔR²"),
            (ax2, delta_corr, "Δ Corr (new − old)", "ΔCorrelation"),
        ]:
            positions = np.arange(len(ALL_MODELS))
            vals = [data[m] for m in ALL_MODELS]
            colors = [MODEL_COLORS[m] for m in ALL_MODELS]

            # Box plot (more robust with small N)
            bp = ax.boxplot(vals, positions=positions, widths=0.5,
                           patch_artist=True, showfliers=False)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)

            # Strip points
            for i, m in enumerate(ALL_MODELS):
                jitter = np.random.uniform(-0.12, 0.12, len(data[m]))
                ax.scatter(positions[i] + jitter, data[m], c=colors[i],
                          s=20, alpha=0.7, edgecolor="black", linewidth=0.3, zorder=3)

            ax.axhline(0, color="black", linewidth=1, linestyle="-")
            ax.set_xticks(positions)
            ax.set_xticklabels([DISPLAY_NAMES[m] for m in ALL_MODELS],
                               rotation=40, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            # Annotate means
            for i, m in enumerate(ALL_MODELS):
                if data[m]:
                    mn = np.mean(data[m])
                    ax.plot(i, mn, "D", color="white", markersize=6,
                            markeredgecolor="black", markeredgewidth=1.2, zorder=4)

        fig.suptitle(f"Improvement: New − Old  (K={K}, motor neurons, {len(worms)} worms)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = out_dir / f"compare_delta_K{K}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Lag sweep: R² vs K for each model, old vs new
# ══════════════════════════════════════════════════════════════════════════════

def plot_lag_sweep_comparison(old, new, K_values, out_dir):
    """Line plot: R² and Corr vs K for each model, solid=new, dashed=old."""
    worms = _common_worms(old, new)

    cond_display = {"nb_fr": "n+b FR", "n_fr": "n FR", "b_1s": "b (1-step)"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (cond_suffix, cond_label) in enumerate(cond_display.items()):
        for row, metric in enumerate(["r2_mean", "corr_mean"]):
            ax = axes[row, col]
            ylabel = "Mean R²" if row == 0 else "Mean Correlation"

            for arch, arch_color in ARCH_COLORS.items():
                model_key = f"{arch.lower()}_{'large_' if arch == 'TRF' else ''}{cond_suffix}"
                if arch == "TRF":
                    model_key = f"trf_large_{cond_suffix}"

                old_means = []
                new_means = []
                old_stds = []
                new_stds = []

                for K in K_values:
                    Ks = str(K)
                    ov = [old[w].get(Ks, {}).get(model_key, {}).get(metric)
                          for w in worms]
                    nv = [new[w].get(Ks, {}).get(model_key, {}).get(metric)
                          for w in worms]
                    ov = [x for x in ov if x is not None]
                    nv = [x for x in nv if x is not None]
                    old_means.append(np.mean(ov) if ov else np.nan)
                    new_means.append(np.mean(nv) if nv else np.nan)
                    old_stds.append(np.std(ov) if ov else 0)
                    new_stds.append(np.std(nv) if nv else 0)

                # Old = dashed
                ax.plot(K_values, old_means, "--", color=arch_color, alpha=0.5,
                        marker="o", markersize=5, label=f"{arch} (old)")
                # New = solid
                ax.plot(K_values, new_means, "-", color=arch_color, alpha=0.9,
                        marker="s", markersize=5, label=f"{arch} (new)")
                ax.fill_between(K_values,
                                np.array(new_means) - np.array(new_stds),
                                np.array(new_means) + np.array(new_stds),
                                color=arch_color, alpha=0.1)

            ax.set_xticks(K_values)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(cond_label, fontsize=12, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Context length K")
            if col == 2:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Lag Sweep — Old vs New  (motor neurons, {len(worms)} worms)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = out_dir / "compare_lag_sweep.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Summary table printed to console + saved as image
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_table(old, new, K_values, out_dir):
    """Table image showing mean R² and Corr for old vs new, and the delta."""
    worms = _common_worms(old, new)

    rows = []
    for K in K_values:
        Ks = str(K)
        for m in ALL_MODELS:
            old_r2 = [old[w].get(Ks, {}).get(m, {}).get("r2_mean")
                      for w in worms]
            new_r2 = [new[w].get(Ks, {}).get(m, {}).get("r2_mean")
                      for w in worms]
            old_corr = [old[w].get(Ks, {}).get(m, {}).get("corr_mean")
                        for w in worms]
            new_corr = [new[w].get(Ks, {}).get(m, {}).get("corr_mean")
                        for w in worms]

            old_r2 = [x for x in old_r2 if x is not None]
            new_r2 = [x for x in new_r2 if x is not None]
            old_corr = [x for x in old_corr if x is not None]
            new_corr = [x for x in new_corr if x is not None]

            or2 = np.mean(old_r2) if old_r2 else np.nan
            nr2 = np.mean(new_r2) if new_r2 else np.nan
            oc = np.mean(old_corr) if old_corr else np.nan
            nc = np.mean(new_corr) if new_corr else np.nan

            rows.append({
                "K": K,
                "Model": DISPLAY_NAMES[m],
                "Old R²": or2,
                "New R²": nr2,
                "ΔR²": nr2 - or2,
                "Old Corr": oc,
                "New Corr": nc,
                "ΔCorr": nc - oc,
            })

    # Print
    print(f"\n{'='*90}")
    print(f"  COMPARISON SUMMARY  ({len(worms)} worms)")
    print(f"{'='*90}")
    hdr = f"{'K':>3}  {'Model':<22}  {'Old R²':>7}  {'New R²':>7}  {'ΔR²':>7}  {'Old Corr':>9}  {'New Corr':>9}  {'ΔCorr':>7}"
    print(hdr)
    print("-" * len(hdr))
    prev_K = None
    for r in rows:
        if r["K"] != prev_K:
            if prev_K is not None:
                print()
            prev_K = r["K"]
        delta_r2_str = f"{r['ΔR²']:+.3f}"
        delta_c_str = f"{r['ΔCorr']:+.3f}"
        print(f"{r['K']:>3}  {r['Model']:<22}  {r['Old R²']:7.3f}  {r['New R²']:7.3f}  {delta_r2_str:>7}  {r['Old Corr']:9.3f}  {r['New Corr']:9.3f}  {delta_c_str:>7}")

    # Save as figure table
    fig, ax = plt.subplots(figsize=(16, 2 + 0.35 * len(rows)))
    ax.axis("off")

    col_labels = ["K", "Model", "Old R²", "New R²", "ΔR²", "Old Corr", "New Corr", "ΔCorr"]
    cell_text = []
    cell_colors = []
    for r in rows:
        dr2 = r["ΔR²"]
        dc = r["ΔCorr"]
        row_text = [
            str(r["K"]),
            r["Model"],
            f"{r['Old R²']:.3f}",
            f"{r['New R²']:.3f}",
            f"{dr2:+.3f}",
            f"{r['Old Corr']:.3f}",
            f"{r['New Corr']:.3f}",
            f"{dc:+.3f}",
        ]
        cell_text.append(row_text)
        # Color delta cells
        row_colors = ["white"] * 8
        row_colors[4] = "#c8e6c9" if dr2 > 0.005 else "#ffcdd2" if dr2 < -0.005 else "white"
        row_colors[7] = "#c8e6c9" if dc > 0.005 else "#ffcdd2" if dc < -0.005 else "white"
        cell_colors.append(row_colors)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="center", loc="center",
                     cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#424242")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle(f"Old vs New Comparison — Motor neurons, {len(worms)} worms",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / "compare_summary_table.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ROOT = Path(__file__).resolve().parent.parent
    old_dir = ROOT / "output_plots" / "behaviour_decoder" / "v8_KbKn"
    new_dir = ROOT / "output_plots" / "behaviour_decoder" / "v8_KbKn_current"
    out_dir = ROOT / "output_plots" / "behaviour_decoder" / "v8_KbKn_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Old results dir: {old_dir}")
    print(f"New results dir: {new_dir}")
    print(f"Output dir:      {out_dir}")

    old = load_old_results(old_dir)
    new = load_new_results(new_dir)

    common = _common_worms(old, new)
    print(f"\nOld worms: {len(old)},  New worms: {len(new)},  Common: {len(common)}")

    if not common:
        print("ERROR: no common worms found!")
        sys.exit(1)

    K_VALUES = [1, 5, 10, 15]

    print("\n── Generating comparison plots ──\n")
    plot_bar_comparison(old, new, K_VALUES, out_dir)
    plot_scatter_old_vs_new(old, new, K_VALUES, out_dir)
    plot_delta_violin(old, new, K_VALUES, out_dir)
    plot_lag_sweep_comparison(old, new, K_VALUES, out_dir)
    plot_summary_table(old, new, K_VALUES, out_dir)

    print("\n✓ All comparison plots saved to:", out_dir)


if __name__ == "__main__":
    main()
