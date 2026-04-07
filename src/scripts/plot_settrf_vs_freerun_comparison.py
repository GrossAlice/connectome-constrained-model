"""
Compare Set Transformer Joint free-run results vs original free-run (MLP/Ridge/TRF × strategies).
Produces:
  1. Per-worm bar charts for the 3 overlapping worms
  2. Aggregate summary table across all 40 SetTRF worms
  3. Side-by-side ensemble plot grid
"""
from __future__ import annotations
import json, pathlib, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETTRF_DIR = ROOT / "output_plots" / "free_run_settrf"
FREERUN_DIR = ROOT / "output_plots" / "free_run"
OUT_DIR = ROOT / "output_plots" / "free_run_settrf" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_LABELS = {
    "psd_log_distance": "PSD log-dist ↓",
    "autocorr_rmse": "ACF RMSE ↓",
    "wasserstein_1": "Wasserstein-1 ↓",
    "ks_statistic": "KS stat ↓",
    "variance_ratio_mean": "Var ratio (≈1)",
    "mean_abs_error": "Mean |err| ↓",
}

OVERLAP_WORMS = ["2022-06-14-01", "2022-06-14-07", "2022-06-14-13"]

# Models to compare from the original free-run
ORIGINAL_MODELS = [
    "Ridge Joint", "MLP Joint", "TRF Joint",
    "Ridge AR+Dec", "MLP AR+Dec", "TRF AR+Dec",
    "Ridge Cascaded", "MLP Cascaded", "TRF Cascaded",
]

# ── helpers ──────────────────────────────────────────────────────────────
def load_json(p: pathlib.Path):
    with open(p) as f:
        return json.load(f)


def _bar_color(model: str) -> str:
    """Color by model family."""
    if "SetTRF" in model:
        return "#e63946"
    if "Ridge" in model:
        return "#457b9d"
    if "MLP" in model:
        return "#2a9d8f"
    if "TRF" in model:
        return "#e9c46a"
    return "#adb5bd"


def _short_name(m: str) -> str:
    return m.replace("Joint", "J").replace("Cascaded", "C").replace("AR+Dec", "A")


# ── 1. Per-worm metric comparison bar charts ─────────────────────────────
def plot_per_worm_bars(worm_id: str):
    settrf_path = SETTRF_DIR / worm_id / "motor" / "results.json"
    freerun_path = FREERUN_DIR / worm_id / "motor" / "results.json"
    if not settrf_path.exists() or not freerun_path.exists():
        return

    settrf = load_json(settrf_path)
    freerun = load_json(freerun_path)

    # Collect model→metrics
    models = {}
    for mname in ORIGINAL_MODELS:
        if mname in freerun["metrics_T1"]:
            models[mname] = freerun["metrics_T1"][mname]
    models["SetTRF Joint"] = settrf["metrics_T1"]

    metric_keys = list(METRIC_LABELS.keys())
    n_metrics = len(metric_keys)
    n_models = len(models)
    model_names = list(models.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, mk in enumerate(metric_keys):
        ax = axes[i]
        vals = [models[m].get(mk, 0) for m in model_names]
        colors = [_bar_color(m) for m in model_names]
        short = [_short_name(m) for m in model_names]
        bars = ax.bar(range(n_models), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[mk], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        # Highlight best
        if mk == "variance_ratio_mean":
            best_idx = np.argmin(np.abs(np.array(vals) - 1.0))
        else:
            best_idx = np.argmin(vals)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    fig.suptitle(f"Free-run comparison — {worm_id} (motor)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / f"metrics_comparison_{worm_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.relative_to(ROOT)}")


# ── 2. Per-worm variance ratio per-dim comparison ──────────────────────
def plot_vardim_comparison(worm_id: str):
    settrf_path = SETTRF_DIR / worm_id / "motor" / "results.json"
    freerun_path = FREERUN_DIR / worm_id / "motor" / "results.json"
    if not settrf_path.exists() or not freerun_path.exists():
        return

    settrf = load_json(settrf_path)
    freerun = load_json(freerun_path)

    models_vr = {}
    for mname in ["Ridge Joint", "TRF Joint", "MLP Joint"]:
        if mname in freerun["metrics_T1"]:
            vr = freerun["metrics_T1"][mname].get("variance_ratio_per_dim", [])
            if vr:
                models_vr[mname] = vr
    models_vr["SetTRF Joint"] = settrf["metrics_T1"].get("variance_ratio_per_dim", [])

    if not models_vr:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(6)
    w = 0.8 / len(models_vr)
    for j, (mname, vr) in enumerate(models_vr.items()):
        offset = (j - len(models_vr) / 2 + 0.5) * w
        color = _bar_color(mname)
        ax.bar(x + offset, vr, w, label=mname, color=color, edgecolor="white")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"EW{i+1}" for i in range(6)])
    ax.set_ylabel("Variance Ratio (ideal=1)")
    ax.set_yscale("log")
    ax.set_title(f"Per-EW-dim variance ratio — {worm_id}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"vardim_comparison_{worm_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.relative_to(ROOT)}")


# ── 3. Side-by-side ensemble images ─────────────────────────────────────
def plot_ensemble_sidebyside(worm_id: str):
    settrf_ens = SETTRF_DIR / worm_id / "motor" / f"ensemble_SetTRF_Joint_{worm_id}.png"
    # find TRF_Joint from original
    freerun_ens = FREERUN_DIR / worm_id / "motor" / f"ensemble_TRF_Joint_{worm_id}.png"

    if not settrf_ens.exists() or not freerun_ens.exists():
        return

    img1 = Image.open(freerun_ens)
    img2 = Image.open(settrf_ens)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    ax1.imshow(img1)
    ax1.set_title(f"TRF Joint (original)", fontsize=12, fontweight="bold")
    ax1.axis("off")
    ax2.imshow(img2)
    ax2.set_title(f"SetTRF Joint (new)", fontsize=12, fontweight="bold")
    ax2.axis("off")
    fig.suptitle(f"Ensemble comparison — {worm_id} (motor)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / f"ensemble_sidebyside_{worm_id}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.relative_to(ROOT)}")


# ── 4. Side-by-side PSD / ACF / marginals ───────────────────────────────
def plot_diagnostic_sidebyside(worm_id: str, kind: str):
    settrf_p = SETTRF_DIR / worm_id / "motor" / f"{kind}_{worm_id}.png"
    freerun_p = FREERUN_DIR / worm_id / "motor" / f"{kind}_{worm_id}.png"

    if not settrf_p.exists() or not freerun_p.exists():
        return

    img1 = Image.open(freerun_p)
    img2 = Image.open(settrf_p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    ax1.imshow(img1)
    ax1.set_title(f"Original free-run ({kind})", fontsize=12, fontweight="bold")
    ax1.axis("off")
    ax2.imshow(img2)
    ax2.set_title(f"SetTRF Joint ({kind})", fontsize=12, fontweight="bold")
    ax2.axis("off")
    fig.suptitle(f"{kind.upper()} comparison — {worm_id} (motor)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / f"{kind}_sidebyside_{worm_id}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.relative_to(ROOT)}")


# ── 5. Neural traces + EW traces grid (SetTRF only, all 40 worms) ──────
def plot_settrf_traces_gallery():
    """Create a gallery of neural trace plots and EW trace plots (4×5 grids)."""
    settrf_worms = sorted([
        d.name for d in SETTRF_DIR.iterdir()
        if d.is_dir() and (d / "motor" / "results.json").exists()
    ])

    for kind, label in [("neural_traces", "Neural Activity Traces"),
                         ("ew_traces", "Eigenworm Traces")]:
        images = []
        names = []
        for wid in settrf_worms:
            p = SETTRF_DIR / wid / "motor" / f"{kind}_{wid}.png"
            if p.exists():
                images.append(Image.open(p))
                names.append(wid)

        if not images:
            continue

        n = len(images)
        ncols = 5
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
        if nrows == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

        for i, (img, name) in enumerate(zip(images, names)):
            axes_flat[i].imshow(img)
            axes_flat[i].set_title(name, fontsize=8)
            axes_flat[i].axis("off")
        for j in range(len(images), len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"SetTRF Joint — {label} (all worms, motor)", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = OUT_DIR / f"gallery_{kind}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.relative_to(ROOT)}")


# ── 6. Aggregate summary: SetTRF vs original Joint models ───────────────
def plot_aggregate_comparison():
    """Box plot of SetTRF (40 worms) vs original Joint models (3 worms)."""
    # Load all SetTRF results
    settrf_summary = load_json(SETTRF_DIR / "batch_summary.json")

    # Load original free-run results for overlapping worms
    original_data = {}  # model_name → {metric → [values]}
    for wid in OVERLAP_WORMS:
        p = FREERUN_DIR / wid / "motor" / "results.json"
        if not p.exists():
            continue
        d = load_json(p)
        for mname in ["Ridge Joint", "MLP Joint", "TRF Joint"]:
            if mname in d["metrics_T1"]:
                if mname not in original_data:
                    original_data[mname] = {k: [] for k in METRIC_LABELS}
                for k in METRIC_LABELS:
                    original_data[mname][k].append(d["metrics_T1"][mname].get(k, np.nan))

    # SetTRF data
    settrf_metrics = {k: [] for k in METRIC_LABELS}
    for d in settrf_summary:
        for k in METRIC_LABELS:
            settrf_metrics[k].append(d["metrics_T1"].get(k, np.nan))

    # SetTRF restricted to same 3 worms
    settrf_3 = {k: [] for k in METRIC_LABELS}
    for d in settrf_summary:
        if d["worm_id"] in OVERLAP_WORMS:
            for k in METRIC_LABELS:
                settrf_3[k].append(d["metrics_T1"].get(k, np.nan))

    metric_keys = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, mk in enumerate(metric_keys):
        ax = axes[i]
        # Prepare data for box plot
        box_data = []
        box_labels = []
        box_colors = []

        # Original models (3-worm values)
        for mname in ["Ridge Joint", "MLP Joint", "TRF Joint"]:
            if mname in original_data:
                box_data.append(original_data[mname][mk])
                box_labels.append(_short_name(mname) + "\n(3w)")
                box_colors.append(_bar_color(mname))

        # SetTRF on same 3 worms
        box_data.append(settrf_3[mk])
        box_labels.append("SetTRF J\n(3w)")
        box_colors.append("#e63946")

        # SetTRF all 40 worms
        box_data.append(settrf_metrics[mk])
        box_labels.append("SetTRF J\n(40w)")
        box_colors.append("#c1121f")

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white", markersize=5))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(box_labels, fontsize=8)
        ax.set_title(METRIC_LABELS[mk], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        if mk == "variance_ratio_mean":
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_yscale("log")

    fig.suptitle("Free-run comparison: Original Joint models vs SetTRF Joint",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "aggregate_boxplot_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.relative_to(ROOT)}")


# ── 7. Print summary table ──────────────────────────────────────────────
def print_summary_table():
    settrf_summary = load_json(SETTRF_DIR / "batch_summary.json")

    # Collect original Joint metrics for the 3 overlap worms
    orig_joint = {k: [] for k in METRIC_LABELS}
    for wid in OVERLAP_WORMS:
        p = FREERUN_DIR / wid / "motor" / "results.json"
        if not p.exists():
            continue
        d = load_json(p)
        for mname in ["TRF Joint"]:
            if mname in d["metrics_T1"]:
                for k in METRIC_LABELS:
                    orig_joint[k].append(d["metrics_T1"][mname].get(k, np.nan))

    # SetTRF on same 3 worms
    settrf_3 = {k: [] for k in METRIC_LABELS}
    settrf_all = {k: [] for k in METRIC_LABELS}
    for d in settrf_summary:
        for k in METRIC_LABELS:
            settrf_all[k].append(d["metrics_T1"].get(k, np.nan))
        if d["worm_id"] in OVERLAP_WORMS:
            for k in METRIC_LABELS:
                settrf_3[k].append(d["metrics_T1"].get(k, np.nan))

    print("\n" + "=" * 85)
    print(f"{'Metric':<22} {'TRF Joint (3w)':>16} {'SetTRF J (3w)':>16} {'SetTRF J (40w)':>16}")
    print("=" * 85)
    for k in METRIC_LABELS:
        o = np.array(orig_joint[k])
        s3 = np.array(settrf_3[k])
        sa = np.array(settrf_all[k])
        print(f"{METRIC_LABELS[k]:<22} "
              f"{o.mean():>7.3f}±{o.std():>5.3f}  "
              f"{s3.mean():>7.3f}±{s3.std():>5.3f}  "
              f"{sa.mean():>7.3f}±{sa.std():>5.3f}")
    print("=" * 85)


# ── main ─────────────────────────────────────────────────────────────────
def main():
    print("=== SetTRF Joint vs Original Free-Run Comparison ===\n")

    # Per-worm comparisons (3 overlapping worms)
    print("1) Per-worm metric bar charts:")
    for wid in OVERLAP_WORMS:
        plot_per_worm_bars(wid)

    print("\n2) Per-worm variance-ratio per EW dim:")
    for wid in OVERLAP_WORMS:
        plot_vardim_comparison(wid)

    print("\n3) Ensemble side-by-side:")
    for wid in OVERLAP_WORMS:
        plot_ensemble_sidebyside(wid)

    print("\n4) Diagnostic side-by-side (PSD, ACF, marginals):")
    for wid in OVERLAP_WORMS:
        for kind in ["psd", "autocorr", "marginals"]:
            plot_diagnostic_sidebyside(wid, kind)

    print("\n5) Trace galleries (all 40 worms):")
    plot_settrf_traces_gallery()

    print("\n6) Aggregate box plot:")
    plot_aggregate_comparison()

    print("\n7) Summary table:")
    print_summary_table()

    print(f"\nAll comparison plots saved to: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
