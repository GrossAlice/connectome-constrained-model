#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def _load_json(path: Path):
    return json.loads(path.read_text())


def _load_labels_from_h5(h5_path: Path) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        raw = f["gcamp/neuron_labels"][:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def _lookup_named_indices(labels: list[str], names: tuple[str, ...]) -> dict[str, int]:
    upper = [label.strip().upper() for label in labels]
    out: dict[str, int] = {}
    for name in names:
        key = name.strip().upper()
        if key not in upper:
            raise KeyError(f"Neuron {name!r} not found in labels")
        out[name] = upper.index(key)
    return out


def _load_ridge_named_points(results_path: Path, worm_id: str) -> dict[str, dict[str, float]]:
    data = _load_json(results_path)
    out: dict[str, dict[str, float]] = {}
    for row in data.get("results", []):
        if row.get("worm") != worm_id:
            continue
        out[row["neuron"]] = {
            "ridge-linear": float(row["r2"]["ridge-linear"]),
            "ridge-mlp": float(row["r2"]["ridge-mlp"]),
        }
    return out


def _parse_stage2_runlog_metrics(run_log_path: Path) -> dict[str, dict[str, float]]:
    loo: dict[str, float] = {}
    loo_windowed: dict[str, float] = {}
    pattern = re.compile(
        r"name='(?P<name>[^']+)'.*?r2=(?P<r2>[-+0-9.eE]+)(?:.*?r2_windowed=(?P<r2w>[-+0-9.eE]+))?"
    )
    for line in run_log_path.read_text().splitlines():
        if "[Stage2][loo_progress]" not in line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        name = match.group("name")
        loo[name] = float(match.group("r2"))
        if match.group("r2w") is not None:
            loo_windowed[name] = float(match.group("r2w"))
    return {"loo": loo, "loo_windowed": loo_windowed}


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _save_metric_overview(run_dir: Path, aggregate: list[dict]) -> None:
    if not aggregate:
        return

    metric_keys = [
        "onestep_r2",
        "loo_r2",
        "loo_windowed_r2",
        "free_run_r2",
        "beh_direct_r2",
        "beh_ridge_r2",
        "beh_gt_r2",
    ]
    labels = [
        "1-step",
        "LOO",
        "LOO win",
        "Free-run",
        "Beh direct",
        "Beh ridge",
        "Beh GT",
    ]

    vals = np.array([[row.get(k, np.nan) for k in metric_keys] for row in aggregate], dtype=float)
    means = np.nanmean(vals, axis=0)

    fig_w = max(9, 1.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, means, color=[
        "#4c78a8", "#f58518", "#e45756", "#54a24b", "#72b7b2", "#b279a2", "#ff9da6",
    ], edgecolor="black", linewidth=0.4)

    for row_i, row in enumerate(vals):
        jitter = np.linspace(-0.10, 0.10, len(aggregate)) if len(aggregate) > 1 else np.array([0.0])
        ax.scatter(x + jitter[row_i], row, color="black", s=18, alpha=0.55, zorder=3)

    for i, mean_val in enumerate(means):
        if np.isfinite(mean_val):
            ax.text(i, mean_val + 0.015, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title(f"Transformer CV summary ({len(aggregate)} worm{'s' if len(aggregate) != 1 else ''})")
    fig.tight_layout()
    fig.savefig(run_dir / "cv_metric_overview.png", bbox_inches="tight")
    plt.close(fig)


def _save_eval_summary(worm_dir: Path, eval_results: dict, train_meta: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    worm_id = eval_results["worm_id"]

    metric_names = ["1-step", "LOO", "LOO win", "Free-run", "Beh direct", "Beh ridge", "Beh GT"]
    metric_vals = [
        eval_results.get("onestep", {}).get("r2_mean", np.nan),
        eval_results.get("loo", {}).get("r2_mean", np.nan),
        eval_results.get("loo_windowed", {}).get("r2_mean", np.nan),
        eval_results.get("free_run", {}).get("r2_mean", np.nan),
        eval_results.get("behaviour_direct", {}).get("r2_mean", np.nan),
        eval_results.get("behaviour_ridge", {}).get("r2_model_mean", np.nan),
        eval_results.get("behaviour_ridge", {}).get("r2_gt_mean", np.nan),
    ]
    ax = axes[0, 0]
    x = np.arange(len(metric_names))
    ax.bar(x, metric_vals, color=["#4c78a8", "#f58518", "#e45756", "#54a24b", "#72b7b2", "#b279a2", "#ff9da6"], edgecolor="black", linewidth=0.4)
    for i, value in enumerate(metric_vals):
        if np.isfinite(value):
            ax.text(i, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Mean metrics")

    ax = axes[0, 1]
    onestep = _finite(eval_results.get("onestep", {}).get("r2", []))
    loo = _finite(eval_results.get("loo", {}).get("r2", []))
    freerun = _finite(eval_results.get("free_run", {}).get("r2", []))
    bins = np.linspace(min([arr.min() for arr in [onestep, loo, freerun] if arr.size] + [-0.5]),
                       max([arr.max() for arr in [onestep, loo, freerun] if arr.size] + [1.0]),
                       28)
    if onestep.size:
        ax.hist(onestep, bins=bins, alpha=0.45, label=f"1-step ({np.nanmean(onestep):.3f})", color="#4c78a8")
    if loo.size:
        ax.hist(loo, bins=bins, alpha=0.45, label=f"LOO ({np.nanmean(loo):.3f})", color="#f58518")
    if freerun.size:
        ax.hist(freerun, bins=bins, alpha=0.45, label=f"Free-run ({np.nanmean(freerun):.3f})", color="#54a24b")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Per-neuron R²")
    ax.set_ylabel("Count")
    ax.set_title("Neural R² distributions")
    ax.legend()

    ax = axes[1, 0]
    beh_direct = np.asarray(eval_results.get("behaviour_direct", {}).get("r2", []), dtype=float)
    beh_model = np.asarray(eval_results.get("behaviour_ridge", {}).get("r2_model", []), dtype=float)
    beh_gt = np.asarray(eval_results.get("behaviour_ridge", {}).get("r2_gt", []), dtype=float)
    n_modes = max(len(beh_direct), len(beh_model), len(beh_gt))
    if n_modes:
        modes = np.arange(n_modes)
        width = 0.26
        if len(beh_direct):
            ax.bar(modes - width, beh_direct, width, label="Direct head", color="#72b7b2")
        if len(beh_model):
            ax.bar(modes, beh_model, width, label="Ridge on model", color="#b279a2")
        if len(beh_gt):
            ax.bar(modes + width, beh_gt, width, label="Ridge on GT", color="#ff9da6")
        ax.set_xticks(modes)
        ax.set_xticklabels([f"a{i+1}" for i in modes])
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_ylabel("R²")
        ax.set_title("Behaviour by eigenworm mode")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No behaviour metrics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[1, 1]
    fold_losses = np.asarray(train_meta.get("fold_val_losses", []), dtype=float)
    fold_elapsed = np.asarray(train_meta.get("fold_elapsed", []), dtype=float)
    folds = np.arange(len(fold_losses))
    ax.bar(folds, fold_losses, color="#9ecae9", edgecolor="black", linewidth=0.4, label="Best val loss")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Best val loss")
    ax.set_title("Fold summary")
    ax2 = ax.twinx()
    if len(fold_elapsed):
        ax2.plot(folds, fold_elapsed, color="#d62728", marker="o", linewidth=1.3, label="Minutes")
    ax2.set_ylabel("Minutes")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.suptitle(f"Transformer CV evaluation — {worm_id}", fontsize=13)
    fig.tight_layout()
    fig.savefig(worm_dir / "cv_eval_summary.png", bbox_inches="tight")
    plt.close(fig)


def _save_training_curves(worm_dir: Path, histories: list[list[dict]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(histories), 1)))

    for fi, hist in enumerate(histories):
        if not hist:
            continue
        epochs = [row["epoch"] for row in hist]
        train_loss = [row.get("train_loss", np.nan) for row in hist]
        val_loss = [row.get("val_loss", np.nan) for row in hist]
        train_neural = [row.get("train_neural_nll", np.nan) for row in hist]
        val_neural = [row.get("val_neural_nll", np.nan) for row in hist]
        train_beh = [row.get("train_beh_nll", np.nan) for row in hist]
        val_beh = [row.get("val_beh_nll", np.nan) for row in hist]
        ss_p = [row.get("ss_p_gt", np.nan) for row in hist]
        color = colors[fi]

        axes[0, 0].plot(epochs, train_loss, color=color, alpha=0.85, label=f"fold {fi} train")
        axes[0, 0].plot(epochs, val_loss, color=color, linestyle="--", alpha=0.95, label=f"fold {fi} val")

        axes[0, 1].plot(epochs, train_neural, color=color, alpha=0.85)
        axes[0, 1].plot(epochs, val_neural, color=color, linestyle="--", alpha=0.95)

        axes[1, 0].plot(epochs, train_beh, color=color, alpha=0.85)
        axes[1, 0].plot(epochs, val_beh, color=color, linestyle="--", alpha=0.95)

        axes[1, 1].plot(epochs, ss_p, color=color, alpha=0.95, label=f"fold {fi}")

    axes[0, 0].set_title("Total loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(ncol=2, fontsize=8)

    axes[0, 1].set_title("Neural NLL")
    axes[0, 1].set_ylabel("NLL")

    axes[1, 0].set_title("Behaviour NLL")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("NLL")

    axes[1, 1].set_title("Scheduled sampling p(GT)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("p")
    axes[1, 1].legend(ncol=2, fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.2, linewidth=0.5)

    fig.suptitle(f"Transformer CV training curves — {worm_dir.name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(worm_dir / "cv_training_curves.png", bbox_inches="tight")
    plt.close(fig)


def _plot_fair_worm_comparison(
    transformer_run_dir: Path,
    worm_id: str,
    h5_path: Path,
    ridge_inclusive_path: Path,
    ridge_strict_path: Path,
    stage2_run_dir: Path,
    out_path: Path,
) -> None:
    worm_dir = transformer_run_dir / worm_id
    eval_results = _load_json(worm_dir / "eval_results.json")
    train_meta = _load_json(worm_dir / "train_meta.json")
    ridge_inclusive = _load_ridge_named_points(ridge_inclusive_path, worm_id)
    ridge_strict = _load_ridge_named_points(ridge_strict_path, worm_id)
    stage2_metrics = _parse_stage2_runlog_metrics(stage2_run_dir / "run.log")

    labels = _load_labels_from_h5(h5_path)
    named_idx = _lookup_named_indices(labels, ("AVAL", "AVAR"))
    transformer_onestep = np.asarray(eval_results["onestep"]["r2"], dtype=float)
    transformer_finite = transformer_onestep[np.isfinite(transformer_onestep)]

    neuron_colors = {"AVAL": "#2f6db3", "AVAR": "#58a5f0"}
    neuron_markers = {"AVAL": "D", "AVAR": "o"}

    fig, axes = plt.subplots(
        1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    ax = axes[0]

    categories = [
        "Transformer CV\n1-step OOF",
        "Ridge-MLP\ninclusive",
        "Ridge-MLP\nstrict",
        "Stage 2\nOOF 1-step",
    ]
    xpos = np.arange(len(categories), dtype=float)

    violin = ax.violinplot(
        [transformer_finite],
        positions=[xpos[0]],
        widths=0.62,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_edgecolor("#1f3552")
        body.set_alpha(0.30)
    violin["cmedians"].set_color("#1f3552")
    violin["cmedians"].set_linewidth(1.5)

    jitter = np.linspace(-0.18, 0.18, transformer_finite.size) if transformer_finite.size else np.array([])
    ax.scatter(
        np.full(transformer_finite.size, xpos[0]) + jitter,
        transformer_finite,
        s=10,
        color="#4c78a8",
        alpha=0.18,
        linewidths=0,
        zorder=2,
    )

    def add_named_points(x: float, values: dict[str, float], *, annotate: bool = True) -> None:
        for name in ("AVAL", "AVAR"):
            if name not in values or not np.isfinite(values[name]):
                continue
            y = values[name]
            ax.scatter(
                [x], [y],
                s=110,
                marker=neuron_markers[name],
                color=neuron_colors[name],
                edgecolor="white",
                linewidth=1.1,
                zorder=5,
            )
            if annotate:
                ax.text(x, y + 0.028, f"{y:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    transformer_named = {
        name: float(transformer_onestep[idx]) for name, idx in named_idx.items()
    }
    add_named_points(xpos[0], transformer_named)
    add_named_points(
        xpos[1],
        {name: ridge_inclusive.get(name, {}).get("ridge-mlp", np.nan) for name in ("AVAL", "AVAR")},
    )
    add_named_points(
        xpos[2],
        {name: ridge_strict.get(name, {}).get("ridge-mlp", np.nan) for name in ("AVAL", "AVAR")},
    )

    y_min = min(-0.45, float(np.nanmin(transformer_finite)) - 0.05 if transformer_finite.size else -0.45)
    y_max = max(1.0, float(np.nanmax(transformer_finite)) + 0.08 if transformer_finite.size else 1.0)
    placeholder = Rectangle(
        (xpos[3] - 0.30, y_min + 0.02),
        0.60,
        (y_max - y_min) * 0.86,
        facecolor="#d9d9d9",
        edgecolor="#777777",
        hatch="///",
        alpha=0.45,
        linewidth=0.9,
        zorder=1,
    )
    ax.add_patch(placeholder)
    ax.text(
        xpos[3],
        y_min + (y_max - y_min) * 0.48,
        "Need 5 Stage 2\nre-trains + stitched\nheld-out predictions",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
        fontweight="bold",
    )

    ax.set_title("Primary target: one-step neural R²\n(5-fold OOF where artifacts exist)", fontweight="bold")
    ax.set_ylabel("R²")
    ax.set_xticks(xpos)
    ax.set_xticklabels(categories)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    summary_text = (
        f"Transformer CV: n={train_meta['n_neural']} neurons, mean={eval_results['onestep']['r2_mean']:.3f}\n"
        f"Folds: {train_meta['n_folds']} contiguous blocks, K={train_meta['folds'][0]['test'][0]} start context"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    legend_handles = [
        Patch(facecolor="#4c78a8", edgecolor="#1f3552", alpha=0.30, label="Transformer CV 1-step OOF distribution"),
        Line2D([0], [0], marker=neuron_markers["AVAL"], color="w", markerfacecolor=neuron_colors["AVAL"], markeredgecolor="white", markersize=9, label="AVAL"),
        Line2D([0], [0], marker=neuron_markers["AVAR"], color="w", markerfacecolor=neuron_colors["AVAR"], markeredgecolor="white", markersize=9, label="AVAR"),
    ]
    ax.legend(handles=legend_handles, loc="lower left")

    ax2 = axes[1]
    stage2_loo = stage2_metrics.get("loo", {})
    stage2_loo_w = stage2_metrics.get("loo_windowed", {})
    loo_categories = ["Stage 2 LOO", "Stage 2 LOO\nwindowed (50)"]
    loo_x = np.arange(len(loo_categories), dtype=float)
    width = 0.33
    aval_vals = [stage2_loo.get("AVAL", np.nan), stage2_loo_w.get("AVAL", np.nan)]
    avar_vals = [stage2_loo.get("AVAR", np.nan), stage2_loo_w.get("AVAR", np.nan)]

    bars1 = ax2.bar(loo_x - width / 2, aval_vals, width, color=neuron_colors["AVAL"], label="AVAL")
    bars2 = ax2.bar(loo_x + width / 2, avar_vals, width, color=neuron_colors["AVAR"], label="AVAR")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_title("Secondary panel: available recursive LOO reference\n(not fold-matched)", fontweight="bold")
    ax2.set_ylabel("R²")
    ax2.set_xticks(loo_x)
    ax2.set_xticklabels(loo_categories)
    ax2.set_ylim(0.0, max(0.42, np.nanmax([*aval_vals, *avar_vals]) + 0.08))
    ax2.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax2.legend(loc="upper left")
    ax2.text(
        0.02,
        0.98,
        "Transformer CV LOO subset in attached run does not include AVAL/AVAR.\n"
        "Ridge-MLP omitted here because its saved result is next-frame masked prediction,\n"
        "not recursive held-out rollout.",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.suptitle(f"Model comparison — worm {worm_id} (artifact-aware update)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_run(run_dir: Path) -> None:
    aggregate_path = run_dir / "aggregate_results.json"
    aggregate = _load_json(aggregate_path) if aggregate_path.exists() else []
    _save_metric_overview(run_dir, aggregate)

    for worm_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        eval_path = worm_dir / "eval_results.json"
        meta_path = worm_dir / "train_meta.json"
        if not eval_path.exists() or not meta_path.exists():
            continue

        eval_results = _load_json(eval_path)
        train_meta = _load_json(meta_path)
        histories = []
        for history_path in sorted(worm_dir.glob("history_fold*.json")):
            histories.append(_load_json(history_path))

        _save_eval_summary(worm_dir, eval_results, train_meta)
        _save_training_curves(worm_dir, histories)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved outputs from a transformer CV run")
    parser.add_argument("run_dir", type=Path, help="Run directory containing aggregate_results.json and per-worm subdirs")
    parser.add_argument("--comparison-worm", type=str, default=None, help="Optional worm ID for the artifact-aware comparison plot")
    parser.add_argument("--h5-path", type=Path, default=None, help="H5 file used to resolve neuron labels for the comparison plot")
    parser.add_argument("--ridge-inclusive", type=Path, default=None, help="Path to same-worm inclusive ridge-MLP results.json")
    parser.add_argument("--ridge-strict", type=Path, default=None, help="Path to same-worm strict ridge-MLP results.json")
    parser.add_argument("--stage2-run-dir", type=Path, default=None, help="Stage 2 run directory whose run.log provides current LOO reference values")
    parser.add_argument("--out-path", type=Path, default=None, help="Output path for the comparison figure")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    plot_run(run_dir)

    if args.comparison_worm is not None:
        missing = []
        if args.h5_path is None:
            missing.append("--h5-path")
        if args.ridge_inclusive is None:
            missing.append("--ridge-inclusive")
        if args.ridge_strict is None:
            missing.append("--ridge-strict")
        if args.stage2_run_dir is None:
            missing.append("--stage2-run-dir")
        if args.out_path is None:
            missing.append("--out-path")
        if missing:
            raise SystemExit(f"Comparison plot requested but missing required args: {', '.join(missing)}")

        _plot_fair_worm_comparison(
            transformer_run_dir=run_dir,
            worm_id=args.comparison_worm,
            h5_path=args.h5_path.resolve(),
            ridge_inclusive_path=args.ridge_inclusive.resolve(),
            ridge_strict_path=args.ridge_strict.resolve(),
            stage2_run_dir=args.stage2_run_dir.resolve(),
            out_path=args.out_path.resolve(),
        )


if __name__ == "__main__":
    main()