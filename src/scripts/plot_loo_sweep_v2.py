#!/usr/bin/env python3
"""
Plot LOO sweep v2 results.

Reads sweep_results.json + per-neuron cv_onestep.npz files from
output_plots/stage2/loo_sweep_v2 and produces a multi-panel summary figure.

Usage:
    python -m scripts.plot_loo_sweep_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ─────────────────────────────────────────────────────────────────
SWEEP_DIR = Path("output_plots/stage2/loo_sweep_v2")
SAVE_DIR  = SWEEP_DIR          # save the figure next to the data


def load_results(sweep_dir: Path) -> list[dict]:
    with open(sweep_dir / "sweep_results.json") as f:
        return json.load(f)


def load_per_neuron_loo(sweep_dir: Path, tag: str) -> np.ndarray | None:
    """Return per-neuron LOO R² from cv_onestep.npz (finite values only)."""
    npz_path = sweep_dir / tag / "cv_onestep.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    if "cv_loo_r2" not in d:
        return None
    arr = d["cv_loo_r2"]
    return arr[np.isfinite(arr)]


def main():
    results = load_results(SWEEP_DIR)

    # ── Filter to the primary worm and drop degenerate (R²=1) runs ────────
    worm = "2022-08-02-01"
    rows = [
        r for r in results
        if r["worm"] == worm
        and r.get("cv_loo_r2_mean") is not None
        and r["cv_loo_r2_mean"] < 0.99        # drop no_kernels artefacts
        and r.get("cv_onestep_r2_median") is not None
    ]

    # Sort by LOO windowed median descending
    rows.sort(key=lambda r: r.get("cv_loo_r2_windowed_median", 0), reverse=True)

    labels = [r["condition"] for r in rows]
    n      = len(labels)

    # ── Metric arrays ─────────────────────────────────────────────────────
    onestep_med    = np.array([r["cv_onestep_r2_median"] for r in rows])
    loo_med        = np.array([r["cv_loo_r2_median"]     for r in rows])
    loo_w_med      = np.array([r["cv_loo_r2_windowed_median"] for r in rows])
    loo_mean       = np.array([r["cv_loo_r2_mean"]       for r in rows])
    loo_w_mean     = np.array([r["cv_loo_r2_windowed_mean"] for r in rows])
    n_pos          = np.array([r.get("loo_n_positive", 0) for r in rows])
    q25            = np.array([r.get("loo_q25", 0)        for r in rows])
    q75            = np.array([r.get("loo_q75", 0)        for r in rows])

    # ── Per-neuron LOO distributions ──────────────────────────────────────
    per_neuron = []
    for r in rows:
        arr = load_per_neuron_loo(SWEEP_DIR, r["tag"])
        per_neuron.append(arr if arr is not None else np.array([]))

    # ── Highlight baseline ────────────────────────────────────────────────
    baseline_idx = labels.index("baseline") if "baseline" in labels else None
    colors = []
    for i, lab in enumerate(labels):
        if lab == "baseline":
            colors.append("#FF7043")       # orange-red for baseline
        elif loo_w_med[i] > loo_w_med[baseline_idx] if baseline_idx is not None else False:
            colors.append("#66BB6A")       # green beats baseline
        else:
            colors.append("#42A5F5")       # blue

    # ══════════════════════════════════════════════════════════════════════
    #  Figure
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.42, wspace=0.30,
        left=0.08, right=0.96, top=0.94, bottom=0.06,
    )

    # ── Panel A: One-step R² (median) ─────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    bars = ax_a.barh(range(n), onestep_med, color=colors, edgecolor="black",
                     linewidth=0.5)
    ax_a.set_yticks(range(n))
    ax_a.set_yticklabels(labels, fontsize=9)
    ax_a.set_xlabel("One-step R² (median)", fontsize=10)
    ax_a.set_title("A. One-step prediction quality", fontsize=12, fontweight="bold")
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.65, max(onestep_med) + 0.02)
    for i, v in enumerate(onestep_med):
        ax_a.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)

    # ── Panel B: LOO R² median vs windowed median ────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(n)
    w = 0.35
    bars1 = ax_b.barh(x - w / 2, loo_med, w, label="LOO R² median",
                       color=[c + "88" for c in colors],
                       edgecolor="black", linewidth=0.5)
    bars2 = ax_b.barh(x + w / 2, loo_w_med, w, label="Windowed LOO R² median",
                       color=colors, edgecolor="black", linewidth=0.5)
    ax_b.set_yticks(x)
    ax_b.set_yticklabels(labels, fontsize=9)
    ax_b.set_xlabel("LOO R² (median)", fontsize=10)
    ax_b.set_title("B. Leave-one-out prediction (median)", fontsize=12,
                   fontweight="bold")
    ax_b.invert_yaxis()
    ax_b.legend(fontsize=8, loc="lower right")
    for i, v in enumerate(loo_w_med):
        ax_b.text(v + 0.002, i + w / 2, f"{v:.3f}", va="center", fontsize=7)

    # ── Panel C: Per-neuron LOO R² box-plots ──────────────────────────────
    ax_c = fig.add_subplot(gs[1, :])
    bp = ax_c.boxplot(
        per_neuron, vert=True, patch_artist=True, widths=0.6,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_c.set_xticks(range(1, n + 1))
    ax_c.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax_c.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_c.set_ylabel("Per-neuron LOO R²", fontsize=10)
    ax_c.set_title("C. Per-neuron LOO R² distribution by condition",
                   fontsize=12, fontweight="bold")

    # ── Panel D: Δ LOO windowed median vs baseline ────────────────────────
    ax_d = fig.add_subplot(gs[2, 0])
    if baseline_idx is not None:
        bl_val = loo_w_med[baseline_idx]
        delta  = loo_w_med - bl_val
        bar_colors = ["#66BB6A" if d > 0 else "#EF5350" for d in delta]
        bar_colors[baseline_idx] = "#BDBDBD"
        ax_d.barh(range(n), delta, color=bar_colors, edgecolor="black",
                  linewidth=0.5)
        ax_d.axvline(0, color="black", lw=0.8)
        ax_d.set_yticks(range(n))
        ax_d.set_yticklabels(labels, fontsize=9)
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Δ windowed LOO R² (vs baseline)", fontsize=10)
        ax_d.set_title("D. Improvement over baseline", fontsize=12,
                       fontweight="bold")
        for i, d in enumerate(delta):
            ha = "left" if d >= 0 else "right"
            ax_d.text(d + 0.002 * np.sign(d), i, f"{d:+.4f}", va="center",
                      fontsize=8, ha=ha)

    # ── Panel E: Scatter – one-step vs LOO ────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 1])
    ax_e.scatter(onestep_med, loo_w_med, c=colors, s=80, edgecolors="black",
                 linewidths=0.6, zorder=3)
    for i, lab in enumerate(labels):
        ax_e.annotate(
            lab, (onestep_med[i], loo_w_med[i]),
            textcoords="offset points", xytext=(6, 4), fontsize=7,
            alpha=0.85,
        )
    ax_e.set_xlabel("One-step R² (median)", fontsize=10)
    ax_e.set_ylabel("Windowed LOO R² (median)", fontsize=10)
    ax_e.set_title("E. One-step vs LOO trade-off", fontsize=12,
                   fontweight="bold")
    ax_e.grid(True, alpha=0.3)

    # ── Suptitle ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"LOO Sweep v2 — worm {worm}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = SAVE_DIR / "loo_sweep_v2_summary.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    # ── Also print a ranked table ─────────────────────────────────────────
    print(f"\n{'Condition':<25} {'1-step':>7} {'LOO-w':>7} {'LOO':>7} "
          f"{'Δ LOO-w':>8} {'#pos':>5}")
    print("-" * 65)
    bl = loo_w_med[baseline_idx] if baseline_idx is not None else 0
    for i in range(n):
        d = loo_w_med[i] - bl
        print(f"{labels[i]:<25} {onestep_med[i]:>7.4f} {loo_w_med[i]:>7.4f} "
              f"{loo_med[i]:>7.4f} {d:>+8.4f} {n_pos[i]:>5}")


if __name__ == "__main__":
    main()
