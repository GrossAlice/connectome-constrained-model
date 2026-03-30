#!/usr/bin/env python
r"""Post-process ar2mlp_wd_cv results: print statistics + generate video.

Usage:
    python scripts/ar2mlp_wd_cv_postprocess.py --worm 2022-06-14-07
    python scripts/ar2mlp_wd_cv_postprocess.py --worm 2023-01-17-14
    python scripts/ar2mlp_wd_cv_postprocess.py --worm 2022-06-14-07 --compare 2023-01-17-14
"""
from __future__ import annotations

import argparse, json, sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"
OUT_BASE = ROOT / "output_plots/behaviour_decoder/ar2mlp_wd_cv"


def load_worm_results(worm_id: str):
    """Load predictions.npz and wd_cv_summary.json for a worm."""
    wdir = OUT_BASE / worm_id
    npz = np.load(wdir / "predictions.npz")
    with open(wdir / "wd_cv_summary.json") as f:
        summary = json.load(f)
    return npz, summary


# ── Statistics ────────────────────────────────────────────────────

def print_statistics(worm_id: str, summary: dict, npz):
    """Print comprehensive statistics table."""
    r2_cv = np.array(summary["r2_cv"])
    r2_fixed = np.array(summary["r2_fixed"])
    K = len(r2_cv)

    print(f"\n{'═'*80}")
    print(f"  STATISTICS: {worm_id}")
    print(f"{'═'*80}")

    # Per-mode R²
    mode_names = [f"a{j+1}" for j in range(K)]
    print(f"\n  Per-mode R² (aggregate over all folds):")
    header = f"    {'Model':<25s}" + "".join(f"  {m:>6s}" for m in mode_names)
    header += f"  {'mean':>7s}"
    print(header)
    print(f"    {'─'*65}")
    for label, r2 in [("AR2+MLP (wd-CV)", r2_cv),
                      ("AR2+MLP (wd=1e-3)", r2_fixed)]:
        row = f"    {label:<25s}"
        for v in r2:
            row += f"  {v:6.3f}"
        row += f"  {np.mean(r2):7.3f}"
        print(row)

    delta = r2_cv - r2_fixed
    row = f"    {'Δ (CV − fixed)':<25s}"
    for v in delta:
        row += f"  {v:+6.3f}"
    row += f"  {np.mean(delta):+7.3f}"
    print(row)

    # Per-fold breakdown
    print(f"\n  Per-fold mean R² and Δ:")
    has_fold_r2 = "r2_cv_fold" in summary["fold_results"][0]
    if has_fold_r2:
        fold_r2s_cv = np.array([fr["r2_cv_fold"] for fr in summary["fold_results"]])
        fold_r2s_fixed = np.array([fr["r2_fixed_fold"] for fr in summary["fold_results"]])
    else:
        # Compute from predictions on the fly
        ho_cv = npz["ho_cv"]
        ho_fixed = npz["ho_fixed"]
        b_true = npz["b_true"]
        T_full = b_true.shape[0]
        warmup = 8  # default
        valid_len = T_full - warmup
        n_folds = len(summary["fold_results"])
        fold_size = valid_len // n_folds
        fold_r2s_cv_list, fold_r2s_fixed_list = [], []
        for fi in range(n_folds):
            ts = warmup + fi * fold_size
            te = warmup + (fi + 1) * fold_size if fi < n_folds - 1 else T_full
            for preds, out_list in [(ho_cv, fold_r2s_cv_list),
                                     (ho_fixed, fold_r2s_fixed_list)]:
                seg = preds[ts:te]
                ok = np.all(np.isfinite(seg), axis=1)
                if ok.any():
                    from scripts.benchmark_ar_decoder_v2 import r2_score as _r2
                    r2s = [_r2(b_true[ts:te, j][ok], seg[ok, j]) for j in range(K)]
                else:
                    r2s = [np.nan] * K
                out_list.append(r2s)
        fold_r2s_cv = np.array(fold_r2s_cv_list)
        fold_r2s_fixed = np.array(fold_r2s_fixed_list)

    fold_diff = fold_r2s_cv.mean(axis=1) - fold_r2s_fixed.mean(axis=1)
    se = fold_diff.std(ddof=1) / np.sqrt(len(fold_diff))

    for i in range(len(summary["fold_results"])):
        fr = summary["fold_results"][i]
        cv_m = float(np.mean(fold_r2s_cv[i]))
        fx_m = float(np.mean(fold_r2s_fixed[i]))
        print(f"    Fold {i+1}: CV={cv_m:.3f}  fixed={fx_m:.3f}  "
              f"Δ={cv_m - fx_m:+.3f}  wd={fr['best_wd']:.0e}")

    print(f"\n  Mean Δ(CV − fixed) = {fold_diff.mean():+.4f} ± {se:.4f} "
          f"(SE, n={len(fold_diff)})")
    t_stat = fold_diff.mean() / (se + 1e-12)
    print(f"  t-statistic = {t_stat:.2f}")

    # Selected wd summary
    best_wds = [fr["best_wd"] for fr in summary["fold_results"]]
    print(f"\n  Selected wd across folds:")
    print(f"    values: {best_wds}")
    print(f"    median: {np.median(best_wds):.1e}  "
          f"range: [{min(best_wds):.1e}, {max(best_wds):.1e}]")

    # Prediction variance ratio
    ho_cv = npz["ho_cv"]
    ho_fixed = npz["ho_fixed"]
    b = npz["b_true"]
    ok_cv = np.all(np.isfinite(ho_cv), axis=1)
    ok_fx = np.all(np.isfinite(ho_fixed), axis=1)
    print(f"\n  Variance ratio σ_pred/σ_GT:")
    for label, preds, ok in [("wd-CV", ho_cv, ok_cv),
                              ("wd=1e-3", ho_fixed, ok_fx)]:
        srs = [preds[ok, j].std() / b[ok, j].std() for j in range(K)]
        print(f"    {label:<15s}: "
              + "  ".join(f"a{j+1}={sr:.3f}" for j, sr in enumerate(srs))
              + f"  mean={np.mean(srs):.3f}")

    # MSE landscape
    wd_grid = summary["wd_grid"]
    n_folds = len(summary["fold_results"])
    print(f"\n  Inner-val MSE landscape (×1000):")
    header = f"    {'wd':>10s}"
    for fi in range(n_folds):
        header += f"  {'f'+str(fi+1):>8s}"
    header += f"  {'mean':>8s}"
    print(header)
    print(f"    {'─'*65}")
    for wd in wd_grid:
        row = f"    {wd:10.1e}"
        vals = []
        for fr in summary["fold_results"]:
            v = fr["wd_mses"][str(wd)] * 1000
            vals.append(v)
            row += f"  {v:8.2f}"
        row += f"  {np.mean(vals):8.2f}"
        print(row)

    return r2_cv, r2_fixed


# ── Comparison across worms ───────────────────────────────────────

def compare_worms(worm_ids: list[str]):
    """Print side-by-side comparison table."""
    print(f"\n{'═'*80}")
    print(f"  CROSS-WORM COMPARISON")
    print(f"{'═'*80}\n")

    rows = []
    for wid in worm_ids:
        npz_w, summary_w = load_worm_results(wid)
        r2_cv = np.array(summary_w["r2_cv"])
        r2_fx = np.array(summary_w["r2_fixed"])

        # Compute fold-level R² (handle old JSONs without r2_cv_fold)
        if "r2_cv_fold" in summary_w["fold_results"][0]:
            fold_r2s_cv = np.array([fr["r2_cv_fold"]
                                    for fr in summary_w["fold_results"]])
            fold_r2s_fx = np.array([fr["r2_fixed_fold"]
                                    for fr in summary_w["fold_results"]])
        else:
            # Recompute from npz
            from scripts.benchmark_ar_decoder_v2 import r2_score as _r2
            b_w = npz_w["b_true"]
            T_w, K_w = b_w.shape
            warmup_w = 8
            n_f = len(summary_w["fold_results"])
            fs = (T_w - warmup_w) // n_f
            cv_list, fx_list = [], []
            for fi in range(n_f):
                ts = warmup_w + fi * fs
                te = warmup_w + (fi + 1) * fs if fi < n_f - 1 else T_w
                for preds_key, out in [("ho_cv", cv_list), ("ho_fixed", fx_list)]:
                    seg = npz_w[preds_key][ts:te]
                    ok = np.all(np.isfinite(seg), axis=1)
                    r = [_r2(b_w[ts:te, j][ok], seg[ok, j]) for j in range(K_w)] if ok.any() else [np.nan]*K_w
                    out.append(r)
            fold_r2s_cv = np.array(cv_list)
            fold_r2s_fx = np.array(fx_list)

        fd = fold_r2s_cv.mean(axis=1) - fold_r2s_fx.mean(axis=1)
        se = fd.std(ddof=1) / np.sqrt(len(fd))
        rows.append({
            "worm": wid,
            "cv_mean": np.mean(r2_cv),
            "fx_mean": np.mean(r2_fx),
            "delta": np.mean(r2_cv) - np.mean(r2_fx),
            "fold_delta_mean": fd.mean(),
            "fold_delta_se": se,
            "best_wds": [fr["best_wd"] for fr in summary_w["fold_results"]],
        })

    header = f"  {'Worm':<20s}  {'CV R²':>7s}  {'Fix R²':>7s}  {'Δ':>7s}  {'Δ±SE':>14s}  {'Median wd':>10s}"
    print(header)
    print(f"  {'─'*75}")
    for r in rows:
        med_wd = np.median(r["best_wds"])
        print(f"  {r['worm']:<20s}  {r['cv_mean']:7.3f}  {r['fx_mean']:7.3f}  "
              f"{r['delta']:+7.3f}  {r['fold_delta_mean']:+.3f}±{r['fold_delta_se']:.3f}  "
              f"{med_wd:10.1e}")


# ── Plots ─────────────────────────────────────────────────────────

def make_timeseries_plot(worm_id: str, npz, out_dir: pathlib.Path):
    """Time-series overlay plot: GT vs wd-CV vs fixed."""
    b = npz["b_true"]
    ho_cv = npz["ho_cv"]
    ho_fixed = npz["ho_fixed"]
    T, K = b.shape

    n_modes = min(K, 6)
    fig, axes = plt.subplots(n_modes, 1, figsize=(16, 2.5 * n_modes),
                             sharex=True)
    if n_modes == 1:
        axes = [axes]

    t = np.arange(T)
    for j, ax in enumerate(axes):
        ax.plot(t, b[:, j], "k-", lw=0.7, alpha=0.7, label="Ground truth")
        ok_cv = np.isfinite(ho_cv[:, j])
        ok_fx = np.isfinite(ho_fixed[:, j])
        ax.plot(t[ok_cv], ho_cv[ok_cv, j], "-", color="#2196F3", lw=0.8,
                alpha=0.8, label="wd-CV")
        ax.plot(t[ok_fx], ho_fixed[ok_fx, j], "-", color="#E91E63", lw=0.8,
                alpha=0.8, label="wd=1e-3")
        ax.set_ylabel(f"a{j+1}", fontsize=11)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    axes[-1].set_xlabel("Frame", fontsize=11)
    fig.suptitle(f"AR2+MLP wd-CV vs fixed — {worm_id}", fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "timeseries_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def make_mse_landscape_plot(worm_id: str, summary: dict, out_dir: pathlib.Path):
    """MSE landscape plot across wd values per fold."""
    wd_grid = summary["wd_grid"]
    n_folds = len(summary["fold_results"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for fi, fr in enumerate(summary["fold_results"]):
        mses = [fr["wd_mses"][str(wd)] for wd in wd_grid]
        ax.plot(range(len(wd_grid)), mses, "o-", lw=1.5, ms=4,
                label=f"Fold {fi+1} (best={fr['best_wd']:.0e})", alpha=0.8)

    # Mean curve
    mean_mses = []
    for wd in wd_grid:
        vals = [fr["wd_mses"][str(wd)] for fr in summary["fold_results"]]
        mean_mses.append(np.mean(vals))
    ax.plot(range(len(wd_grid)), mean_mses, "k-", lw=2.5, ms=6,
            marker="s", label="Mean", zorder=10)

    ax.set_xticks(range(len(wd_grid)))
    ax.set_xticklabels([f"{wd:.0e}" for wd in wd_grid], rotation=45, fontsize=8)
    ax.set_xlabel("Weight decay", fontsize=11)
    ax.set_ylabel("Inner-val MSE", fontsize=11)
    ax.set_title(f"MSE landscape — {worm_id}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "mse_landscape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def make_r2_bar_plot(worm_id: str, summary: dict, out_dir: pathlib.Path):
    """Grouped bar chart: R² per mode, wd-CV vs fixed."""
    r2_cv = np.array(summary["r2_cv"])
    r2_fx = np.array(summary["r2_fixed"])
    K = len(r2_cv)
    modes = [f"a{j+1}" for j in range(K)]

    fig, ax = plt.subplots(figsize=(max(8, K * 1.2), 4.5))
    x = np.arange(K)
    w = 0.35
    bars1 = ax.bar(x - w/2, r2_cv, w, label="wd-CV", color="#2196F3",
                   edgecolor="white", lw=0.5)
    bars2 = ax.bar(x + w/2, r2_fx, w, label="wd=1e-3 fixed", color="#E91E63",
                   edgecolor="white", lw=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.set_ylabel("R² (held-out, free-run)", fontsize=11)
    ax.set_title(f"AR2+MLP wd-CV vs fixed — {worm_id}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "r2_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Video ─────────────────────────────────────────────────────────

def make_video(worm_id: str, npz, out_dir: pathlib.Path, max_frames=200):
    """Generate posture comparison video."""
    from scripts.benchmark_ar_decoder_v2 import make_comparison_video

    h5_path = str(DATA_DIR / f"{worm_id}.h5")
    b_gt = npz["b_true"]
    ho_cv = npz["ho_cv"]
    ho_fixed = npz["ho_fixed"]

    # Find a good contiguous segment (middle fold typically)
    # Use frames where both predictions are valid
    ok = np.all(np.isfinite(ho_cv), axis=1) & np.all(np.isfinite(ho_fixed), axis=1)
    valid_idx = np.where(ok)[0]
    if len(valid_idx) == 0:
        print("  ⚠ No valid frames for video!")
        return

    # Find longest contiguous run
    diffs = np.diff(valid_idx)
    breaks = np.where(diffs > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(valid_idx)]])
    lengths = ends - starts
    best_run = np.argmax(lengths)
    seg_start = valid_idx[starts[best_run]]
    seg_end = valid_idx[ends[best_run] - 1] + 1
    seg_len = seg_end - seg_start

    print(f"  Video segment: [{seg_start}:{seg_end}) ({seg_len} frames)")

    # Slice to segment
    T_vid = min(seg_len, max_frames)
    s, e = seg_start, seg_start + T_vid
    b_seg = b_gt[s:e]
    predictions = {
        "AR2+MLP (wd-CV)": ho_cv[s:e],
        "AR2+MLP (wd=1e-3)": ho_fixed[s:e],
    }

    out_path = str(out_dir / "posture_comparison.mp4")
    make_comparison_video(
        h5_path, out_path,
        b_seg, predictions,
        dt=0.6, fps=15, dpi=100, max_frames=T_vid,
        body_angle_offset=s,
    )


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-process ar2mlp_wd_cv: statistics + plots + video")
    parser.add_argument("--worm", required=True,
                        help="Worm ID (h5 stem)")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="Additional worm IDs for cross-worm comparison")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip video generation")
    parser.add_argument("--max_frames", type=int, default=200,
                        help="Max frames for video (default 200)")
    args = parser.parse_args()

    worm_id = args.worm
    wdir = OUT_BASE / worm_id

    print(f"Loading results for {worm_id} …")
    npz, summary = load_worm_results(worm_id)

    # Statistics
    print_statistics(worm_id, summary, npz)

    # Plots
    print(f"\n  Generating plots …")
    make_timeseries_plot(worm_id, npz, wdir)
    make_mse_landscape_plot(worm_id, summary, wdir)
    make_r2_bar_plot(worm_id, summary, wdir)

    # Cross-worm comparison
    if args.compare:
        compare_worms([worm_id] + args.compare)

    # Video
    if not args.no_video:
        print(f"\n  Generating posture comparison video …")
        make_video(worm_id, npz, wdir, max_frames=args.max_frames)

    print(f"\n  All outputs → {wdir}/")
    print("  Done.")


if __name__ == "__main__":
    main()
