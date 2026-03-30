#!/usr/bin/env python
"""
Render a comparison video for the top-1 worm: ground truth + Ridge + MLP + AR2+MLP + AR2d+MLP.
Re-runs all 4 models with 5-fold CV, collects hold-out predictions,
then reconstructs body-angle posture from eigenworms and animates.
"""
import sys, pathlib, numpy as np, h5py
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.unified_benchmark import (
    _ridge_fit, _train_mlp, _predict_mlp, _train_e2e,
    _log_ridge_grid,
)
from scripts.benchmark_ar_decoder_v2 import (
    load_data, build_lagged_features_np,
)
from stage2.posture_videos import angles_to_xy, _load_eigenvectors
from stage1.add_stephens_eigenworms import _preprocess_worm, _get_d_w, _ensure_TN

# ── Config ────────────────────────────────────────────────────────
H5 = ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-17-14.h5"
OUT_DIR = ROOT / "output_plots/behaviour_decoder/top1_video"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 6
N_LAGS = 8
N_FOLDS = 5
E2E_EPOCHS = 200
TBPTT = 64
MAX_FRAMES = 600   # 600 frames * 0.6s = 6 min of video
FPS = 10
MODELS = ["Ridge", "MLP", "AR2+MLP"]
COLORS = {"Ridge": "#1b9e77", "MLP": "#d95f02",
          "AR2+MLP": "#7570b3"}

# ── Check for saved predictions (skip training if available) ──────
PRED_PATH = OUT_DIR / "predictions.npz"
SKIP_TRAIN = PRED_PATH.exists()

# ── Load data ─────────────────────────────────────────────────────
print("Loading data …")
u, b_full, dt = load_data(str(H5), all_neurons=False)
K = min(K, b_full.shape[1])
b = b_full[:, :K]
T = b.shape[0]
warmup = max(2, N_LAGS)
X = build_lagged_features_np(u, N_LAGS)
d_in = X.shape[1]
print(f"  T={T}, N_motor={u.shape[1]}, K={K}, d_in={d_in}, dt={dt:.3f}s")

if SKIP_TRAIN:
    print(f"\nLoading saved predictions from {PRED_PATH} …")
    _saved = np.load(str(PRED_PATH))
    ho = {m: _saved[m] for m in MODELS}
    del _saved
else:
    # ── 5-fold temporal CV ────────────────────────────────────────────
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = N_FOLDS - 1
    valid_len = T - warmup
    fold_size = valid_len // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < N_FOLDS - 1 else T
        folds.append((s, e))

    ho = {m: np.full((T, K), np.nan) for m in MODELS}

    for fi, (ts, te) in enumerate(folds):
        print(f"\n── Fold {fi+1}/{N_FOLDS}  test=[{ts}:{te}] ──")
        tr_idx = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        X_tr, X_te = X[tr_idx], X[ts:te]
        b_tr = b[tr_idx]
        mu, sig = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sig
        X_te_s = (X_te - mu) / sig

        # Ridge
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j], ridge_grid, n_inner)
            ho["Ridge"][ts:te, j] = X_te_s @ coef + intc

        # MLP
        mlp = _train_mlp(X_tr_s, b_tr, K)
        ho["MLP"][ts:te] = _predict_mlp(mlp, X_te_s)

        # E2E segments
        segs = []
        if ts > warmup + 2:
            segs.append((warmup, ts))
        if te + 2 < T:
            segs.append((te, T))

        # AR2+MLP (full matrix, raw inputs)
        M1e, M2e, ce_np, drv_np = _train_e2e(
            d_in, K, segs, b, X,
            E2E_EPOCHS, TBPTT,
            weight_decay=1e-3, tag=f"E2E f{fi+1}")
        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1e @ p1 + M2e @ p2 + drv_np[t] + ce_np
            ho["AR2+MLP"][t] = p_new
            p2, p1 = p1, p_new

        # AR2d+MLP (diagonal, raw inputs)
        M1d, M2d, cd_np, drvd_np = _train_e2e(
            d_in, K, segs, b, X,
            E2E_EPOCHS, TBPTT,
            weight_decay=1e-3,
            diagonal_ar=True, max_rho=0.98,
            tag=f"E2Ed f{fi+1}")
        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1d @ p1 + M2d @ p2 + drvd_np[t] + cd_np
            ho["AR2d+MLP"][t] = p_new
        p2, p1 = p1, p_new

# ── R² ────────────────────────────────────────────────────────────
from sklearn.metrics import r2_score
valid = np.arange(warmup, T)
print("\n── R² per mode ──")
print(f"{'Model':14s}", "  ".join(f"{'a'+str(j+1):>7s}" for j in range(K)), "  mean")
for m in MODELS:
    ok = np.isfinite(ho[m][valid, 0])
    idx = valid[ok]
    r2s = [r2_score(b[idx, j], ho[m][idx, j]) for j in range(K)]
    mn = np.mean(r2s)
    print(f"{m:14s}", "  ".join(f"{r:.3f}" for r in r2s), f"  {mn:.3f}")

# ── Save predictions ─────────────────────────────────────────────
np.savez(OUT_DIR / "predictions.npz",
         b_true=b, **{m: ho[m] for m in MODELS})
print(f"\nSaved predictions to {OUT_DIR / 'predictions.npz'}")

# ── Load eigenvectors & compute per-frame mean ──────────────────
with h5py.File(str(H5), "r") as f:
    _ew_d_target = int(f["behaviour/eigenworms_stephens"].attrs["d_target"])
    _ew_d_w      = int(f["behaviour/eigenworms_stephens"].attrs["d_w"])
    _ew_source   = f["behaviour/eigenworms_stephens"].attrs["source"]
    ew_raw = np.asarray(f["behaviour/eigenworms_stephens"][:, :K], dtype=float)
    # Load the SOURCE body angles used for eigenworm computation
    ba_src = _ensure_TN(np.asarray(f[_ew_source][:], dtype=float))

eigvecs_all = _load_eigenvectors(h5_path=str(H5), d_target=_ew_d_target)
E = eigvecs_all[:, :K]  # (d_target, K) = (100, 6)
d_recon = E.shape[0]    # 100

# Reproduce the same preprocessing as add_stephens_eigenworms.py
# to recover the per-frame mean body angle that was subtracted.
proc = _preprocess_worm(ba_src, _ew_d_w, _ew_d_target)  # (T, 100)
per_frame_mean = proc.mean(axis=1, keepdims=True)        # (T, 1)
print(f"  Per-frame mean angle: [{per_frame_mean.min():.3f}, {per_frame_mean.max():.3f}] rad")

# ── Reconstruct body angles from eigenworms + restore mean ───────
Tmax = min(T, MAX_FRAMES)
recon = {}
recon["GT"] = ew_raw[:Tmax, :K] @ E.T + per_frame_mean[:Tmax]
for m_name in MODELS:
    recon[m_name] = ho[m_name][:Tmax, :K] @ E.T + per_frame_mean[:Tmax]

# Ground-truth preprocessed body angles (100-dim, NOT mean-subtracted)
proc_gt = proc[:Tmax]  # (Tmax, 100)

# ── Convert to XY ────────────────────────────────────────────────
panels = ["GT"] + MODELS
xy = {}
for p in panels:
    arr = np.zeros((Tmax, d_recon, 2), dtype=float)
    for t in range(Tmax):
        a = recon[p][t]
        if np.all(np.isfinite(a)):
            x, y = angles_to_xy(a)
        else:
            x = np.full(d_recon, np.nan)
            y = np.full(d_recon, np.nan)
        arr[t, :, 0], arr[t, :, 1] = x, y
    xy[p] = arr

# ── Preprocessed body angles XY for GT panel overlay (100-dim) ───
xy_raw = np.zeros((Tmax, d_recon, 2), dtype=float)
for t in range(Tmax):
    a = proc_gt[t]
    if np.all(np.isfinite(a)):
        x, y = angles_to_xy(a)
    else:
        x = np.full(d_recon, np.nan)
        y = np.full(d_recon, np.nan)
    xy_raw[t, :, 0], xy_raw[t, :, 1] = x, y

# ── Axis limits ───────────────────────────────────────────────────
all_coords = np.concatenate([xy[p].reshape(-1, 2) for p in panels] + [xy_raw.reshape(-1, 2)])
m_ok = np.all(np.isfinite(all_coords), axis=1)
xmin, xmax = all_coords[m_ok, 0].min(), all_coords[m_ok, 0].max()
ymin, ymax = all_coords[m_ok, 1].min(), all_coords[m_ok, 1].max()
cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
span = 0.5 * max(xmax - xmin, ymax - ymin) + 2.0

# ── R² for subtitle ──────────────────────────────────────────────
r2_mean = {}
for m2 in MODELS:
    ok = np.isfinite(ho[m2][valid, 0])
    idx = valid[ok]
    r2_mean[m2] = np.mean([r2_score(b[idx, j], ho[m2][idx, j]) for j in range(K)])

# ── Figure ────────────────────────────────────────────────────────
n_panels = 1 + len(MODELS)
fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5), facecolor="white")
panel_labels = ["Ground Truth"] + [f"{m}\n(R²={r2_mean[m]:.3f})" for m in MODELS]
panel_colors = ["black"] + [COLORS[m] for m in MODELS]

lines, heads = [], []
line_raw, head_raw = None, None

for i, (ax, ttl, col) in enumerate(zip(axes, panel_labels, panel_colors)):
    ax.set_title(ttl, fontsize=10, fontweight="bold", color=col if i > 0 else "black")
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f7f7")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_visible(False)
    lw = 2.5 if i == 0 else 2.0
    c = "black" if i == 0 else col
    ln, = ax.plot([], [], "-", color=c, lw=lw, alpha=0.9)
    hd, = ax.plot([], [], "o", color="crimson", ms=5, alpha=0.9)
    lines.append(ln)
    heads.append(hd)
    if i == 0:
        # Also show preprocessed body angles (100-dim) behind eigenworm recon
        line_raw, = ax.plot([], [], "-", color="gray", lw=1.5, alpha=0.4)
        head_raw, = ax.plot([], [], "o", color="gray", ms=3, alpha=0.4)

time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=11)

panel_data = [("GT", xy)] + [(m, xy) for m in MODELS]

def _update(frame):
    # GT panel: show raw body angles + eigenworm reconstruction
    xr, yr = xy_raw[frame, :, 0], xy_raw[frame, :, 1]
    line_raw.set_data(xr, yr)
    if np.isfinite(xr[0]) and np.isfinite(yr[0]):
        head_raw.set_data([xr[0]], [yr[0]])
    else:
        head_raw.set_data([], [])

    for i, key in enumerate(panels):
        x = xy[key][frame, :, 0]
        y = xy[key][frame, :, 1]
        lines[i].set_data(x, y)
        if np.isfinite(x[0]) and np.isfinite(y[0]):
            heads[i].set_data([x[0]], [y[0]])
        else:
            heads[i].set_data([], [])
    time_text.set_text(f"t = {frame * dt:.1f} s    frame {frame + 1}/{Tmax}")
    return lines + heads + [line_raw, head_raw, time_text]

print(f"\nRendering {Tmax} frames at {FPS} fps …")
anim = FuncAnimation(fig, _update, frames=Tmax,
                     interval=max(1, 1000 // max(1, FPS)), blit=False)
out_mp4 = OUT_DIR / "top1_4models.mp4"
writer = FFMpegWriter(fps=FPS, metadata={"title": "Worm decoder comparison"},
                      bitrate=2400)
anim.save(str(out_mp4), writer=writer, dpi=120)
plt.close(fig)
print(f"Saved video: {out_mp4}")

# ── Also: static trace comparison plot ────────────────────────────
fig2, axes2 = plt.subplots(K, 1, figsize=(14, 2 * K), sharex=True)
t_axis = np.arange(Tmax) * dt
for j in range(K):
    ax = axes2[j]
    ax.plot(t_axis, b[:Tmax, j], "k-", lw=1, alpha=0.8, label="GT")
    for m2 in MODELS:
        ax.plot(t_axis, ho[m2][:Tmax, j], "-", color=COLORS[m2], lw=0.8,
                alpha=0.7, label=m2)
    ax.set_ylabel(f"a{j+1}", fontsize=10)
    if j == 0:
        ax.legend(fontsize=7, ncol=5, loc="upper right")
axes2[-1].set_xlabel("Time (s)", fontsize=10)
fig2.suptitle("Eigenworm mode predictions — 2023-01-17-14", fontsize=12)
fig2.tight_layout()
out_png = OUT_DIR / "top1_traces.png"
fig2.savefig(str(out_png), dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved traces: {out_png}")
