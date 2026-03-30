#!/usr/bin/env python
"""
Render a comparison video from AR2+MLP wd-CV predictions:
  Panel 1: Ground Truth (raw body angles + eigenworm reconstruction)
  Panel 2: AR2+MLP (wd-CV)
  Panel 3: AR2+MLP (wd=1e-3 fixed)

Loads predictions from ar2mlp_wd_cv/predictions.npz — no retraining.
"""
import sys, pathlib, numpy as np, h5py
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stage2.posture_videos import angles_to_xy, _load_eigenvectors
from stage1.add_stephens_eigenworms import _preprocess_worm, _get_d_w, _ensure_TN
from scripts.benchmark_ar_decoder_v2 import r2_score

# ── Config ────────────────────────────────────────────────────────
H5 = ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-17-14.h5"
PRED_PATH = ROOT / "output_plots/behaviour_decoder/ar2mlp_wd_cv/predictions.npz"
OUT_DIR = ROOT / "output_plots/behaviour_decoder/ar2mlp_wd_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 6
MAX_FRAMES = 600   # 600 * 0.6s = 6 min
FPS = 10
MODELS = ["AR2+MLP (wd-CV)", "AR2+MLP (wd=1e-3)"]
COLORS = {"AR2+MLP (wd-CV)": "#7570b3", "AR2+MLP (wd=1e-3)": "#d95f02"}

# ── Load predictions ─────────────────────────────────────────────
print("Loading predictions …")
_saved = np.load(str(PRED_PATH))
b = _saved["b_true"]          # (T, K)
ho_cv = _saved["ho_cv"]       # (T, K)
ho_fixed = _saved["ho_fixed"] # (T, K)
best_wds = _saved["best_wds"]
T = b.shape[0]
warmup = 8  # consistent with N_LAGS=8

ho = {"AR2+MLP (wd-CV)": ho_cv, "AR2+MLP (wd=1e-3)": ho_fixed}

print(f"  T={T}, K={K}")
print(f"  Best wd per fold: {best_wds}")

# ── Load dt from H5 ──────────────────────────────────────────────
with h5py.File(str(H5), "r") as f:
    tv = np.asarray(f["timing/timestamp_confocal"])
    dt = float(np.median(np.diff(tv)))
    _ew_d_target = int(f["behaviour/eigenworms_stephens"].attrs["d_target"])
    _ew_d_w = int(f["behaviour/eigenworms_stephens"].attrs["d_w"])
    _ew_source = f["behaviour/eigenworms_stephens"].attrs["source"]
    ew_raw = np.asarray(f["behaviour/eigenworms_stephens"][:, :K], dtype=float)
    ba_src = _ensure_TN(np.asarray(f[_ew_source][:], dtype=float))

print(f"  dt={dt:.3f}s")

# ── R² ────────────────────────────────────────────────────────────
valid = np.arange(warmup, T)
r2_per_model = {}
print("\n── R² per mode ──")
header = f"{'Model':28s}" + "  ".join(f"{'a'+str(j+1):>7s}" for j in range(K)) + "  mean"
print(header)
for m_name, preds in ho.items():
    ok = np.all(np.isfinite(preds[valid]), axis=1)
    idx = valid[ok]
    r2s = [r2_score(b[idx, j], preds[idx, j]) for j in range(K)]
    mn = np.mean(r2s)
    r2_per_model[m_name] = mn
    print(f"{m_name:28s}" + "  ".join(f"{r:7.3f}" for r in r2s) + f"  {mn:.3f}")

# ── Load eigenvectors & compute per-frame mean ──────────────────
eigvecs_all = _load_eigenvectors(h5_path=str(H5), d_target=_ew_d_target)
E = eigvecs_all[:, :K]  # (d_target, K) = (100, 6)
d_recon = E.shape[0]    # 100

proc = _preprocess_worm(ba_src, _ew_d_w, _ew_d_target)  # (T, 100)
per_frame_mean = proc.mean(axis=1, keepdims=True)        # (T, 1)
print(f"  Per-frame mean angle: [{per_frame_mean.min():.3f}, {per_frame_mean.max():.3f}] rad")

# ── Reconstruct body angles from eigenworms + restore mean ───────
Tmax = min(T, MAX_FRAMES)
recon = {}
recon["GT"] = ew_raw[:Tmax, :K] @ E.T + per_frame_mean[:Tmax]
for m_name, preds in ho.items():
    recon[m_name] = preds[:Tmax, :K] @ E.T + per_frame_mean[:Tmax]

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

# Preprocessed body angles XY for GT panel overlay (100-dim)
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
all_coords = np.concatenate(
    [xy[p].reshape(-1, 2) for p in panels] + [xy_raw.reshape(-1, 2)])
m_ok = np.all(np.isfinite(all_coords), axis=1)
xmin, xmax = all_coords[m_ok, 0].min(), all_coords[m_ok, 0].max()
ymin, ymax = all_coords[m_ok, 1].min(), all_coords[m_ok, 1].max()
cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
span = 0.5 * max(xmax - xmin, ymax - ymin) + 2.0

# ── Figure ────────────────────────────────────────────────────────
n_panels = 1 + len(MODELS)
fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5),
                         facecolor="white")
panel_labels = (
    ["Ground Truth"] +
    [f"{m}\n(R²={r2_per_model[m]:.3f})" for m in MODELS]
)
panel_colors = ["black"] + [COLORS[m] for m in MODELS]

lines, heads = [], []
line_raw, head_raw = None, None

for i, (ax, ttl, col) in enumerate(zip(axes, panel_labels, panel_colors)):
    ax.set_title(ttl, fontsize=10, fontweight="bold",
                 color=col if i > 0 else "black")
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
        line_raw, = ax.plot([], [], "-", color="gray", lw=1.5, alpha=0.4)
        head_raw, = ax.plot([], [], "o", color="gray", ms=3, alpha=0.4)

time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=11)

def _update(frame):
    # GT panel: raw body angles overlay
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
out_mp4 = OUT_DIR / "wd_cv_comparison.mp4"
writer = FFMpegWriter(fps=FPS, metadata={"title": "AR2+MLP wd-CV comparison"},
                      bitrate=2400)
anim.save(str(out_mp4), writer=writer, dpi=120)
plt.close(fig)
print(f"Saved video: {out_mp4}")

# ── Static trace comparison ──────────────────────────────────────
fig2, axes2 = plt.subplots(K, 1, figsize=(14, 2 * K), sharex=True)
t_axis = np.arange(Tmax) * dt
for j in range(K):
    ax = axes2[j]
    ax.plot(t_axis, b[:Tmax, j], "k-", lw=1, alpha=0.8, label="GT")
    for m_name in MODELS:
        ax.plot(t_axis, ho[m_name][:Tmax, j], "-", color=COLORS[m_name],
                lw=0.8, alpha=0.7, label=m_name)
    ax.set_ylabel(f"a{j+1}", fontsize=10)
    if j == 0:
        ax.legend(fontsize=7, ncol=3, loc="upper right")
axes2[-1].set_xlabel("Time (s)", fontsize=10)
fig2.suptitle(
    f"AR2+MLP: wd-CV (median wd={np.median(best_wds):.0e}) vs fixed wd=1e-3\n"
    f"Worm 2023-01-17-14",
    fontsize=12)
fig2.tight_layout()
out_png = OUT_DIR / "wd_cv_traces.png"
fig2.savefig(str(out_png), dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved traces: {out_png}")

print("\nDone.")
