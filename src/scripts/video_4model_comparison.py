#!/usr/bin/env python
"""
5-panel comparison video:  GT | Ridge | MLP | MLP→Ridge | AR2+MLP
Loads cached predictions from variance_comparison/predictions.npz — no retraining.
"""
import sys, pathlib, numpy as np, h5py
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stage2.posture_videos import angles_to_xy, _load_eigenvectors
from stage1.add_stephens_eigenworms import _preprocess_worm, _ensure_TN
from scripts.benchmark_ar_decoder_v2 import r2_score

# ── Config ────────────────────────────────────────────────────────
H5 = ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-17-14.h5"
PRED_TOP1 = ROOT / "output_plots/behaviour_decoder/top1_video/predictions.npz"
PRED_VAR  = ROOT / "output_plots/behaviour_decoder/variance_comparison/predictions.npz"
PRED_WD   = ROOT / "output_plots/behaviour_decoder/ar2mlp_wd_cv/predictions.npz"
OUT_DIR = ROOT / "output_plots/behaviour_decoder/video_4model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 6
MAX_FRAMES = 600
FPS = 10

MODELS = ["Ridge", "MLP", "MLP→Ridge", "AR2+MLP"]
COLORS = {
    "Ridge":      "#1b9e77",
    "MLP":        "#d95f02",
    "MLP→Ridge":  "#e7298a",
    "AR2+MLP":    "#7570b3",
}

# ── Load predictions (merge two sources) ─────────────────────────
print("Loading predictions …")
_top1 = np.load(str(PRED_TOP1), allow_pickle=True)
_var  = np.load(str(PRED_VAR), allow_pickle=True)
_wd   = np.load(str(PRED_WD), allow_pickle=True)
b = _top1["b_true"][:, :K]
ho = {
    "Ridge":      _top1["Ridge"][:, :K],
    "MLP":        _top1["MLP"][:, :K],
    "MLP→Ridge":  _var["MLP→Ridge_raw"][:, :K],
    "AR2+MLP":    _wd["ho_fixed"][:, :K],       # wd=1e-3 fixed (R²≈0.543)
}
T = b.shape[0]
warmup = 8
print(f"  T={T}, K={K}")

# ── Load dt and eigenworm info from H5 ───────────────────────────
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
r2_mean = {}
print("\n── R² per mode ──")
hdr = f"{'Model':16s}" + "  ".join(f"{'a'+str(j+1):>7s}" for j in range(K)) + "    mean"
print(hdr)
for name in MODELS:
    preds = ho[name]
    ok = np.all(np.isfinite(preds[valid]), axis=1)
    idx = valid[ok]
    r2s = [r2_score(b[idx, j], preds[idx, j]) for j in range(K)]
    mn = float(np.mean(r2s))
    r2_mean[name] = mn
    print(f"{name:16s}" + "  ".join(f"{r:7.3f}" for r in r2s) + f"    {mn:.3f}")

# ── Eigenvectors & per-frame mean ────────────────────────────────
eigvecs_all = _load_eigenvectors(h5_path=str(H5), d_target=_ew_d_target)
E = eigvecs_all[:, :K]
d_recon = E.shape[0]

# ── Reconstruct body angles (mean-subtracted) ───────────────────
Tmax = min(T, MAX_FRAMES)
recon = {}
recon["GT"] = ew_raw[:Tmax, :K] @ E.T
for name in MODELS:
    recon[name] = ho[name][:Tmax, :K] @ E.T

# Mean-subtracted preprocessed body angles for GT overlay
proc = _preprocess_worm(ba_src, _ew_d_w, _ew_d_target)
per_frame_mean = proc.mean(axis=1, keepdims=True)
proc_gt = (proc - per_frame_mean)[:Tmax]

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
fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5),
                         facecolor="white")
panel_labels = (
    ["Ground Truth"] +
    [f"{n}\n(R²={r2_mean[n]:.3f})" for n in MODELS]
)
panel_colors = ["black"] + [COLORS[n] for n in MODELS]

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
out_mp4 = OUT_DIR / "4model_comparison.mp4"
writer = FFMpegWriter(fps=FPS, metadata={"title": "Worm decoder 4-model comparison"},
                      bitrate=2400)
anim.save(str(out_mp4), writer=writer, dpi=120)
plt.close(fig)
print(f"Saved video: {out_mp4}")

# ── Static trace comparison ──────────────────────────────────────
fig2, axes2 = plt.subplots(K, 1, figsize=(16, 2 * K), sharex=True)
t_axis = np.arange(Tmax) * dt
for j in range(K):
    ax = axes2[j]
    ax.plot(t_axis, b[:Tmax, j], "k-", lw=1, alpha=0.8, label="GT")
    for n in MODELS:
        ax.plot(t_axis, ho[n][:Tmax, j], "-", color=COLORS[n],
                lw=0.8, alpha=0.7, label=n)
    ax.set_ylabel(f"a{j+1}", fontsize=10)
    if j == 0:
        ax.legend(fontsize=7, ncol=5, loc="upper right")
axes2[-1].set_xlabel("Time (s)", fontsize=10)
fig2.suptitle("Eigenworm predictions — Ridge · MLP · MLP→Ridge · AR2+MLP\n"
              "Worm 2023-01-17-14", fontsize=12)
fig2.tight_layout()
out_png = OUT_DIR / "4model_traces.png"
fig2.savefig(str(out_png), dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved traces: {out_png}")

print("\nDone.")
