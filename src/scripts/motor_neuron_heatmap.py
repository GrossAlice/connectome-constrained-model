#!/usr/bin/env python
"""
Heatmap: rows = worms sorted by Ridge R², columns = motor+control neurons.
Cell = 1 if neuron present in that worm, 0 otherwise.
"""
import json, pathlib, h5py, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── 1. Load Ridge results ────────────────────────────────────────
res_path = ROOT / "output_plots/behaviour_decoder/batch_ridge_mlp/results.json"
res = json.loads(res_path.read_text())

worm_ridge = []
for worm, v in res.items():
    if "Ridge" in v:
        mn = np.mean(v["Ridge"][:6])
        worm_ridge.append((mn, worm))
worm_ridge.sort(reverse=True)

print("Top-15 worms by Ridge R² (mean a1-a6):")
for i, (mn, w) in enumerate(worm_ridge[:15]):
    print(f"  {i+1:3d}  {mn:.4f}  {w}")

second_best = worm_ridge[1][1]
print(f"\nSecond best worm: {second_best}  (Ridge R² = {worm_ridge[1][0]:.4f})")

# ── 2. Load motor+control neuron list ────────────────────────────
mc_path = ROOT / "data/used/masks+motor neurons/motor_neurons_with_control.txt"
mc_neurons_raw = [l.strip() for l in mc_path.read_text().splitlines() if l.strip()]
mc_set = set(mc_neurons_raw)
print(f"\nMotor+control neuron classes: {len(mc_set)}")

# ── 3. For each worm, check which motor+control neurons are present ──
h5_dir = ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"

# Collect all unique neuron names that appear AND are in mc_set
all_present = set()
worm_neurons = {}   # worm_id -> set of neuron names

for _, worm_id in worm_ridge:
    h5_file = h5_dir / f"{worm_id}.h5"
    if not h5_file.exists():
        continue
    with h5py.File(h5_file, "r") as f:
        labels = [l.decode() if isinstance(l, bytes) else l
                  for l in f["gcamp/neuron_labels"][:]]
    present = set()
    for l in labels:
        if l in mc_set:
            present.add(l)
            all_present.add(l)
    worm_neurons[worm_id] = present

# Sort neuron columns: group by class prefix (SMD, RMD, etc), then alphabetical
neuron_cols = sorted(all_present)
print(f"Motor+control neurons observed across all worms: {len(neuron_cols)}")
print(f"  {neuron_cols}")

# ── 4. Build binary matrix ───────────────────────────────────────
n_worms = len(worm_ridge)
n_neurons = len(neuron_cols)
mat = np.zeros((n_worms, n_neurons), dtype=int)

worm_labels = []
ridge_vals = []
for i, (r2, worm_id) in enumerate(worm_ridge):
    worm_labels.append(worm_id)
    ridge_vals.append(r2)
    if worm_id in worm_neurons:
        for j, n in enumerate(neuron_cols):
            if n in worm_neurons[worm_id]:
                mat[i, j] = 1

# ── 5. Plot heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(14, n_neurons * 0.35), max(10, n_worms * 0.35)))

cmap = ListedColormap(["#f0f0f0", "#2166ac"])
im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")

# Y axis: worm names + Ridge R²
y_labels = [f"{w}  ({r:.3f})" for w, r in zip(worm_labels, ridge_vals)]
ax.set_yticks(range(n_worms))
ax.set_yticklabels(y_labels, fontsize=7)

# X axis: neuron names
ax.set_xticks(range(n_neurons))
ax.set_xticklabels(neuron_cols, fontsize=7, rotation=90)

ax.set_xlabel("Motor + Control Neurons", fontsize=10)
ax.set_ylabel("Worms (sorted by Ridge R², best on top)", fontsize=10)
ax.set_title("Motor+Control Neuron Presence Across Worms", fontsize=12)

# Add coverage count at top
counts = mat.sum(axis=0)
for j in range(n_neurons):
    ax.text(j, -0.7, str(counts[j]), ha="center", va="bottom", fontsize=6, color="red")

# Add count per worm on right
for i in range(n_worms):
    ax.text(n_neurons + 0.3, i, str(mat[i].sum()), ha="left", va="center",
            fontsize=6, color="green")

plt.tight_layout()
out = ROOT / "output_plots/behaviour_decoder/motor_control_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")

# Also print the second-best worm's neurons
print(f"\n--- Second best worm: {second_best} ---")
if second_best in worm_neurons:
    print(f"Motor+control neurons ({len(worm_neurons[second_best])}):")
    for n in sorted(worm_neurons[second_best]):
        print(f"  {n}")

# Compare top-2
best_id = worm_ridge[0][1]
print(f"\n--- Comparison: {best_id} vs {second_best} ---")
s1 = worm_neurons.get(best_id, set())
s2 = worm_neurons.get(second_best, set())
print(f"  {best_id}: {len(s1)} motor+ctrl neurons")
print(f"  {second_best}: {len(s2)} motor+ctrl neurons")
print(f"  Shared: {len(s1 & s2)}")
print(f"  Only in {best_id}: {sorted(s1 - s2)}")
print(f"  Only in {second_best}: {sorted(s2 - s1)}")
