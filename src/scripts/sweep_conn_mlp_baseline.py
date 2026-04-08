#!/usr/bin/env python
"""
Connectome-constrained MLP baseline sweep
==========================================
Same hyperparameter grid as sweep_mlp_baseline.py (hidden, dropout, wd),
but each neuron i gets its OWN MLP that only sees connectome neighbors + self.

For each neuron i:
  - Features = K-step context of (neighbors ∪ {i}), so d_in = K * |nbr(i)+1|
  - Separate MLP trained per neuron with early stopping

This tests the hypothesis that connectome structure + per-neuron MLP can
match or beat the joint unconstrained MLP from sweep_mlp_baseline.py.

Grid: 5 hidden configs × 3 dropouts × 3 weight_decays = 45 configs.
"""
import sys, os, time, json, itertools
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/conn_mlp_baseline_sweep"
os.makedirs(SAVE_ROOT, exist_ok=True)

K = 5                  # context window (same as vanilla MLP sweep)
N_FOLDS = 3
WINDOW_SIZE = 50       # LOO window
MAX_EPOCHS = 200
PATIENCE = 20
LOO_N = 30             # number of LOO neurons
LOO_SEED = 0
LOO_MODE = "variance"

ROOT = "/home/agross/Downloads/connectome-constrained model/src"

# ──────────────────────────────────────────────────────────────────────
#  Sweep grid (mirrors sweep_mlp_baseline.py)
# ──────────────────────────────────────────────────────────────────────
# For per-neuron models, the hidden sizes are for a SINGLE neuron's MLP.
# Since input dim is much smaller (~K * 20 ≈ 100 vs K * 123 = 615),
# we use the same hidden specs but they effectively give more capacity.
HIDDEN_SIZES = [
    (64,),
    (128,),
    (256,),
    (64, 64),
    (128, 64),
]
DROPOUTS = [0.0, 0.2, 0.5]
WEIGHT_DECAYS = [1e-5, 1e-3, 1e-2]

CONFIGS = []
for hs, dp, wd in itertools.product(HIDDEN_SIZES, DROPOUTS, WEIGHT_DECAYS):
    tag = f"h{'_'.join(map(str, hs))}_dp{dp}_wd{wd}"
    CONFIGS.append(dict(tag=tag, hidden=hs, dropout=dp, weight_decay=wd))

print(f"Conn-MLP sweep: {len(CONFIGS)} configurations, {N_FOLDS}-fold CV, K={K}")

# ──────────────────────────────────────────────────────────────────────
#  Load data
# ──────────────────────────────────────────────────────────────────────
import h5py
from pathlib import Path

h5_path = os.path.join(ROOT, H5)
with h5py.File(h5_path, "r") as f:
    u = f["stage1/u_mean"][:]          # (T, N)
    labels = [s.decode() if isinstance(s, bytes) else s
              for s in f["gcamp/neuron_labels"][:]]
T, N = u.shape
print(f"Data: T={T}, N={N}")

# Choose LOO subset: top-30 highest-variance neurons
var_order = np.argsort(np.nanvar(u, axis=0))[::-1]
loo_neurons = sorted(int(i) for i in var_order[:LOO_N])
print(f"LOO neurons ({len(loo_neurons)}): {loo_neurons[:10]}...")

# ──────────────────────────────────────────────────────────────────────
#  Load connectome  →  partners per neuron
# ──────────────────────────────────────────────────────────────────────
d = Path(ROOT) / "data/used/masks+motor neurons"
atlas_names = np.load(d / "neuron_names.npy")
n2a = {str(n): i for i, n in enumerate(atlas_names)}

T_all = sum(
    np.abs(np.load(d / f"{t}.npy")) for t in ("T_e", "T_sv", "T_dcv")
)

wa = [n2a.get(lab, -1) for lab in labels]
adj = np.zeros((N, N), np.float64)
for i in range(N):
    for j in range(N):
        if wa[i] >= 0 and wa[j] >= 0:
            adj[j, i] = T_all[wa[j], wa[i]]

partners = {}
for i in range(N):
    partners[i] = sorted(j for j in range(N) if j != i and adj[j, i] > 0)

np_arr = [len(partners[i]) for i in range(N)]
n_zero = sum(1 for v in np_arr if v == 0)
print(f"Connectome: partners per neuron — "
      f"min={min(np_arr)}, median={int(np.median(np_arr))}, "
      f"max={max(np_arr)}, isolated={n_zero}/{N}")

# ──────────────────────────────────────────────────────────────────────
#  Temporal folds (same as vanilla MLP sweep)
# ──────────────────────────────────────────────────────────────────────
from stage2.train import _make_temporal_folds

raw_folds = _make_temporal_folds(T, N_FOLDS)
folds = []
for te_s, te_e in raw_folds:
    if te_s == raw_folds[0][0]:
        folds.append((te_e, T, te_s, te_e))
    else:
        folds.append((0, te_s, te_s, te_e))


# ──────────────────────────────────────────────────────────────────────
#  Per-neuron MLP
# ──────────────────────────────────────────────────────────────────────
class PerNeuronMLP(nn.Module):
    def __init__(self, d_in, hidden=(64,), dropout=0.0):
        super().__init__()
        layers = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-12)


# ──────────────────────────────────────────────────────────────────────
#  Train + evaluate one config
# ──────────────────────────────────────────────────────────────────────
def run_one_config(cfg, u, folds, loo_neurons, partners, device="cuda"):
    x = u.astype(np.float32)
    T, N = x.shape
    dev = torch.device(device)

    os_pred = np.full((T, N), np.nan, np.float32)
    loo_pred = np.full((T, N), np.nan, np.float32)

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        u_tr = x[tr_s:tr_e]
        T_tr = len(u_tr)

        # Train per-neuron models
        models = {}
        for i in range(N):
            feats = sorted(partners[i] + [i])   # connectome neighbors + self
            d_in = K * len(feats)

            # Build sliding-window features for this neuron
            indices = np.arange(K, T_tr)
            X_np = np.stack([u_tr[t - K:t][:, feats].flatten()
                             for t in indices])  # (T_tr-K, K*|feats|)
            Y_np = u_tr[indices, i]              # (T_tr-K,)

            # Train/val split (last 15%)
            nv = max(int(X_np.shape[0] * 0.15), 1)
            nf = X_np.shape[0] - nv
            Xf = torch.tensor(X_np[:nf], dtype=torch.float32, device=dev)
            Xv = torch.tensor(X_np[nf:], dtype=torch.float32, device=dev)
            Yf = torch.tensor(Y_np[:nf], dtype=torch.float32, device=dev)
            Yv = torch.tensor(Y_np[nf:], dtype=torch.float32, device=dev)

            model = PerNeuronMLP(d_in, hidden=cfg["hidden"],
                                 dropout=cfg["dropout"]).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                   weight_decay=cfg["weight_decay"])

            bvl, bst, w = float("inf"), None, 0
            for ep in range(MAX_EPOCHS):
                model.train()
                perm = torch.randperm(nf, device=dev)
                for bs in range(0, nf, 256):
                    idx = perm[bs:bs + 256]
                    loss = nn.functional.mse_loss(model(Xf[idx]), Yf[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(model(Xv), Yv).item()
                if vl < bvl - 1e-6:
                    bvl = vl
                    bst = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
                    w = 0
                else:
                    w += 1
                    if w >= PATIENCE:
                        break
            if bst:
                model.load_state_dict(
                    {k: v.to(dev) for k, v in bst.items()})
            model.eval()
            models[i] = (model, feats)

            del opt, Xf, Xv, Yf, Yv

        # ── One-step predictions on test fold ──
        first = max(K, te_s)
        n_te = te_e - first
        if n_te > 0:
            for i in range(N):
                mdl, feats = models[i]
                X_te = np.stack([x[t - K:t][:, feats].flatten()
                                 for t in range(first, te_e)])
                with torch.no_grad():
                    for bs in range(0, n_te, 512):
                        be = min(bs + 512, n_te)
                        ct = torch.tensor(X_te[bs:be],
                                          dtype=torch.float32, device=dev)
                        os_pred[first + bs:first + be, i] = (
                            mdl(ct).cpu().numpy())

        # ── Windowed LOO ──
        T_te = te_e - te_s
        for ni in loo_neurons:
            mdl, feats = models[ni]
            ni_local = feats.index(ni)  # position of ni in feats

            pred_arr = np.full(T_te, np.nan, np.float32)
            for s in range(0, T_te, WINDOW_SIZE):
                e = min(s + WINDOW_SIZE, T_te)
                pred_arr[s] = x[te_s + s, ni]
                for t_loc in range(s, e - 1):
                    t_abs = te_s + t_loc
                    ctx_start = t_abs - K + 1
                    if ctx_start < 0:
                        continue
                    # Build K-step context for neuron ni's features
                    ctx = x[ctx_start:t_abs + 1][:, feats].copy()  # (K, |feats|)
                    # Replace neuron ni's values with predictions where available
                    for k in range(K):
                        pa_idx = t_loc - (K - 1) + k
                        if (pa_idx >= s and pa_idx <= t_loc
                                and np.isfinite(pred_arr[pa_idx])):
                            ctx[k, ni_local] = pred_arr[pa_idx]
                    ctx_flat = ctx.flatten()
                    with torch.no_grad():
                        ct = torch.tensor(ctx_flat, dtype=torch.float32,
                                          device=dev).unsqueeze(0)
                        pred_arr[t_loc + 1] = mdl(ct).cpu().numpy().item()
            loo_pred[te_s:te_e, ni] = pred_arr

        # Cleanup fold models
        for i in list(models.keys()):
            del models[i]
        del models
        torch.cuda.empty_cache()

    # ── Compute R² ──
    onestep_r2 = np.full(N, np.nan)
    for i in range(N):
        m = np.isfinite(os_pred[:, i])
        if m.sum() > 3:
            onestep_r2[i] = _r2(u[m, i], os_pred[m, i])

    loo_r2 = np.full(len(loo_neurons), np.nan)
    for ki, ni in enumerate(loo_neurons):
        m = np.isfinite(loo_pred[:, ni])
        if m.sum() > 3:
            loo_r2[ki] = _r2(u[m, ni], loo_pred[m, ni])

    # Count total parameters (sum over all N per-neuron models)
    total_params = 0
    for i in range(N):
        feats = sorted(partners[i] + [i])
        d_in = K * len(feats)
        tmp = PerNeuronMLP(d_in, hidden=cfg["hidden"], dropout=cfg["dropout"])
        total_params += sum(p.numel() for p in tmp.parameters())

    return {
        "onestep_r2_mean": float(np.nanmean(onestep_r2)),
        "onestep_r2_median": float(np.nanmedian(onestep_r2)),
        "loo_r2_mean": float(np.nanmean(loo_r2)),
        "loo_r2_median": float(np.nanmedian(loo_r2)),
        "n_params_total": total_params,
    }


# ──────────────────────────────────────────────────────────────────────
#  Run sweep
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    results = []
    for ci, cfg in enumerate(CONFIGS):
        tag = cfg["tag"]
        out_path = os.path.join(SAVE_ROOT, f"{tag}.json")

        if os.path.exists(out_path):
            print(f"[{ci+1}/{len(CONFIGS)}] SKIP {tag} (exists)")
            with open(out_path) as f:
                results.append(json.load(f))
            continue

        print(f"\n[{ci+1}/{len(CONFIGS)}] {tag}")
        t0 = time.time()
        r = run_one_config(cfg, u, folds, loo_neurons, partners,
                           device=args.device)
        elapsed = time.time() - t0
        r["tag"] = tag
        r["hidden"] = list(cfg["hidden"])
        r["dropout"] = cfg["dropout"]
        r["weight_decay"] = cfg["weight_decay"]
        r["elapsed_sec"] = round(elapsed, 1)
        results.append(r)

        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  OS={r['onestep_r2_mean']:.4f}  "
              f"LOO={r['loo_r2_mean']:.4f}  "
              f"params={r['n_params_total']:,}  "
              f"time={elapsed:.1f}s")

    # Final summary
    results.sort(key=lambda x: x.get("loo_r2_mean", -999), reverse=True)
    with open(os.path.join(SAVE_ROOT, "sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Conn-MLP sweep complete ({len(results)} configs)")
    print(f"{'Tag':40s} {'OS_mean':>8s} {'LOO_mean':>8s} {'Params':>10s}")
    for r in results[:10]:
        print(f"{r['tag']:40s} {r['onestep_r2_mean']:8.4f} "
              f"{r['loo_r2_mean']:8.4f} {r['n_params_total']:10,}")
