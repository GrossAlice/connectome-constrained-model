"""Explore effect of different initialization settings on Stage2 model."""
import torch
import numpy as np
from pathlib import Path
from stage2.io_h5 import load_data_pt
from stage2.config import make_config
from stage2.model import Stage2ModelPT
from stage2.init import InitConfig, init_lambda_u, init_all_from_data, estimate_mode_baseline

H5_PATH = "data/used/behaviour+neuronal activity atanas (2023)/2022-07-20-01.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(cfg, data, lambda_u_init):
    N = data["u_stage1"].shape[1]
    return Stage2ModelPT(
        N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
        cfg, torch.device(DEVICE), d_ell=0,
        lambda_u_init=lambda_u_init,
        sign_t=data.get("sign_t"),
    ).to(DEVICE)

def eval_init_loss(model, u):
    """Compute one-step prediction loss before any training."""
    u = u.to(DEVICE)
    T, N = u.shape
    s_sv = torch.zeros(N, model.r_sv, device=DEVICE)
    s_dcv = torch.zeros(N, model.r_dcv, device=DEVICE)
    gating = torch.ones(N, device=DEVICE)
    
    pred_loss = 0.0
    for t in range(T - 1):
        u_pred, s_sv, s_dcv = model.prior_step(u[t], s_sv, s_dcv, gating)
        pred_loss += (u_pred - u[t + 1]).pow(2).mean().item()
    return pred_loss / (T - 1)

def run_experiment(name, init_cfg, network_frac):
    cfg = make_config(H5_PATH, device=DEVICE)
    data = load_data_pt(cfg)
    u = data["u_stage1"]
    N = u.shape[1]
    
    lambda_u_init = init_lambda_u(data.get("rho_stage1"), N, init_cfg)
    model = build_model(cfg, data, lambda_u_init)
    init_all_from_data(model, u, init_cfg, network_frac)
    
    loss = eval_init_loss(model, u)
    print(f"\n=== {name} ===")
    print(f"  Initial MSE: {loss:.6f}")
    print(f"  G: {model.G.item():.6f}")
    print(f"  a_sv mean: {model.a_sv.mean().item():.6f}")
    print(f"  I0 mean: {model.I0.mean().item():.4f}")
    if model.E_sv.numel() > 1:
        print(f"  E_sv: exc={model.E_sv[model.E_sv > 0].mean().item():.4f}, inh={model.E_sv[model.E_sv < 0].mean().item():.4f}")
    return loss

if __name__ == "__main__":
    results = {}
    
    # Baseline
    results["baseline"] = run_experiment(
        "Baseline (defaults)",
        InitConfig(),
        network_frac=0.3
    )
    
    # No network scaling
    results["no_scale"] = run_experiment(
        "No network scaling",
        InitConfig(),
        network_frac=0.0
    )
    
    # Larger network contribution
    results["large_net"] = run_experiment(
        "Large network (frac=0.5)",
        InitConfig(),
        network_frac=0.5
    )
    
    # Smaller network contribution
    results["small_net"] = run_experiment(
        "Small network (frac=0.1)",
        InitConfig(),
        network_frac=0.1
    )
    
    # No data-driven reversals
    results["no_rev"] = run_experiment(
        "No auto reversals",
        InitConfig(auto_reversals=False),
        network_frac=0.3
    )
    
    # Different reversal quantiles (narrower)
    results["narrow_rev"] = run_experiment(
        "Narrow reversals (10-90%)",
        InitConfig(reversal_q_lo=0.10, reversal_q_hi=0.90),
        network_frac=0.3
    )
    
    # Different reversal quantiles (wider)
    results["wide_rev"] = run_experiment(
        "Wide reversals (1-99%)",
        InitConfig(reversal_q_lo=0.01, reversal_q_hi=0.99),
        network_frac=0.3
    )
    
    # Tighter lambda_u bounds
    results["tight_lam"] = run_experiment(
        "Tight lambda_u (0.1-0.9)",
        InitConfig(lambda_u_min=0.1, lambda_u_max=0.9),
        network_frac=0.3
    )
    
    print("\n" + "="*60)
    print("SUMMARY (Initial MSE, lower = better initial fit)")
    print("="*60)
    for name, loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:20s}: {loss:.6f}")
