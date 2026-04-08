from __future__ import annotations
import math
import numpy as np
import torch
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .model import Stage2ModelPT



def _build_per_source_reversals(T_sv, sign_t, e_exc, e_inh):
    """Per-neuron reversal: weighted average of E_exc / E_inh by input sign."""
    mask = T_sv > 0
    total = mask.sum(1).float().clamp(min=1)
    E = (((sign_t > 0) & mask).sum(1).float() / total * e_exc
         + ((sign_t < 0) & mask).sum(1).float() / total * e_inh)
    return torch.where(total <= 1, torch.zeros_like(E), E)


def _build_edge_specific_reversals(T_sv, sign_t, e_exc, e_inh):
    """Per-edge reversal: E_exc for excitatory, E_inh for inhibitory edges."""
    mask = T_sv > 0
    E = torch.where(
        sign_t > 0,
        torch.full_like(sign_t, float(e_exc)),
        torch.where(sign_t < 0, torch.full_like(sign_t, float(e_inh)),
                     torch.zeros_like(sign_t)),
    )
    return E * mask.float()


def _ols_lambda_u(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    x, y = u[:-1], u[1:]
    mx, my = x.mean(0), y.mean(0)
    var_x = ((x - mx) ** 2).mean(0)
    cov_xy = ((x - mx) * (y - my)).mean(0)
    a = torch.where(var_x > 1e-30, cov_xy / var_x, torch.ones_like(var_x))
    return (1.0 - a).clamp(lo, hi)


def init_lambda_u(u: torch.Tensor, cfg=None) -> torch.Tensor:
    # Slightly tighter than the physical bounds to avoid degenerate
    # starting points (λ=0 ⇒ no leak, λ≈1 ⇒ memoryless).
    lo = float(getattr(cfg, "lambda_u_lo", 0.0)) if cfg else 0.0
    hi = float(getattr(cfg, "lambda_u_hi", 0.9999)) if cfg else 0.9999
    init_lo = max(lo + 1e-3, 1e-3)
    init_hi = min(hi - 5e-4, 1.0 - 5e-4)
    lam = _ols_lambda_u(u, init_lo, init_hi)
    print(f"[init] lambda_u (OLS): mean={lam.mean():.4f}, "
          f"min={lam.min():.4f}, max={lam.max():.4f}")
    return lam

def init_I0(model: "Stage2ModelPT", u: torch.Tensor) -> torch.Tensor:
    lam = model.lambda_u.detach()
    x, y = u[:-1].to(model.I0.device), u[1:].to(model.I0.device)
    b = y.mean(0) - (1 - lam) * x.mean(0)
    I0_ols = b / lam.clamp(min=0.01)
    with torch.no_grad():
        model.I0.data.copy_(I0_ols)
    print(f"[init] I0 (OLS): mean={I0_ols.mean():.4f}, "
          f"min={I0_ols.min():.4f}, max={I0_ols.max():.4f}")
    return I0_ols

def init_reversals(model: "Stage2ModelPT", u: torch.Tensor, baseline: torch.Tensor, cfg=None):
    """Set reversal potentials from data quantiles and neurotransmitter signs.

    * E_exc ← 99th percentile of neural traces
    * E_inh ← 1st percentile
    * E_dcv ← median tonic drive (rest)

    The sign structure is built from ``model.sign_t`` (stored by the
    constructor) and combined with data-driven magnitudes in one shot.
    """
    q_lo, q_hi = 0.01, 0.99
    finite = u.detach().cpu().numpy().ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size < 10:
        return
    rest = float(torch.nanmedian(baseline.to(u.device)))
    e_inh, e_exc = float(np.quantile(finite, q_lo)), float(np.quantile(finite, q_hi))
    if not np.isfinite(rest):
        rest = float(np.median(finite))
    pad = max(1e-3, 0.1 * max(abs(rest), 1.0))
    if not np.isfinite(e_inh) or e_inh >= rest:
        e_inh = rest - pad
    if not np.isfinite(e_exc) or e_exc <= rest:
        e_exc = rest + pad

    mode = str(getattr(cfg, "reversal_mode", "per_neuron") if cfg else "per_neuron")
    sign_t = getattr(model, "sign_t", None)

    with torch.no_grad():
        if mode == "per_edge" and sign_t is not None:
            E_sv = _build_edge_specific_reversals(
                model.T_sv, sign_t, e_exc, e_inh)
        elif mode == "per_neuron" and sign_t is not None:
            E_sv = _build_per_source_reversals(
                model.T_sv, sign_t, e_exc, e_inh)
        else:
            E_sv = torch.full_like(model.E_sv, e_exc)
        model.E_sv.data.copy_(E_sv)
        model.E_dcv.data.fill_(rest)
    print(f"[init] reversals: E_exc={e_exc:.4f}, E_inh={e_inh:.4f}, E_dcv={rest:.4f}")

# ── Connectivity-based weight initialisation ────────────────────────────

def _pairwise_abs_corr(u: torch.Tensor) -> torch.Tensor:
    """Compute (N, N) pairwise |Pearson correlation| from neural traces.

    Parameters
    ----------
    u : (T, N) tensor of neural activity

    Returns
    -------
    C : (N, N) symmetric matrix of absolute correlations
    """
    u_np = u.detach().cpu().float()
    u_centered = u_np - u_np.mean(0, keepdim=True)
    norms = u_centered.pow(2).sum(0).sqrt().clamp(min=1e-10)  # (N,)
    C = (u_centered.T @ u_centered) / (norms.unsqueeze(1) * norms.unsqueeze(0))
    return C.abs()


def init_W_from_config(model: "Stage2ModelPT", cfg=None,
                       u: Optional[torch.Tensor] = None) -> None:
    """Apply correlation-weighted W initialisation for chemical synapses.

    Modes (from ``cfg.W_init_mode``):

    * ``uniform`` (default) — keep the constructor's constant W.
    * ``corr_weighted`` — scale each edge's W by the empirical |corr(i,j)|,
      so that functionally-coupled edges start with higher synaptic weight.
    """
    mode = str(getattr(cfg, "W_init_mode", "uniform")) if cfg else "uniform"
    if mode == "uniform":
        return
    if mode != "corr_weighted" or u is None:
        return

    from .model import _reparam_inv
    C = _pairwise_abs_corr(u).to(model.T_sv.device)

    for attr in ("W_sv", "W_dcv"):
        T_mat = getattr(model, f"T_{attr.split('_')[1]}")
        mask = (T_mat > 0).float()
        n_edges = mask.sum().item()
        if n_edges == 0:
            continue
        # Scale current W values by relative |corr| (normalised to mean=1)
        corr_edge = C * mask
        mean_corr = corr_edge.sum() / max(n_edges, 1)
        scale = torch.where(mask.bool(),
                            corr_edge / mean_corr.clamp(min=1e-6),
                            torch.ones_like(mask))
        cur_W = getattr(model, attr)
        new_W = (cur_W * scale).clamp(min=1e-6)
        lo = getattr(model, f"_{attr}_lo")
        hi = getattr(model, f"_{attr}_hi")
        new_raw = _reparam_inv(new_W, lo, hi) * mask
        with torch.no_grad():
            getattr(model, f"_{attr}_raw").data.copy_(new_raw)
        print(f"[init] {attr} (corr_weighted): mean_corr={mean_corr:.4f}, "
              f"scale range=[{scale[mask.bool()].min():.2f}, {scale[mask.bool()].max():.2f}]")


def init_G_from_config(model: "Stage2ModelPT", cfg=None,
                       u: Optional[torch.Tensor] = None) -> None:
    """Apply connectivity-structure-based G initialisation for edge-specific G.

    Modes (from ``cfg.G_init_mode``):

    * ``uniform`` (default) — keep the constructor's constant G.
    * ``log_counts`` — ``T_e·G = log₂(1+c)``.
    * ``sqrt_counts`` — ``T_e·G = √c``.
    * ``corr_weighted`` — scale each edge's G by empirical |corr(i,j)|.
    """
    if not model.edge_specific_G:
        return  # scalar G has no mode switching
    from .model import _reparam_inv, _G_INIT

    mode = str(getattr(cfg, "G_init_mode", "uniform")) if cfg else "uniform"
    if mode == "uniform":
        return

    T_e = model.T_e
    mask = (T_e > 0).float()
    safe_te = T_e.clamp(min=1.0)
    if mode == "log_counts":
        G_mat = torch.where(mask.bool(),
                            torch.log2(1.0 + T_e) / safe_te,
                            torch.full_like(T_e, _G_INIT))
    elif mode == "sqrt_counts":
        G_mat = torch.where(mask.bool(),
                            1.0 / safe_te.sqrt(),
                            torch.full_like(T_e, _G_INIT))
    elif mode == "corr_weighted" and u is not None:
        C = _pairwise_abs_corr(u).to(T_e.device)
        corr_edge = C * mask
        n_edges = mask.sum().item()
        mean_corr = corr_edge.sum() / max(n_edges, 1)
        # Scale default G by relative |corr| (normalised so mean=_G_INIT)
        G_mat = torch.where(mask.bool(),
                            _G_INIT * corr_edge / mean_corr.clamp(min=1e-6),
                            torch.full_like(T_e, _G_INIT))
        print(f"[init] G (corr_weighted): mean_corr={mean_corr:.4f}, "
              f"G range=[{G_mat[mask.bool()].min():.4f}, {G_mat[mask.bool()].max():.4f}]")
    else:
        return
    G_mat = G_mat * mask
    G_raw = _reparam_inv(G_mat, model._G_lo, model._G_hi)
    with torch.no_grad():
        model._G_raw.data.copy_(G_raw)
    if mode != "corr_weighted":  # corr_weighted already printed
        print(f"[init] G ({mode}): mean(edges)={G_mat[mask.bool()].mean():.4f}")


def init_corr_reg_mask(model: "Stage2ModelPT", u: torch.Tensor) -> None:
    """Compute and register a correlation-based regularisation mask.

    Stores ``_corr_reg_mask`` as a buffer on the model: an (N, N) matrix
    with values ``(1 - |corr(i,j)|)`` for connected edges (any type) and
    0 for non-edges.  Used by train.py when ``corr_reg_weight > 0`` to
    penalise low-correlation edges more heavily.
    """
    C = _pairwise_abs_corr(u).to(model.T_sv.device)
    # Any-type connectivity mask
    any_conn = ((model.T_sv > 0) | (model.T_dcv > 0) | (model.T_e > 0)).float()
    reg_mask = (1.0 - C) * any_conn
    model.register_buffer("_corr_reg_mask", reg_mask, persistent=False)
    n_edges = any_conn.sum().item()
    mean_penalty = reg_mask.sum() / max(n_edges, 1)
    print(f"[init] corr_reg_mask: {int(n_edges)} edges, "
          f"mean penalty weight={mean_penalty:.3f}")


def _collect_current_patterns(
    model: "Stage2ModelPT", u: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Forward-simulate gap / sv / dcv currents at current parameter values.

    Returns
    -------
    resid : (T-1, N)  AR(1) residual
    patterns : list of (T-1, N) tensors, one per channel
    names : list of channel names ("G", "a_sv", "a_dcv")
    """
    dev, lam = model.I0.device, model.lambda_u
    u = u.to(dev)
    T, N = u.shape

    resid = u[1:] - ((1 - lam) * u[:-1] + lam * model.I0)

    L = model.laplacian()
    g = torch.ones(N, device=dev)
    names = ["G"]
    c_gap = torch.zeros(T - 1, N, device=dev)
    s_sv  = torch.zeros(N, model.r_sv,  device=dev)
    s_dcv = torch.zeros(N, model.r_dcv, device=dev)
    c_sv_all  = torch.zeros(T - 1, N, device=dev) if model.r_sv  > 0 else None
    c_dcv_all = torch.zeros(T - 1, N, device=dev) if model.r_dcv > 0 else None

    for t in range(T - 1):
        c_gap[t] = lam * (L @ u[t])
        phi_g = model.phi(u[t]) * g
        if model.r_sv > 0:
            I, s_sv = model._synaptic_current(
                u[t], phi_g, s_sv,
                model.T_sv * model._get_W("W_sv"),
                model.a_sv, model.tau_sv, model.E_sv)
            c_sv_all[t] = lam * I
        if model.r_dcv > 0:
            I, s_dcv = model._synaptic_current(
                u[t], phi_g, s_dcv,
                model.T_dcv * model._get_W("W_dcv"),
                model.a_dcv, model.tau_dcv, model.E_dcv)
            c_dcv_all[t] = lam * I

    patterns = [c_gap]
    if model.r_sv  > 0: names.append("a_sv");  patterns.append(c_sv_all)
    if model.r_dcv > 0: names.append("a_dcv"); patterns.append(c_dcv_all)
    return resid, patterns, names


def _apply_scales(
    model: "Stage2ModelPT",
    names: list[str],
    betas: list[float],
    rms_list: list[float],
    rms_resid: float,
    label: str,
) -> None:
    """Scale model parameters by *betas* and log diagnostics.

    For synaptic channels (a_sv, a_dcv) the scale is absorbed into W_sv / W_dcv
    rather than into the per-rank amplitudes, so the user-specified rank
    structure in a_sv_init / a_dcv_init is preserved and doesn't saturate the
    sigmoid reparameterisation.
    """
    total_rms = math.sqrt(sum(r ** 2 for r in rms_list))
    print(f"[init] network_scale ({label}): AR1_rms={rms_resid:.4f}"
          f"  net_rms={total_rms:.4f} ({total_rms / rms_resid:.0%})")
    # Map synaptic amplitude name → weight matrix name
    _AMP_TO_W = {"a_sv": "W_sv", "a_dcv": "W_dcv"}
    for i, name in enumerate(names):
        b = betas[i]
        rms_after = rms_list[i]
        frac = rms_after / rms_resid
        if name == "G" and model.edge_specific_G:
            cur = getattr(model, name)
            if cur.numel() == 0:
                continue
            from .model import _reparam_inv
            new_raw = _reparam_inv(cur * b, model._G_lo, model._G_hi)
            if hasattr(model, "_G_mask"):
                new_raw = new_raw * model._G_mask
            model._G_raw.data.copy_(new_raw)
            print(f"[init]   G: \u03b2={b:.4f}, RMS={rms_after:.4f} ({frac:.0%} of AR1)")
        elif name in _AMP_TO_W:
            # Absorb the scale into W rather than collapsing a_sv/a_dcv
            w_name = _AMP_TO_W[name]
            cur_w = getattr(model, w_name)
            if cur_w.numel() == 0:
                continue
            model.set_param_constrained(w_name, cur_w * b)
            print(f"[init]   {name}: \u03b2={b:.4f} \u2192 applied to {w_name}, "
                  f"RMS={rms_after:.4f} ({frac:.0%} of AR1)")
        else:
            cur = getattr(model, name)
            if cur.numel() == 0:
                continue
            model.set_param_constrained(name, cur * b)
            print(f"[init]   {name}: \u03b2={b:.4f}, RMS={rms_after:.4f} ({frac:.0%} of AR1)")


# ── Global-OLS init (original) ──────────────────────────────────────────

def _init_network_scale_ols(
    model: "Stage2ModelPT", u: torch.Tensor,
    min_network_frac: float = 0.25,
) -> None:
    with torch.no_grad():
        resid, patterns, names = _collect_current_patterns(model, u)
        rms_resid = resid.pow(2).mean().sqrt().item()
        if rms_resid < 1e-12:
            return

        dev = model.I0.device
        K = len(patterns)
        r_flat = resid.reshape(-1)
        C = torch.stack([p.reshape(-1) for p in patterns], dim=1)
        CTC = C.T @ C + 1e-8 * torch.eye(K, device=dev)
        CTr = C.T @ r_flat
        beta = torch.linalg.solve(CTC, CTr).clamp(min=0)

        betas = [max(beta[i].item(), 1e-6) for i in range(K)]
        rms_list = [patterns[i].pow(2).mean().sqrt().item() * betas[i]
                    for i in range(K)]
        total_rms = math.sqrt(sum(r ** 2 for r in rms_list))

        target_rms = min_network_frac * rms_resid
        if total_rms < target_rms and total_rms > 1e-12:
            base_boost = target_rms / total_rms
        else:
            base_boost = 1.0

        scales = [base_boost for _ in range(K)]
        channel_floor_frac = {
            "G": 0.10,
            "a_sv": 0.20,
            "a_dcv": 0.10,
        }
        for i, name in enumerate(names):
            floor_frac = channel_floor_frac.get(name, 0.0)
            if floor_frac <= 0:
                continue
            rms_after_base = rms_list[i] * base_boost
            target_channel_rms = floor_frac * rms_resid
            if rms_after_base < target_channel_rms and rms_list[i] > 1e-12:
                scales[i] = max(scales[i], target_channel_rms / rms_list[i])

        final_betas = [betas[i] * scales[i] for i in range(K)]
        final_rms   = [rms_list[i] * scales[i] for i in range(K)]

        print(f"  boost={base_boost:.2f}")
        _apply_scales(model, names, final_betas, final_rms, rms_resid, "OLS")


# ── Dispatch ────────────────────────────────────────────────────────────

def init_network_scale(model: "Stage2ModelPT", u: torch.Tensor,
                       cfg=None) -> None:
    _init_network_scale_ols(model, u)


def init_all_from_data(model: "Stage2ModelPT", u: torch.Tensor, cfg=None):
    init_W_from_config(model, cfg, u=u)
    init_G_from_config(model, cfg, u=u)
    baseline = init_I0(model, u)
    init_reversals(model, u, baseline, cfg)
    init_network_scale(model, u, cfg)
    # Optionally pre-compute correlation-weighted regularisation mask
    corr_reg = float(getattr(cfg, "corr_reg_weight", 0.0) or 0.0) if cfg else 0.0
    if corr_reg > 0:
        init_corr_reg_mask(model, u)
