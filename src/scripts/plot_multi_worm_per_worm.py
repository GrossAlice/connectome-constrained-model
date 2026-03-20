"""Generate per-worm single-worm diagnostic plots from a saved multi-worm run.

Usage (from the src/ directory):
    python -m scripts.plot_multi_worm_per_worm \
        --model_pt output_plots/stage2/multi-pair1/model_best.pt \
        --worm_states_pt output_plots/stage2/multi-pair1/worm_states_final.pt \
        --save_dir output_plots/stage2/multi-pair1

If --worm_states_pt is not found (e.g. from an older run), the script falls back
to using the WormState parameters as initialised from the data (OLS lambda_u, I0=0).

The script generates, for each worm:
    <save_dir>/<worm_id>/00_summary_slide.png
    <save_dir>/<worm_id>/param_trajectories.png  (empty — no epoch history)
    <save_dir>/<worm_id>/02_prediction_traces.png
    <save_dir>/<worm_id>/ridge_cv_diagnostics.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the src/ directory is on the path when running as a script
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch

from stage2.config import Stage2PTConfig, MultiWormConfig, make_config
from stage2.io_multi import load_multi_worm_data
from stage2.model import Stage2ModelPT
from stage2.worm_state import build_worm_states
from stage2.init_from_data import init_all_from_data
from stage2.plot_eval import generate_eval_loo_plots


def build_cfg(h5_paths: list[str], device: str = "cpu",
              atlas_min_worm_count: int = 0) -> Stage2PTConfig:
    """Minimal config for loading data and running evaluation."""
    # Use the first H5 path as the nominal single-worm path (required by DataConfig)
    cfg = make_config(h5_paths[0], device=device)
    cfg.multi = MultiWormConfig(
        multi_worm=True,
        h5_paths=tuple(h5_paths),
        atlas_min_worm_count=atlas_min_worm_count,
    )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate per-worm single-worm plots from a saved multi-worm run."
    )
    ap.add_argument(
        "--model_pt", required=True,
        help="Path to model_best.pt (or model_final.pt).",
    )
    ap.add_argument(
        "--worm_states_pt", default=None,
        help="Path to worm_states_final.pt (saved per-worm parameters). "
             "Optional — falls back to freshly-initialised values if absent.",
    )
    ap.add_argument(
        "--h5", nargs="+", required=True,
        help="Paths to the H5 files used during training (same order).",
    )
    ap.add_argument(
        "--save_dir", required=True,
        help="Directory in which to write per-worm subdirectories.",
    )
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu).")
    ap.add_argument(
        "--atlas_min_worm_count", type=int, default=0,
        help="Restrict atlas to neurons observed in at least this many worms. "
             "Use the same value as during training (e.g. 2 for intersection-only atlas).",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Config & data ──────────────────────────────────────────────────────
    print("[PostHoc] Loading multi-worm data …")
    cfg = build_cfg(args.h5, args.device, atlas_min_worm_count=args.atlas_min_worm_count)
    data = load_multi_worm_data(cfg)
    atlas_lbl = data["atlas_labels"]
    N_atlas = data["atlas_size"]
    worm_dicts = data["worms"]
    common_dt = float(worm_dicts[0]["dt"]) if worm_dicts else float(getattr(cfg.multi, "common_dt", 0.6))
    print(f"[PostHoc] Loaded {len(worm_dicts)} worms, atlas size = {N_atlas}, dt = {common_dt}")

    # ── 2. Build shared model ─────────────────────────────────────────────────
    print("[PostHoc] Building model …")
    all_lu = [torch.as_tensor(wd["lambda_u_init"]) for wd in worm_dicts]
    lu_init = torch.stack(all_lu).median(dim=0).values.to(device)

    model = Stage2ModelPT(
        N_atlas,
        data["T_e"].to(device),
        data["T_sv"].to(device) if data["T_sv"] is not None else None,
        data["T_dcv"].to(device) if data["T_dcv"] is not None else None,
        common_dt,
        cfg,
        device,
        d_ell=0,
        lambda_u_init=lu_init,
        sign_t=data["sign_t"].to(device) if data["sign_t"] is not None else None,
    ).to(device)

    # ── 3. Load saved shared-model weights ────────────────────────────────────
    print(f"[PostHoc] Loading model weights from {args.model_pt} …")
    state_dict = torch.load(args.model_pt, map_location=device, weights_only=False)
    # snapshot_model_state stores raw params + derived (W_sv, lambda_u, G, …)
    # load only the nn.Parameter keys (those present in model.state_dict)
    model_sd = model.state_dict()
    loadable = {k: v for k, v in state_dict.items() if k in model_sd}
    missing = [k for k in model_sd if k not in loadable]
    model_sd.update(loadable)
    model.load_state_dict(model_sd, strict=False)
    # Also restore constrained buffers stored in snapshot (E_sv, E_dcv, etc.)
    with torch.no_grad():
        for name in ("E_sv", "E_dcv"):
            if name in state_dict and hasattr(model, name):
                try:
                    getattr(model, name).copy_(state_dict[name].to(device))
                except Exception:
                    pass
    print(f"[PostHoc]   Loaded {len(loadable)} tensors"
          + (f"  ({len(missing)} missing: {missing[:5]})" if missing else ""))
    model.eval()

    # ── 4. Build WormStates ───────────────────────────────────────────────────
    print("[PostHoc] Building WormState objects …")
    worm_states = build_worm_states(data, cfg)
    for ws in worm_states:
        ws.to(device)

    # ── 5. Optionally load saved per-worm params ──────────────────────────────
    ws_pt = args.worm_states_pt
    if ws_pt is not None and Path(ws_pt).exists():
        print(f"[PostHoc] Loading per-worm parameters from {ws_pt} …")
        worm_states_state = torch.load(ws_pt, map_location=device)
        for ws in worm_states:
            if ws.worm_id in worm_states_state:
                ws_sd = worm_states_state[ws.worm_id]
                with torch.no_grad():
                    for name, param in ws.named_parameters():
                        if name in ws_sd:
                            param.data.copy_(ws_sd[name])
                print(f"[PostHoc]   Loaded per-worm params for {ws.worm_id}")
            else:
                print(f"[PostHoc]   No saved params for {ws.worm_id} — using init values")
    else:
        if ws_pt is not None:
            print(f"[PostHoc] WARNING: {ws_pt} not found — using freshly-initialised per-worm params")
        else:
            print("[PostHoc] No --worm_states_pt given — using freshly-initialised per-worm params")

    # ── 6. Resolve motor neuron indices from names ────────────────────────────
    motor_neurons_raw = getattr(cfg, "motor_neurons", None)
    if motor_neurons_raw is not None:
        lbl_lower = [str(l).strip().lower() for l in atlas_lbl]
        resolved_motor: list[int] = []
        for mn in motor_neurons_raw:
            try:
                idx = int(mn)
                if 0 <= idx < N_atlas:
                    resolved_motor.append(idx)
            except (ValueError, TypeError):
                key = str(mn).strip().lower()
                if key in lbl_lower:
                    resolved_motor.append(lbl_lower.index(key))
        cfg.motor_neurons = tuple(resolved_motor) if resolved_motor else None
        print(f"[PostHoc] Resolved {len(resolved_motor)} motor neurons from atlas")

    # ── 7. Save original shared-model lambda_u / I0 ──────────────────────────
    orig_lambda_u_raw = model._lambda_u_raw.data.clone()
    orig_I0 = model.I0.data.clone()

    # ── 7. Per-worm single-worm plots ─────────────────────────────────────────
    for ws in worm_states:
        worm_save = save_dir / ws.worm_id
        worm_save.mkdir(parents=True, exist_ok=True)
        print(f"\n[PostHoc] Generating plots for {ws.worm_id} → {worm_save}")
        try:
            # Patch shared model with this worm's per-worm lambda_u / I0
            with torch.no_grad():
                model._lambda_u_raw.data.copy_(ws._lambda_u_raw.data)
                model.I0.data.copy_(ws.I0.data)

            beh_t = ws.behaviour
            worm_data = {
                "u_stage1":      ws.assemble_detached().detach(),
                "sigma_u":       ws.sigma_u,
                "gating":        ws.gating,
                "stim":          ws.stim,
                "dt":            common_dt,
                "neuron_labels": atlas_lbl,
                "b":      torch.nan_to_num(beh_t, nan=0.0) if beh_t is not None else None,
                "b_mask": torch.isfinite(beh_t).float()   if beh_t is not None else None,
            }

            generate_eval_loo_plots(
                model, worm_data, cfg, [],
                str(worm_save), show=False,
            )
            print(f"[PostHoc]   Done: {ws.worm_id}")
        except Exception as exc:
            import traceback
            print(f"[PostHoc]   FAILED for {ws.worm_id}: {exc}")
            traceback.print_exc()
        finally:
            # Always restore original shared-model params
            with torch.no_grad():
                model._lambda_u_raw.data.copy_(orig_lambda_u_raw)
                model.I0.data.copy_(orig_I0)

    print(f"\n[PostHoc] All per-worm plots saved under {save_dir}")


if __name__ == "__main__":
    main()
