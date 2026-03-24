from __future__ import annotations

import argparse
from pathlib import Path
from .config import Stage2PTConfig, make_config
from . import get_stage2_logger
from .train import train_stage2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 model")
    parser.add_argument("--h5", dest="h5_path", default=None, nargs="+",
                        help="Path(s) to input HDF5. One path → single-worm; "
                             "multiple paths → multi-worm.")
    parser.add_argument("--save_dir", default=None,
                        help="Directory for diagnostic plots")
    parser.add_argument("--device", default=None,
                        help="torch device (default: from config, usually cuda)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--set", nargs=2, action="append", metavar=("KEY", "VALUE"),
                        default=[], dest="overrides",
                        help="Override config key with value, e.g. --set stim_kernel_len 20")
    args = parser.parse_args(argv)

    h5_list = args.h5_path or []
    is_multi = len(h5_list) > 1

    extra_kw = {}
    if args.device:
        extra_kw["device"] = args.device
    if is_multi:
        extra_kw["multi_worm"] = True
        extra_kw["h5_paths"] = tuple(h5_list)

    # Parse --set overrides: attempt int, then float, then bool, else str
    for key, val in args.overrides:
        for converter in (int, float):
            try:
                val = converter(val)
                break
            except ValueError:
                continue
        else:
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
        extra_kw[key] = val

    cfg = make_config(h5_list[0] if h5_list else "", **extra_kw)

    if cfg.multi_worm:
        if not cfg.h5_paths:
            parser.error("multi_worm=True requires h5_paths in config")
        from .train_multi import train_multi_worm
        train_multi_worm(cfg, save_dir=args.save_dir, show=args.show)
    else:
        if not h5_list:
            parser.error("--h5 is required (or pass multiple paths for multi-worm)")
        eval_result = train_stage2(cfg, save_dir=args.save_dir, show=args.show)
        if cfg.make_posture_video and eval_result is not None:
            _make_posture_video(cfg, eval_result, args.save_dir)


def _make_posture_video(
    cfg: Stage2PTConfig, eval_result: dict, save_dir: str | None,
) -> None:
    try:
        try:
            from stage2.posture_videos import make_posture_compare_video
        except ModuleNotFoundError:
            from scripts.posture_compare_video import make_posture_compare_video

        beh = eval_result.get("beh")
        if beh is None:
            get_stage2_logger().warning("posture_video_skipped",
                                        reason="no_behaviour_predictions")
            return
        out_path = (cfg.posture_video_out
                    or str(Path(save_dir or ".") / "13_posture_comparison.mp4"))
        make_posture_compare_video(
            h5_path=str(cfg.h5_path),
            out_path=out_path,
            ew_raw=beh["b_actual"],
            ew_stage1=beh["b_pred_gt"],
            ew_model_cv=beh["b_pred_model"],
            fps=cfg.posture_video_fps,
            dpi=cfg.posture_video_dpi,
            max_frames=cfg.posture_video_max_frames,
        )
    except Exception as e:
        import traceback
        get_stage2_logger().warning("posture_video_failed", error=str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
