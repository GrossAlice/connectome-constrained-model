try:
    from stage2.posture_videos import make_posture_compare_video
    __all__ = ["make_posture_compare_video"]
except ImportError:
    __all__ = []