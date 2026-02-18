"""VSRVRT: Vapoursynth plugin for RVRT (Recurrent Video Restoration Transformer)

RVRT is a state-of-the-art video restoration method supporting:
- Video denoising (non-blind, sigma 0-50)
- Video deblurring (GoPro and DVD models)
- Video super-resolution (4x upscaling)

Usage:
    import vapoursynth as vs
    from vapoursynth import core
    import vsrvrt

    # Load video
    clip = core.bs.VideoSource("input.mp4")

    # Convert to RGB if needed
    clip = clip.resize.Bicubic(format=vs.RGB24)

    # Apply denoising
    denoised = vsrvrt.Denoise(clip, sigma=25.0)

    # Apply deblurring
    deblurred = vsrvrt.Deblur(clip, model="gopro")

    # Apply super-resolution (4x)
    upscaled = vsrvrt.SuperRes(clip, scale=4, model="reds")
"""

__version__ = "0.1.0"
__author__ = "RVRT Vapoursynth Plugin"

from .rvrt_filter import Denoise, Deblur, SuperRes, estimate_requirements
from .model_configs import list_available_tasks, get_config
from .rvrt_core import SPATIAL_TILE_SIZE, MAX_TEMPORAL_TILE, VRAM_USAGE_MARGIN

__all__ = [
    "Denoise",
    "Deblur",
    "SuperRes",
    "estimate_requirements",
    "list_available_tasks",
    "get_config",
    "SPATIAL_TILE_SIZE",
    "MAX_TEMPORAL_TILE",
    "VRAM_USAGE_MARGIN",
]
