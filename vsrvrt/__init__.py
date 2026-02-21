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

import os
import sys

__version__ = "1.1.0"
__author__ = "RVRT Vapoursynth Plugin"


def _add_vapoursynth_dll_path():
    """Add VapourSynth DLL directory to search path on Windows.
    
    Python 3.8+ requires explicit DLL directory registration via
    os.add_dll_directory() for DLLs not in standard system paths.
    """
    if sys.platform != "win32":
        return
    
    # Check if already loaded
    try:
        import vapoursynth
        return
    except ImportError:
        pass
    
    vs_paths = [
        os.environ.get("VAPOURSYNTH_PATH"),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "VapourSynth", "core"),
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "VapourSynth", "core"),
    ]
    
    for vs_path in vs_paths:
        if vs_path and os.path.isdir(vs_path):
            try:
                os.add_dll_directory(vs_path)
                return
            except OSError:
                pass


_add_vapoursynth_dll_path()


def _check_cuda_available():
    """Verify PyTorch has CUDA support at import time.
    
    Provides a clear error message if the user installed the CPU-only
    version of PyTorch instead of the CUDA version.
    """
    import torch
    
    if not torch.cuda.is_available():
        cuda_version = getattr(torch.version, 'cuda', None)
        
        if cuda_version is None:
            raise RuntimeError(
                "vsrvrt requires PyTorch with CUDA support, but you have the CPU-only version installed.\n\n"
                "To fix this, reinstall PyTorch with CUDA:\n"
                "  pip uninstall torch torchvision -y\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n\n"
                "For other CUDA versions, see: https://pytorch.org/get-started/locally/"
            )
        else:
            raise RuntimeError(
                f"PyTorch was built with CUDA {cuda_version}, but CUDA is not available.\n\n"
                "This usually means:\n"
                "1. You don't have an NVIDIA GPU\n"
                "2. Your GPU driver is too old\n"
                "3. CUDA toolkit is not installed\n\n"
                "Please ensure you have:\n"
                "- An NVIDIA GPU with CUDA support\n"
                "- Updated GPU drivers\n"
                f"- CUDA {cuda_version} compatible drivers (or newer with forward compatibility)"
            )


_check_cuda_available()

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
