"""Model configurations for RVRT tasks.

Configurations match the original RVRT paper implementation exactly.
See: https://github.com/JingyunLiang/RVRT
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RVRTConfig:
    """Configuration for RVRT model."""

    task: str
    upscale: int
    clip_size: int
    img_size: List[int]
    window_size: List[int]
    num_blocks: List[int]
    depths: List[int]
    embed_dims: List[int]
    num_heads: List[int]
    inputconv_groups: List[int]
    deformable_groups: int
    attention_heads: int
    attention_window: List[int]
    cpu_cache_length: int
    nonblind_denoising: bool
    model_url: str
    description: str


# Task configurations matching RVRT paper exactly
TASK_CONFIGS = {
    # Video Super-Resolution tasks
    "001_RVRT_videosr_bi_REDS_30frames": RVRTConfig(
        task="001_RVRT_videosr_bi_REDS_30frames",
        upscale=4,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[144, 144, 144],
        num_heads=[6, 6, 6],
        inputconv_groups=[1, 1, 1, 1, 1, 1],
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=128,
        nonblind_denoising=False,
        model_url="https://github.com/JingyunLiang/RVRT/releases/download/v0.0/001_RVRT_videosr_bi_REDS_30frames.pth",
        description="Video Super-Resolution (BI) trained on REDS, 4x upscale",
    ),
    "002_RVRT_videosr_bi_Vimeo_14frames": RVRTConfig(
        task="002_RVRT_videosr_bi_Vimeo_14frames",
        upscale=4,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[144, 144, 144],
        num_heads=[6, 6, 6],
        inputconv_groups=[1, 1, 1, 1, 1, 1],
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=128,
        nonblind_denoising=False,
        model_url="https://github.com/JingyunLiang/RVRT/releases/download/v0.0/002_RVRT_videosr_bi_Vimeo_14frames.pth",
        description="Video Super-Resolution (BI) trained on Vimeo, 4x upscale",
    ),
    "003_RVRT_videosr_bd_Vimeo_14frames": RVRTConfig(
        task="003_RVRT_videosr_bd_Vimeo_14frames",
        upscale=4,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[144, 144, 144],
        num_heads=[6, 6, 6],
        inputconv_groups=[1, 1, 1, 1, 1, 1],
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=128,
        nonblind_denoising=False,
        model_url="https://github.com/JingyunLiang/RVRT/releases/download/v0.0/003_RVRT_videosr_bd_Vimeo_14frames.pth",
        description="Video Super-Resolution (BD) trained on Vimeo, 4x upscale",
    ),
    # Video Deblurring tasks
    "004_RVRT_videodeblurring_DVD_16frames": RVRTConfig(
        task="004_RVRT_videodeblurring_DVD_16frames",
        upscale=1,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[192, 192, 192],
        num_heads=[6, 6, 6],
        inputconv_groups=[1, 3, 3, 3, 3, 3],
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=128,
        nonblind_denoising=False,
        model_url="https://github.com/JingyunLiang/RVRT/releases/download/v0.0/004_RVRT_videodeblurring_DVD_16frames.pth",
        description="Video Deblurring trained on DVD dataset",
    ),
    "005_RVRT_videodeblurring_GoPro_16frames": RVRTConfig(
        task="005_RVRT_videodeblurring_GoPro_16frames",
        upscale=1,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[192, 192, 192],
        num_heads=[6, 6, 6],
        inputconv_groups=[1, 3, 3, 3, 3, 3],
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=128,
        nonblind_denoising=False,
        model_url="https://github.com/JingyunLiang/RVRT/releases/download/v0.0/005_RVRT_videodeblurring_GoPro_16frames.pth",
        description="Video Deblurring trained on GoPro dataset",
    ),
    # Video Denoising task
    "006_RVRT_videodenoising_DAVIS_16frames": RVRTConfig(
        task="006_RVRT_videodenoising_DAVIS_16frames",
        upscale=1,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[192, 192, 192],
        num_heads=[6, 6, 6],
        inputconv_groups=[1, 3, 4, 6, 8, 4],
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=128,
        nonblind_denoising=True,
        model_url="https://github.com/JingyunLiang/RVRT/releases/download/v0.0/006_RVRT_videodenoising_DAVIS_16frames.pth",
        description="Video Denoising trained on DAVIS (non-blind, sigma 0-50)",
    ),
}


def get_config(task: str) -> RVRTConfig:
    """Get configuration for a specific task."""
    if task not in TASK_CONFIGS:
        available = ", ".join(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown task: {task}. Available tasks: {available}")
    return TASK_CONFIGS[task]


def get_default_task(task_type: str) -> str:
    """Get default task for a given type.

    Args:
        task_type: One of 'denoise', 'deblur', 'superres'
    """
    defaults = {
        "denoise": "006_RVRT_videodenoising_DAVIS_16frames",
        "deblur": "005_RVRT_videodeblurring_GoPro_16frames",
        "superres": "001_RVRT_videosr_bi_REDS_30frames",
    }
    if task_type not in defaults:
        raise ValueError(f"Unknown task type: {task_type}")
    return defaults[task_type]


def list_available_tasks():
    """List all available tasks with descriptions."""
    for task_id, config in TASK_CONFIGS.items():
        print(f"{task_id}: {config.description}")
