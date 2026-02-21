"""Core RVRT inference wrapper with FP16 support and model management."""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import requests
from tqdm import tqdm

from .rvrt_src.models.network_rvrt import RVRT
from .model_configs import RVRTConfig, get_config


SPATIAL_TILE_SIZE = 256
MAX_TEMPORAL_TILE = 64
VRAM_USAGE_MARGIN = 0.85


class RVRTInference:
    """RVRT inference wrapper with FP16 support and efficient model management."""

    # Model cache to avoid reloading
    _model_cache = {}
    _device = None

    def __init__(
        self,
        config: RVRTConfig,
        use_fp16: bool = True,
        device: Optional[torch.device] = None,
    ):
        """Initialize RVRT inference.

        Args:
            config: Model configuration
            use_fp16: Whether to use FP16 precision
            device: Device to run on (defaults to CUDA if available)
        """
        self.config = config
        self.use_fp16 = use_fp16

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        RVRTInference._device = self.device

        # Load model
        self.model = self._load_model()

        # Setup for inference
        self.model.eval()
        # Enable FP16 for faster inference on CUDA
        if self.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        self.model = self.model.to(self.device)

    def _get_model_path(self) -> Path:
        """Get path to model weights, downloading if necessary."""
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)

        model_name = f"{self.config.task}.pth"
        model_path = model_dir / model_name

        if not model_path.exists():
            print(f"Downloading model: {self.config.task}")
            self._download_model(self.config.model_url, model_path)

        return model_path

    def _download_model(self, url: str, output_path: Path):
        """Download model with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(output_path, "wb") as f,
            tqdm(
                desc=output_path.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                pbar.update(size)

    def _load_model(self) -> nn.Module:
        """Load or retrieve cached model."""
        cache_key = self.config.task

        if cache_key in RVRTInference._model_cache:
            print(f"Using cached model: {self.config.task}")
            return RVRTInference._model_cache[cache_key]

        # Create model with config parameters
        model = RVRT(
            upscale=self.config.upscale,
            clip_size=self.config.clip_size,
            img_size=self.config.img_size,
            window_size=self.config.window_size,
            num_blocks=self.config.num_blocks,
            depths=self.config.depths,
            embed_dims=self.config.embed_dims,
            num_heads=self.config.num_heads,
            inputconv_groups=self.config.inputconv_groups,
            deformable_groups=self.config.deformable_groups,
            attention_heads=self.config.attention_heads,
            attention_window=self.config.attention_window,
            cpu_cache_length=self.config.cpu_cache_length,
            nonblind_denoising=self.config.nonblind_denoising,
        )

        # Load weights
        model_path = self._get_model_path()
        print(f"Loading model from: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "params" in checkpoint:
            state_dict = checkpoint["params"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)

        # Cache model
        RVRTInference._model_cache[cache_key] = model

        return model

    @torch.no_grad()
    def inference(
        self,
        clip: torch.Tensor,
        sigma: Optional[float] = None,
        tile_size: Optional[Tuple[int, int, int]] = None,
        tile_overlap: Tuple[int, int, int] = (2, 20, 20),
    ) -> torch.Tensor:
        """Run inference on a video clip.

        Args:
            clip: Input tensor of shape (B, T, C, H, W) in [0, 1] range
            sigma: Noise level for denoising (0-50), scaled to [0, 1]
            tile_size: Tuple of (T, H, W) for tiling, None for automatic
            tile_overlap: Overlap for tiling (T, H, W)

        Returns:
            Output tensor of shape (B, T, C, H*scale, W*scale)
        """
        # Add sigma channel for non-blind denoising
        if self.config.nonblind_denoising and sigma is not None:
            B, T, C, H, W = clip.shape
            sigma_map = torch.full(
                (B, T, 1, H, W), sigma, dtype=clip.dtype, device=clip.device
            )
            clip = torch.cat([clip, sigma_map], dim=2)

        # Move to device and convert to FP16 if enabled
        clip = clip.to(self.device)
        if self.use_fp16 and self.device.type == "cuda":
            clip = clip.half()

        # Automatic tiling if not specified
        if tile_size is None:
            tile_size = self._get_auto_tile_size(clip)

        # Run inference with tiling
        output = self._test_video(clip, tile_size, tile_overlap)

        # Convert back to float32 if needed
        if self.use_fp16:
            output = output.float()

        return output

    def _get_available_vram(self) -> int:
        """Get available VRAM in bytes."""
        if self.device.type != "cuda":
            return 0
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        return total_memory - allocated

    def _estimate_tile_memory(self, num_frames: int, tile_h: int, tile_w: int) -> int:
        """Estimate VRAM needed for given tile configuration.

        Calibrated based on real measurements:
        - Denoise (64,256,256) ~9GB with 192 embed_dims
        - SuperRes (8,256,256) ~11.5GB with 144 embed_dims, 4x upscale

        Memory scales roughly:
        - Base (model + overhead): ~0.5 GB
        - Per-frame denoise (256x256): ~0.14 GB
        - Per-frame superres (256x256 input → 1024x1024 output): ~1.4 GB (10x denoise)
        """
        bytes_per_elem = 2 if self.use_fp16 else 4
        base_memory = 0.5 * 1e9

        tile_pixels = tile_h * tile_w
        ref_pixels = 256 * 256
        pixel_scale = tile_pixels / ref_pixels

        per_frame_base = 0.125 * 1e9 * pixel_scale

        if self.config.upscale > 1:
            output_scale = self.config.upscale**2
            per_frame_memory = per_frame_base * (1 + output_scale * 0.56)
        else:
            per_frame_memory = per_frame_base

        if not hasattr(self, "_model_memory"):
            self._model_memory = sum(
                p.numel() * bytes_per_elem for p in self.model.parameters()
            )

        model_mem = self._model_memory

        total = base_memory + per_frame_memory * num_frames + model_mem
        total *= 1.02

        return int(total)

    def _find_max_temporal_frames(
        self, spatial_tile: int, available_vram: int, max_frames: int
    ) -> int:
        """Binary search for max temporal frames that fit in VRAM."""
        min_frames = self.config.clip_size * 2
        max_frames = min(max_frames, MAX_TEMPORAL_TILE)

        if (
            self._estimate_tile_memory(max_frames, spatial_tile, spatial_tile)
            <= available_vram
        ):
            return max_frames

        if (
            self._estimate_tile_memory(min_frames, spatial_tile, spatial_tile)
            > available_vram
        ):
            print(
                f"RVRT: Warning - even minimum tile size ({min_frames} frames) may exceed VRAM"
            )
            return min_frames

        low, high = min_frames, max_frames
        result = min_frames

        while low <= high:
            mid = (low + high) // 2
            estimated = self._estimate_tile_memory(mid, spatial_tile, spatial_tile)

            if estimated <= available_vram:
                result = mid
                low = mid + 1
            else:
                high = mid - 1

        return result

    def _get_auto_tile_size(self, clip: torch.Tensor) -> Tuple[int, int, int]:
        """Determine optimal tile size based on available VRAM.

        Spatial tile is FIXED at 256x256 (matches model training).
        Only temporal dimension is auto-calculated via binary search.
        """
        if self.device.type != "cuda":
            return (min(clip.shape[1], 16), SPATIAL_TILE_SIZE, SPATIAL_TILE_SIZE)

        B, T, C, H, W = clip.shape
        available_vram = int(self._get_available_vram() * VRAM_USAGE_MARGIN)

        spatial_tile = SPATIAL_TILE_SIZE

        optimal_frames = self._find_max_temporal_frames(spatial_tile, available_vram, T)

        print(f"RVRT: Auto-tiling: ({optimal_frames}, {spatial_tile}, {spatial_tile})")
        print(
            f"RVRT: Available VRAM: {available_vram / 1e9:.1f} GB, "
            f"Estimated: {self._estimate_tile_memory(optimal_frames, spatial_tile, spatial_tile) / 1e9:.1f} GB"
        )

        return (optimal_frames, spatial_tile, spatial_tile)

    def _test_video(
        self,
        lq: torch.Tensor,
        tile: Tuple[int, int, int],
        tile_overlap: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Test video with temporal and spatial tiling."""
        num_frame_testing = tile[0]

        if num_frame_testing == 0:
            # Test as one clip (the whole video)
            return self._test_clip_full(lq)
        else:
            # Test as multiple clips
            return self._test_video_tiled(lq, tile, tile_overlap)

    def _test_video_tiled(
        self,
        lq: torch.Tensor,
        tile: Tuple[int, int, int],
        tile_overlap: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Test video with temporal tiling."""
        sf = self.config.upscale
        num_frame_testing = tile[0]
        num_frame_overlapping = tile_overlap[0]
        # NOTE: Original RVRT sets not_overlap_border=False for temporal tiling
        # meaning NO border blending is applied to temporal tiles
        not_overlap_border = False

        b, d, c, h, w = lq.size()
        c = c - 1 if self.config.nonblind_denoising else c
        stride = num_frame_testing - num_frame_overlapping
        d_idx_list = list(range(0, d - num_frame_testing, stride)) + [
            max(0, d - num_frame_testing)
        ]

        # Allocate output tensors on CPU to save VRAM, only move processed clips to GPU
        # This is necessary for long videos that would otherwise exceed GPU memory
        E = torch.zeros(b, d, c, h * sf, w * sf, device="cpu", dtype=lq.dtype)
        W = torch.zeros(b, d, 1, 1, 1, device="cpu", dtype=lq.dtype)

        for d_idx in d_idx_list:
            lq_clip = lq[:, d_idx : d_idx + num_frame_testing, ...]
            out_clip = self._test_clip(lq_clip, tile, tile_overlap)
            out_clip_mask = torch.ones(
                (b, min(num_frame_testing, d), 1, 1, 1),
                device=lq.device,
                dtype=lq.dtype,
            )

            # NOTE: Border blending is DISABLED for temporal tiles in original RVRT
            # (not_overlap_border=False). This preserves temporal coherence.
            if not_overlap_border:
                if d_idx < d_idx_list[-1]:
                    # Not the last clip: zero out the end of this clip
                    out_clip[:, -num_frame_overlapping // 2 :, ...] *= 0
                    out_clip_mask[:, -num_frame_overlapping // 2 :, ...] *= 0
                if d_idx > d_idx_list[0]:
                    # Not the first clip: zero out the beginning of this clip
                    out_clip[:, : num_frame_overlapping // 2, ...] *= 0
                    out_clip_mask[:, : num_frame_overlapping // 2, ...] *= 0

            # Move to CPU and accumulate (saves VRAM for long videos)
            E[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip.cpu())
            W[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip_mask.cpu())

            # Clear GPU memory after each temporal clip
            del out_clip, out_clip_mask
            torch.cuda.empty_cache()

        output = E.div_(W)
        # Keep output on CPU to avoid OOM when returning large tensors
        # The filter will move individual frames to GPU as needed
        return output

    def _test_clip_full(self, lq: torch.Tensor) -> torch.Tensor:
        """Test clip without temporal tiling."""
        clip_size = self.config.clip_size
        d_old = lq.size(1)
        d_pad = (clip_size - d_old % clip_size) % clip_size

        if d_pad > 0:
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)

        output = self.model(lq)
        output = output[:, :d_old, :, :, :]

        return output.detach().cpu()

    def _test_clip(
        self,
        lq: torch.Tensor,
        tile: Tuple[int, int, int],
        tile_overlap: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Test clip with optional spatial tiling."""
        sf = self.config.upscale
        window_size = self.config.window_size
        size_patch_testing = tile[1]

        if size_patch_testing == 0:
            # No spatial tiling
            return self._test_clip_no_spatial_tiling(lq)

        # Validate tile size is multiple of window size (matches original RVRT)
        assert size_patch_testing % window_size[-1] == 0, (
            "testing patch size should be a multiple of window_size."
        )

        # Spatial tiling
        overlap_size = tile_overlap[1]

        b, d, c, h, w = lq.size()
        c = c - 1 if self.config.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h - size_patch_testing, stride)) + [
            max(0, h - size_patch_testing)
        ]
        w_idx_list = list(range(0, w - size_patch_testing, stride)) + [
            max(0, w - size_patch_testing)
        ]

        # Allocate on CPU to save VRAM
        E = torch.zeros(b, d, c, h * sf, w * sf, device="cpu", dtype=lq.dtype)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[
                    ...,
                    h_idx : h_idx + size_patch_testing,
                    w_idx : w_idx + size_patch_testing,
                ]
                out_patch = self._test_clip_no_spatial_tiling(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                # Apply spatial border blending: zero out overlapping regions
                # This matches the original RVRT implementation
                overlap_sf_h = overlap_size // 2 * sf
                overlap_sf_w = overlap_size // 2 * sf

                if h_idx < h_idx_list[-1]:
                    # Not the last row: zero out the bottom
                    out_patch[..., -overlap_sf_h:, :] *= 0
                    out_patch_mask[..., -overlap_sf_h:, :] *= 0
                if h_idx > h_idx_list[0]:
                    # Not the first row: zero out the top
                    out_patch[..., :overlap_sf_h, :] *= 0
                    out_patch_mask[..., :overlap_sf_h, :] *= 0
                if w_idx < w_idx_list[-1]:
                    # Not the last column: zero out the right
                    out_patch[..., :, -overlap_sf_w:] *= 0
                    out_patch_mask[..., :, -overlap_sf_w:] *= 0
                if w_idx > w_idx_list[0]:
                    # Not the first column: zero out the left
                    out_patch[..., :, :overlap_sf_w] *= 0
                    out_patch_mask[..., :, :overlap_sf_w] *= 0

                # Move to CPU and accumulate
                E[
                    ...,
                    h_idx * sf : (h_idx + size_patch_testing) * sf,
                    w_idx * sf : (w_idx + size_patch_testing) * sf,
                ].add_(out_patch.cpu())
                W[
                    ...,
                    h_idx * sf : (h_idx + size_patch_testing) * sf,
                    w_idx * sf : (w_idx + size_patch_testing) * sf,
                ].add_(out_patch_mask.cpu())

        output = E.div_(W)
        # Keep output on CPU to be consistent with temporal tiling path
        return output

    def _test_clip_no_spatial_tiling(self, lq: torch.Tensor) -> torch.Tensor:
        """Test clip without spatial tiling.

        Handles padding for both temporal and spatial dimensions.
        Temporal dimension must be multiple of clip_size (2).
        Spatial dimensions must be multiple of window_size (8).
        """
        window_size = self.config.window_size
        clip_size = self.config.clip_size
        _, d_old, _, h_old, w_old = lq.size()

        d_pad = (clip_size - d_old % clip_size) % clip_size
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        if d_pad > 0:
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
        if h_pad > 0:
            lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3)
        if w_pad > 0:
            lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4)

        output = self.model(lq)
        output = output[
            :, :d_old, :, : h_old * self.config.upscale, : w_old * self.config.upscale
        ]

        return output.detach().cpu()
