"""Vapoursynth filter functions for RVRT with chunked CPU streaming for long videos."""

import vapoursynth as vs
import numpy as np
import torch
from typing import Optional, Tuple, List

from .rvrt_core import (
    RVRTInference,
    SPATIAL_TILE_SIZE,
    MAX_TEMPORAL_TILE,
    VRAM_USAGE_MARGIN,
)
from .model_configs import RVRTConfig, get_config, get_default_task


def estimate_requirements(
    clip: vs.VideoNode,
    task: str = "denoise",
    model: Optional[str] = None,
    use_fp16: bool = True,
) -> dict:
    """Estimate VRAM requirements and recommended settings for processing.

    Args:
        clip: Input video clip
        task: Task type - 'denoise', 'deblur', or 'superres'
        model: Specific model name (optional, uses default for task if not specified)
        use_fp16: Whether using FP16 precision

    Returns:
        dict with keys: recommended_tile_size, estimated_vram_gb, available_vram_gb,
                       max_temporal_frames, task_info
    """
    if model:
        config = get_config(model)
    else:
        task_name = get_default_task(task)
        config = get_config(task_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_frames = clip.num_frames
    H, W = clip.height, clip.width

    result = {
        "task": config.task,
        "description": config.description,
        "upscale": config.upscale,
        "input_resolution": f"{W}x{H}",
        "total_frames": total_frames,
        "use_fp16": use_fp16,
    }

    if device.type != "cuda":
        result["device"] = "cpu"
        result["recommended_tile_size"] = (
            min(total_frames, 16),
            SPATIAL_TILE_SIZE,
            SPATIAL_TILE_SIZE,
        )
        result["note"] = "CPU mode - no VRAM constraints"
        return result

    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    available = total_memory - allocated
    available_with_margin = int(available * VRAM_USAGE_MARGIN)

    temp_inference = RVRTInference(config, use_fp16=use_fp16, device=device)

    min_frames = config.clip_size * 2
    max_possible = min(total_frames, MAX_TEMPORAL_TILE)

    if (
        temp_inference._estimate_tile_memory(
            max_possible, SPATIAL_TILE_SIZE, SPATIAL_TILE_SIZE
        )
        <= available_with_margin
    ):
        optimal_frames = max_possible
    elif (
        temp_inference._estimate_tile_memory(
            min_frames, SPATIAL_TILE_SIZE, SPATIAL_TILE_SIZE
        )
        <= available_with_margin
    ):
        low, high = min_frames, max_possible
        optimal_frames = min_frames
        while low <= high:
            mid = (low + high) // 2
            if (
                temp_inference._estimate_tile_memory(
                    mid, SPATIAL_TILE_SIZE, SPATIAL_TILE_SIZE
                )
                <= available_with_margin
            ):
                optimal_frames = mid
                low = mid + 1
            else:
                high = mid - 1
    else:
        optimal_frames = min_frames

    estimated_vram = temp_inference._estimate_tile_memory(
        optimal_frames, SPATIAL_TILE_SIZE, SPATIAL_TILE_SIZE
    )

    result.update(
        {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_properties(device).name,
            "recommended_tile_size": (
                optimal_frames,
                SPATIAL_TILE_SIZE,
                SPATIAL_TILE_SIZE,
            ),
            "estimated_vram_gb": round(estimated_vram / 1e9, 2),
            "available_vram_gb": round(available / 1e9, 2),
            "total_vram_gb": round(total_memory / 1e9, 2),
            "max_temporal_frames": optimal_frames,
            "vram_margin_percent": int(VRAM_USAGE_MARGIN * 100),
        }
    )

    return result


def _frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    """Convert Vapoursynth frame to PyTorch tensor.

    Supports both RGB24 (8-bit integer) and RGBS (32-bit float) formats.
    Output is always float32 tensor in [0, 1] range.
    """
    import ctypes

    width = frame.width
    height = frame.height
    fmt = frame.format

    # Check if input is float format (RGBS) or integer (RGB24)
    is_float = fmt.sample_type == vs.FLOAT
    bytes_per_sample = fmt.bytes_per_sample

    planes = []
    for i in range(3):
        ptr = frame.get_read_ptr(i)
        stride = frame.get_stride(i)
        addr = ptr.value

        if is_float:
            # RGBS: Read as float32
            # Calculate actual row size in elements (not bytes)
            row_size = stride // bytes_per_sample
            # Create float32 buffer
            buf = (ctypes.c_float * (row_size * height)).from_address(addr)
            arr = np.ctypeslib.as_array(buf).reshape((height, row_size))
            plane = arr[:, :width].copy()
            # Already in [0, 1] range, no division needed
        else:
            # RGB24: Read as uint8
            buf_size = stride * height
            buf = (ctypes.c_uint8 * buf_size).from_address(addr)
            arr = np.ctypeslib.as_array(buf).reshape((height, stride))
            plane = arr[:, :width].copy()
            # Convert to float32 and normalize to [0, 1]
            plane = plane.astype(np.float32) / 255.0

        planes.append(plane)

    img = np.stack(planes, axis=0)
    tensor = torch.from_numpy(img).unsqueeze(0)

    return tensor


def _create_filter_wrapper_chunked(
    src_clip: vs.VideoNode,
    config: RVRTConfig,
    tile_size: Optional[Tuple[int, int, int]],
    tile_overlap: Tuple[int, int, int],
    sigma: Optional[float],
    use_fp16: bool,
    device: Optional[str],
    chunk_size: int = 64,
    chunk_overlap: int = 16,
) -> vs.VideoNode:
    """Create a filter wrapper with chunked CPU streaming for long videos.

    This allows processing videos longer than GPU memory would allow by:
    1. Processing video in chunks with temporal overlap
    2. Only loading current chunk into GPU memory
    3. Blending overlapping regions for smooth transitions

    Args:
        chunk_size: Number of frames to process per chunk (default: 64)
        chunk_overlap: Number of overlapping frames between chunks (default: 16)
    """

    device_obj = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    inference = RVRTInference(config, use_fp16=use_fp16, device=device_obj)

    total_frames = src_clip.num_frames

    # Calculate chunk boundaries with overlap
    if total_frames <= chunk_size:
        # Short video - process all at once
        chunks = [(0, total_frames)]
    else:
        # Long video - process in overlapping chunks
        chunks = []
        stride = chunk_size - chunk_overlap
        start = 0
        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunks.append((start, end))
            if end >= total_frames:
                break
            start += stride

    print(f"RVRT: Processing {total_frames} frames in {len(chunks)} chunk(s)")
    print(f"RVRT: Chunk size={chunk_size}, overlap={chunk_overlap}")

    # Process all chunks and store results
    processed_chunks = []

    for chunk_idx, (start_frame, end_frame) in enumerate(chunks):
        num_frames = end_frame - start_frame
        print(
            f"RVRT: Processing chunk {chunk_idx + 1}/{len(chunks)} (frames {start_frame}-{end_frame - 1})"
        )

        # Load frames for this chunk
        frames = []
        for n in range(start_frame, end_frame):
            frame = src_clip.get_frame(n)
            tensor = _frame_to_tensor(frame)
            frames.append(tensor)

        # Stack into chunk tensor: (1, T, C, H, W)
        chunk_tensor = torch.stack(frames, dim=1)

        # Pad partial chunks to full chunk_size to avoid model dimension mismatches
        original_num_frames = num_frames
        if num_frames < chunk_size:
            # Calculate padding needed
            pad_length = chunk_size - num_frames
            # Repeat the last frame for padding
            last_frame = chunk_tensor[:, -1:, :, :, :]
            padding = last_frame.repeat(1, pad_length, 1, 1, 1)
            chunk_tensor = torch.cat([chunk_tensor, padding], dim=1)
            print(
                f"RVRT: Padded chunk from {original_num_frames} to {chunk_size} frames"
            )

        # Process chunk with RVRT
        sigma_normalized = sigma / 255.0 if sigma is not None else None

        with torch.no_grad():
            output_chunk = inference.inference(
                chunk_tensor,
                sigma=sigma_normalized,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )

        # Crop output back to original frame count if padded
        if original_num_frames < chunk_size:
            output_chunk = output_chunk[:, :original_num_frames, :, :, :]

        # Store on CPU
        processed_chunks.append((start_frame, end_frame, output_chunk.cpu()))

        # Clear GPU memory
        del chunk_tensor, output_chunk
        torch.cuda.empty_cache()

    # Blend overlapping regions between chunks
    if len(chunks) > 1:
        final_output = _blend_chunks(processed_chunks, total_frames, chunk_overlap)
    else:
        final_output = processed_chunks[0][2]

    print(f"RVRT: Final output shape: {final_output.shape}")

    def process_frame(n, f):
        """Extract frame n from the processed output."""
        import ctypes

        frame_tensor = final_output[0, n, ...]

        # Convert to numpy
        img = frame_tensor.cpu().numpy()

        # Create a writable copy
        fout = f.copy()
        h, w = img.shape[1], img.shape[2]

        if is_float_input:
            # RGBS output: write float32 data directly
            for p in range(3):
                ptr = fout.get_write_ptr(p)
                stride = fout.get_stride(p)
                addr = ptr.value
                bytes_per_sample = 4  # float32
                row_size = stride // bytes_per_sample
                # Create float32 buffer
                dst_buf = (ctypes.c_float * (row_size * h)).from_address(addr)
                dst_arr = np.ctypeslib.as_array(dst_buf).reshape((h, row_size))
                plane_data = img[p]
                dst_arr[:, :w] = plane_data
        else:
            # RGB24 output: convert to uint8
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            for p in range(3):
                ptr = fout.get_write_ptr(p)
                stride = fout.get_stride(p)
                addr = ptr.value
                buf_size = stride * h
                dst_buf = (ctypes.c_uint8 * buf_size).from_address(addr)
                dst_arr = np.ctypeslib.as_array(dst_buf).reshape((h, stride))
                plane_data = img[p]
                dst_arr[:, :w] = plane_data

        return fout

    # Detect input format to match output
    input_format = src_clip.format
    is_float_input = input_format.sample_type == vs.FLOAT
    output_format = vs.RGBS if is_float_input else vs.RGB24

    # Create output clip with matching format
    out_width = src_clip.width * config.upscale
    out_height = src_clip.height * config.upscale

    blank = vs.core.std.BlankClip(
        width=out_width,
        height=out_height,
        format=output_format,
        length=src_clip.num_frames,
        fpsnum=src_clip.fps.numerator,
        fpsden=src_clip.fps.denominator,
    )

    return blank.std.ModifyFrame(blank, process_frame)


def _blend_chunks(
    processed_chunks: List[Tuple[int, int, torch.Tensor]],
    total_frames: int,
    overlap: int,
) -> torch.Tensor:
    """Blend overlapping regions between chunks.

    Args:
        processed_chunks: List of (start_frame, end_frame, tensor) tuples
        total_frames: Total number of frames
        overlap: Number of overlapping frames

    Returns:
        Blended output tensor of shape (1, T, C, H, W)
    """
    # Get output shape from first chunk
    _, _, first_chunk = processed_chunks[0]
    _, _, C, H, W = first_chunk.shape

    # Allocate output tensor
    output = torch.zeros(1, total_frames, C, H, W, dtype=first_chunk.dtype)
    weights = torch.zeros(1, total_frames, 1, 1, 1, dtype=first_chunk.dtype)

    for i, (start, end, chunk) in enumerate(processed_chunks):
        num_frames = end - start

        # Create weight mask for this chunk
        chunk_weights = torch.ones(1, num_frames, 1, 1, 1, dtype=chunk.dtype)

        # Apply blending in overlap regions
        if i > 0 and overlap > 0:
            # Beginning of chunk overlaps with previous
            # Ramp up from 0 to 1
            for j in range(min(overlap, num_frames)):
                weight = (j + 1) / (overlap + 1)
                chunk_weights[0, j, 0, 0, 0] = weight

        if i < len(processed_chunks) - 1 and overlap > 0:
            # End of chunk overlaps with next
            # Ramp down from 1 to 0
            for j in range(max(0, num_frames - overlap), num_frames):
                frames_from_end = num_frames - j
                weight = frames_from_end / (overlap + 1)
                chunk_weights[0, j, 0, 0, 0] = min(chunk_weights[0, j, 0, 0, 0], weight)

        # Accumulate
        output[0, start:end] += chunk[0] * chunk_weights[0]
        weights[0, start:end] += chunk_weights[0]

    # Normalize by weights
    output = output / weights.clamp(min=1e-8)

    return output


def _create_filter_wrapper_preview(
    src_clip: vs.VideoNode,
    config: RVRTConfig,
    tile_size: Optional[Tuple[int, int, int]],
    tile_overlap: Tuple[int, int, int],
    sigma: Optional[float],
    use_fp16: bool,
    device: Optional[str],
    chunk_size: int = 64,
    chunk_overlap: int = 16,
) -> vs.VideoNode:
    """Create a filter wrapper with lazy chunk processing for preview mode.

    Processes chunks on-demand as frames are requested, caching results.
    This allows instant preview startup with quality trade-offs at chunk boundaries.

    Args:
        chunk_size: Number of frames to process per chunk (default: 64)
        chunk_overlap: Number of overlapping frames between chunks (default: 16)
    """
    device_obj = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    inference = RVRTInference(config, use_fp16=use_fp16, device=device_obj)

    total_frames = src_clip.num_frames

    # Calculate chunk boundaries
    if total_frames <= chunk_size:
        # Short video - single chunk
        chunks = [(0, total_frames)]
    else:
        # Long video - overlapping chunks
        chunks = []
        stride = chunk_size - chunk_overlap
        start = 0
        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunks.append((start, end))
            if end >= total_frames:
                break
            start += stride

    print(f"RVRT: Preview mode - {total_frames} frames in {len(chunks)} chunk(s)")
    print(f"RVRT: Chunk size={chunk_size}, processing on-demand")

    # Chunk cache - persists for the session
    chunk_cache = {}

    # Detect input format
    input_format = src_clip.format
    is_float_input = input_format.sample_type == vs.FLOAT
    output_format = vs.RGBS if is_float_input else vs.RGB24

    sigma_normalized = sigma / 255.0 if sigma is not None else None

    def process_frame(n, f):
        """Process frame n on-demand from cached chunks."""
        import ctypes

        # Determine which chunk contains frame n
        stride = chunk_size - chunk_overlap
        chunk_idx = n // stride if stride > 0 else 0
        # Clamp chunk index to valid range
        chunk_idx = min(chunk_idx, len(chunks) - 1)

        start_frame, end_frame = chunks[chunk_idx]

        # Check if chunk is cached
        if chunk_idx not in chunk_cache:
            # Process this chunk on-demand
            num_frames = end_frame - start_frame
            print(
                f"RVRT: Processing chunk {chunk_idx + 1}/{len(chunks)} "
                f"(frames {start_frame}-{end_frame - 1}) for preview"
            )

            # Load frames for this chunk
            frames = []
            for frame_n in range(start_frame, end_frame):
                frame = src_clip.get_frame(frame_n)
                tensor = _frame_to_tensor(frame)
                frames.append(tensor)

            # Stack into chunk tensor
            chunk_tensor = torch.stack(frames, dim=1)

            # Pad partial chunks
            original_num_frames = num_frames
            if num_frames < chunk_size:
                pad_length = chunk_size - num_frames
                last_frame = chunk_tensor[:, -1:, :, :, :]
                padding = last_frame.repeat(1, pad_length, 1, 1, 1)
                chunk_tensor = torch.cat([chunk_tensor, padding], dim=1)

            # Process chunk
            with torch.no_grad():
                output_chunk = inference.inference(
                    chunk_tensor,
                    sigma=sigma_normalized,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )

            # Crop if padded
            if original_num_frames < chunk_size:
                output_chunk = output_chunk[:, :original_num_frames, :, :, :]

            # Cache the chunk
            chunk_cache[chunk_idx] = output_chunk.cpu()

            # Clear GPU memory
            del chunk_tensor
            torch.cuda.empty_cache()

        # Get the cached chunk
        cached_chunk = chunk_cache[chunk_idx]

        # Calculate frame index within the chunk
        frame_in_chunk = n - start_frame

        # Extract the requested frame
        frame_tensor = cached_chunk[0, frame_in_chunk, ...]

        # Convert to numpy
        img = frame_tensor.cpu().numpy()

        # Create output frame
        fout = f.copy()
        h, w = img.shape[1], img.shape[2]

        if is_float_input:
            # RGBS output
            for p in range(3):
                ptr = fout.get_write_ptr(p)
                stride = fout.get_stride(p)
                addr = ptr.value
                bytes_per_sample = 4
                row_size = stride // bytes_per_sample
                dst_buf = (ctypes.c_float * (row_size * h)).from_address(addr)
                dst_arr = np.ctypeslib.as_array(dst_buf).reshape((h, row_size))
                plane_data = img[p]
                dst_arr[:, :w] = plane_data
        else:
            # RGB24 output
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            for p in range(3):
                ptr = fout.get_write_ptr(p)
                stride = fout.get_stride(p)
                addr = ptr.value
                buf_size = stride * h
                dst_buf = (ctypes.c_uint8 * buf_size).from_address(addr)
                dst_arr = np.ctypeslib.as_array(dst_buf).reshape((h, stride))
                plane_data = img[p]
                dst_arr[:, :w] = plane_data

        return fout

    # Create output clip
    out_width = src_clip.width * config.upscale
    out_height = src_clip.height * config.upscale

    blank = vs.core.std.BlankClip(
        width=out_width,
        height=out_height,
        format=output_format,
        length=src_clip.num_frames,
        fpsnum=src_clip.fps.numerator,
        fpsden=src_clip.fps.denominator,
    )

    return blank.std.ModifyFrame(blank, process_frame)


def _create_filter_wrapper(
    src_clip: vs.VideoNode,
    config: RVRTConfig,
    tile_size: Optional[Tuple[int, int, int]],
    tile_overlap: Tuple[int, int, int],
    sigma: Optional[float],
    use_fp16: bool,
    device: Optional[str],
    chunk_size: int = 64,
    chunk_overlap: int = 16,
    use_chunking: Optional[bool] = None,
    preview_mode: bool = False,
) -> vs.VideoNode:
    """Create a filter wrapper that processes with RVRT.

    Automatically chooses between full processing and chunked processing
    based on video length and available VRAM.

    Args:
        preview_mode: If True, process chunks on-demand for fast preview
                     (quality trade-off at chunk boundaries)
    """

    # Initialize inference to check VRAM
    device_obj = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    total_frames = src_clip.num_frames
    H, W = src_clip.height, src_clip.width

    # Check for preview mode first
    if preview_mode:
        print(f"RVRT: Using preview mode (lazy chunk processing)")
        return _create_filter_wrapper_preview(
            src_clip,
            config,
            tile_size,
            tile_overlap,
            sigma,
            use_fp16,
            device,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Determine if chunked processing should be used
    should_use_chunking = False
    if use_chunking is True:
        # Force chunked processing
        should_use_chunking = True
        print(f"RVRT: Using chunked processing (user requested)")
    elif use_chunking is False:
        # Force single-pass processing
        should_use_chunking = False
        print(f"RVRT: Using single-pass processing (user requested)")
    else:
        # Auto-detect based on video length and VRAM
        # Estimate memory needed for full processing
        # (1, T, 4, H, W) * 4 bytes * safety_factor
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(device_obj).total_memory
            allocated = torch.cuda.memory_allocated(device_obj)
            available = total_memory - allocated

            # Estimate: input + output + model + intermediate activations
            estimated_needed = total_frames * H * W * 4 * 4 * 20  # Conservative

            # Use chunked processing if video is very long or VRAM is limited
            if total_frames > 100 or estimated_needed > available * 0.85:
                should_use_chunking = True
                print(
                    f"RVRT: Using chunked processing (video too long for single-pass)"
                )

    if should_use_chunking:
        return _create_filter_wrapper_chunked(
            src_clip,
            config,
            tile_size,
            tile_overlap,
            sigma,
            use_fp16,
            device,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Process entire video as a single batch (original behavior)
    # Detect input format
    input_format = src_clip.format
    is_float_input = input_format.sample_type == vs.FLOAT
    output_format = vs.RGBS if is_float_input else vs.RGB24

    inference = RVRTInference(config, use_fp16=use_fp16, device=device_obj)

    print(f"RVRT: Loading {src_clip.num_frames} frames for processing...")

    frames = []
    for n in range(src_clip.num_frames):
        frame = src_clip.get_frame(n)
        tensor = _frame_to_tensor(frame)
        frames.append(tensor)

    video_tensor = torch.stack(frames, dim=1)
    print(f"RVRT: Input tensor shape: {video_tensor.shape}")

    sigma_normalized = sigma / 255.0 if sigma is not None else None
    print(f"RVRT: Processing with sigma={sigma} (normalized={sigma_normalized})")

    with torch.no_grad():
        output_tensor = inference.inference(
            video_tensor,
            sigma=sigma_normalized,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

    print(f"RVRT: Output tensor shape: {output_tensor.shape}")

    num_frames = output_tensor.shape[1]

    def process_frame(n, f):
        import ctypes

        frame_tensor = output_tensor[0, n, ...]

        img = frame_tensor.cpu().numpy()

        fout = f.copy()
        h, w = img.shape[1], img.shape[2]

        if is_float_input:
            # RGBS output: write float32 data directly
            for p in range(3):
                ptr = fout.get_write_ptr(p)
                stride = fout.get_stride(p)
                addr = ptr.value
                bytes_per_sample = 4  # float32
                row_size = stride // bytes_per_sample
                # Create float32 buffer
                dst_buf = (ctypes.c_float * (row_size * h)).from_address(addr)
                dst_arr = np.ctypeslib.as_array(dst_buf).reshape((h, row_size))
                plane_data = img[p]
                dst_arr[:, :w] = plane_data
        else:
            # RGB24 output: convert to uint8
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            for p in range(3):
                ptr = fout.get_write_ptr(p)
                stride = fout.get_stride(p)
                addr = ptr.value
                buf_size = stride * h
                dst_buf = (ctypes.c_uint8 * buf_size).from_address(addr)
                dst_arr = np.ctypeslib.as_array(dst_buf).reshape((h, stride))
                plane_data = img[p]
                dst_arr[:, :w] = plane_data

        return fout

    out_width = src_clip.width * config.upscale
    out_height = src_clip.height * config.upscale

    blank = vs.core.std.BlankClip(
        width=out_width,
        height=out_height,
        format=output_format,
        length=src_clip.num_frames,
        fpsnum=src_clip.fps.numerator,
        fpsden=src_clip.fps.denominator,
    )

    return blank.std.ModifyFrame(blank, process_frame)


def Denoise(
    clip: vs.VideoNode,
    sigma: float = 12.0,
    tile_size: Optional[Tuple[int, int, int]] = None,
    tile_overlap: Tuple[int, int, int] = (2, 20, 20),
    use_fp16: bool = True,
    device: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_chunking: Optional[bool] = None,
    preview_mode: bool = False,
) -> vs.VideoNode:
    """RVRT video denoising filter.

    Args:
        clip: Input video clip (must be RGB format)
        sigma: Noise level for denoising (0-50, default: 12)
        tile_size: Optional tuple of (T, H, W) for tiling. None for automatic.
        tile_overlap: Overlap for tiling as (T, H, W)
        use_fp16: Use FP16 precision for faster inference (default: True)
        device: Device to use ('cuda' or 'cpu'). None for auto.
        chunk_size: Number of frames per chunk for long videos. None for default (64).
        chunk_overlap: Overlapping frames between chunks. None for default (16).
        use_chunking: Force chunked processing on/off. None for auto-detection.
        preview_mode: If True, use lazy chunk processing for fast preview (default: False)
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("Denoise: clip must be a VideoNode")

    if clip.format.color_family != vs.RGB:
        raise vs.Error("Denoise: input must be RGB format")

    if not 0 <= sigma <= 50:
        raise vs.Error("Denoise: sigma must be between 0 and 50")

    if use_fp16 and not torch.cuda.is_available():
        raise vs.Error("Denoise: FP16 requires CUDA, but CUDA is not available")

    task = get_default_task("denoise")
    config = get_config(task)

    return _create_filter_wrapper(
        src_clip=clip,
        config=config,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        sigma=sigma,
        use_fp16=use_fp16,
        device=device,
        chunk_size=chunk_size if chunk_size is not None else 64,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else 16,
        use_chunking=use_chunking,
        preview_mode=preview_mode,
    )


def Deblur(
    clip: vs.VideoNode,
    model: str = "gopro",
    tile_size: Optional[Tuple[int, int, int]] = None,
    tile_overlap: Tuple[int, int, int] = (2, 20, 20),
    use_fp16: bool = True,
    device: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_chunking: Optional[bool] = None,
    preview_mode: bool = False,
) -> vs.VideoNode:
    """RVRT video deblurring filter.

    Args:
        clip: Input video clip (must be RGB format)
        model: Model to use ('gopro' or 'dvd')
        tile_size: Optional tuple of (T, H, W) for tiling. None for automatic.
        tile_overlap: Overlap for tiling as (T, H, W)
        use_fp16: Use FP16 precision for faster inference (default: True)
        device: Device to use ('cuda' or 'cpu'). None for auto.
        chunk_size: Number of frames per chunk for long videos. None for default (64).
        chunk_overlap: Overlapping frames between chunks. None for default (16).
        use_chunking: Force chunked processing on/off. None for auto-detection.
        preview_mode: If True, use lazy chunk processing for fast preview (default: False)
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("Deblur: clip must be a VideoNode")

    if clip.format.color_family != vs.RGB:
        raise vs.Error("Deblur: input must be RGB format")

    model_map = {
        "gopro": "005_RVRT_videodeblurring_GoPro_16frames",
        "dvd": "004_RVRT_videodeblurring_DVD_16frames",
    }

    if model.lower() not in model_map:
        raise vs.Error(f"Deblur: unknown model '{model}'")

    if use_fp16 and not torch.cuda.is_available():
        raise vs.Error("Deblur: FP16 requires CUDA, but CUDA is not available")

    config = get_config(model_map[model.lower()])

    return _create_filter_wrapper(
        src_clip=clip,
        config=config,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        sigma=None,
        use_fp16=use_fp16,
        device=device,
        chunk_size=chunk_size if chunk_size is not None else 64,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else 16,
        use_chunking=use_chunking,
        preview_mode=preview_mode,
    )


def SuperRes(
    clip: vs.VideoNode,
    scale: int = 4,
    model: str = "reds",
    tile_size: Optional[Tuple[int, int, int]] = None,
    tile_overlap: Tuple[int, int, int] = (2, 20, 20),
    use_fp16: bool = True,
    device: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_chunking: Optional[bool] = None,
    preview_mode: bool = False,
) -> vs.VideoNode:
    """RVRT video super-resolution filter.

    Args:
        clip: Input video clip (must be RGB format)
        scale: Upscaling factor (must be 4)
        model: Model to use ('reds', 'vimeo_bi', or 'vimeo_bd')
        tile_size: Optional tuple of (T, H, W) for tiling. None for automatic.
        tile_overlap: Overlap for tiling as (T, H, W)
        use_fp16: Use FP16 precision for faster inference (default: True)
        device: Device to use ('cuda' or 'cpu'). None for auto.
        chunk_size: Number of frames per chunk for long videos. None for default (64).
        chunk_overlap: Overlapping frames between chunks. None for default (16).
        use_chunking: Force chunked processing on/off. None for auto-detection.
        preview_mode: If True, use lazy chunk processing for fast preview (default: False)
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("SuperRes: clip must be a VideoNode")

    if clip.format.color_family != vs.RGB:
        raise vs.Error("SuperRes: input must be RGB format")

    if scale != 4:
        raise vs.Error("SuperRes: scale must be 4")

    model_map = {
        "reds": "001_RVRT_videosr_bi_REDS_30frames",
        "vimeo_bi": "002_RVRT_videosr_bi_Vimeo_14frames",
        "vimeo_bd": "003_RVRT_videosr_bd_Vimeo_14frames",
    }

    if model.lower() not in model_map:
        raise vs.Error(f"SuperRes: unknown model '{model}'")

    if use_fp16 and not torch.cuda.is_available():
        raise vs.Error("SuperRes: FP16 requires CUDA, but CUDA is not available")

    config = get_config(model_map[model.lower()])

    return _create_filter_wrapper(
        src_clip=clip,
        config=config,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        sigma=None,
        use_fp16=use_fp16,
        device=device,
        chunk_size=chunk_size if chunk_size is not None else 64,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else 16,
        use_chunking=use_chunking,
        preview_mode=preview_mode,
    )
