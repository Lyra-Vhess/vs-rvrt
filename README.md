# VSRVRT: Vapoursynth Plugin for RVRT

A Vapoursynth plugin wrapper for RVRT (Recurrent Video Restoration Transformer), implementing state-of-the-art video denoising, deblurring, and super-resolution. Based on https://github.com/JingyunLiang/RVRT

## Features

- **Video Denoising**: Non-blind denoising with tunable sigma parameter (0-50)
- **Video Deblurring**: Support for both GoPro and DVD dataset models
- **Video Super-Resolution**: 4x upscaling with multiple model variants (Extreme VRAM usage)
- **FP16 Support**: Half-precision for faster inference and 50% VRAM reduction, optional FP32
- **Preview Mode**: Lazy chunk processing for faster preview in vspreview (still slow though)
- **Flexible Chunking**: Control chunk size, overlap, and processing strategy
- **Automatic Tiling**: VRAM-aware automatic tiling to handle large videos (may not be perfect)
- **RGBS Format**: Full support for 32-bit float RGB
- **Pre-built Binaries**: No compilation required for PyPI installs

## Requirements

- Python 3.12 or 3.13
- VapourSynth >= 60
- PyTorch >= 2.10.0 **with CUDA support** (see important note below)
- NVIDIA GPU with CUDA 12.8+ driver

## Installation

### pip

```bash
pip install vsrvrt
```

### Arch Linux (AUR)

```bash
yay -S vsrvrt-git
```

**Important Note:** PyPI's default index only has CPU-only PyTorch. You must install the CUDA version first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Building from Source

If you need to build from source (e.g., for development or unsupported platforms):

1. **Prerequisites:**
   - CUDA Toolkit 12.8+
   - C++ compiler (MSVC on Windows, GCC on Linux)
   - ninja build system

2. **Build:**
   ```bash
   git clone https://github.com/Lyra-Vhess/vs-rvrt/
   cd vs-rvrt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install ninja
   python build_wheels.py
   pip install dist/vsrvrt-*.whl
   ```

## Usage

### Basic Usage

```python
# Convert to RGB (required)
clip = clip.resize.Bicubic(format=vs.RGB24) # or RGBS

# Video Denoising (sigma: 0-50)
denoised = vsrvrt.Denoise(clip, sigma=12.0)

# Video Deblurring
deblurred = vsrvrt.Deblur(clip, model="gopro")  # or "dvd"

# Video Super-Resolution (4x)
upscaled = vsrvrt.SuperRes(clip, scale=4, model="reds")  # or "vimeo_bi", "vimeo_bd"
```

## API Reference

### Denoise

```python
vsrvrt.Denoise(
    clip: vs.VideoNode,             # Input clip (RGB format)
    sigma: float = 12.0,            # Noise level (0-50, default: 12)
    tile_size: Tuple[int, int, int] = (64,256,256),    # (Temporal, Height, Width), None for auto
    tile_overlap: Tuple[int, int, int] = (2, 20, 20),  # Overlap for tiling
    use_fp16: [bool] = True,        # Use FP16 precision, (default: True)
    device: [str] = None,           # 'cuda', 'cpu', or auto
    chunk_size: [int] = None,       # Frames per chunk (default: 64)
    chunk_overlap: [int] = None,    # Overlapping frames (default: 16)
    use_chunking: [bool] = None,    # Whether to us chunked processing, (default: True)
    preview_mode: [bool] = False    # Lazy chunk processing for preview
) -> vs.VideoNode
```

### Deblur

```python
vsrvrt.Deblur(
    clip: vs.VideoNode,             # Input clip (RGB format)
    model: str = "gopro",           # 'gopro' or 'dvd'
    tile_size: Tuple[int, int, int] = None,
    tile_overlap: Tuple[int, int, int] = (2, 20, 20),
    use_fp16: [bool] = True,
    device: [str] = None,
    chunk_size: [int] = None,
    chunk_overlap: [int] = None,
    use_chunking: [bool] = None,
    preview_mode: [bool] = False
) -> vs.VideoNode
```

### SuperRes

```python
vsrvrt.SuperRes(
    clip: vs.VideoNode,             # Input clip (RGB format)
    scale: int = 4,                 # Must be 4 (only 4x models available)
    model: str = "reds",            # 'reds', 'vimeo_bi', or 'vimeo_bd'
    tile_size: Optional[Tuple[int, int, int]] = None,
    tile_overlap: Tuple[int, int, int] = (2, 20, 20),
    use_fp16: bool = True,
    device: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_chunking: Optional[bool] = None,
    preview_mode: bool = False
) -> vs.VideoNode
```

## Tiling Options

The `tile_size` parameter controls memory usage:

- `None`: Automatic based on available VRAM
- `(0, 0, 0)`: No tiling (process entire video at once) - requires significant VRAM
- `(T, H, W)`: Manual tile size (e.g., `(64, 256, 256)`)

**Spatial Tile Size Guidelines:**
- As far as I can tell, HxW = 256x256 is ideal because the models were trained with that tile size. Experiment at your own risk.
- Higher temporal windows appear to improve quality, though there appears to be a ceiling after which you will get artifacts.
- Temporal windows from 16 to 64 seem safe
- The interaction between the temporal tiling and chunk length is not entirely clear to me, experiment at your own risk
- Automatic sizing may not be perfect, consider `(16,256,256)` as a good default and increase T as memory allows
- Super-Resolution is extremely memory-hungry and likely requires bare minimal tiling for 12GB GPUs of `(8,256,256)` with chunk sizes of 16
- Expect poor SR quality below 16GB of VRAM due to the need for extremely minimal tiling sizes

**Note:** Tile size must be a multiple of 8 (the model's window size).

### Preview Mode (for vspreview)

Preview mode processes chunks on-demand for instant startup:

```python
# Preview mode - faster startup, processes chunks as needed
denoised = vsrvrt.Denoise(clip, sigma=10.0, preview_mode=True)

# Normal mode - process all chunks upfront, best quality but extreme delay
denoised = vsrvrt.Denoise(clip, sigma=10.0, preview_mode=False)
```

**Preview Mode Notes:**
- Each chunk is processed independently (no recurrence from previous chunks)
- Slight quality trade-off at chunk boundaries
- Processed chunks are cached for the session
- Best for quickly checking settings before final encode

### Chunk Control

Control how video is processed in chunks:

```python
# Customize chunk processing
denoised = vsrvrt.Denoise(
    clip, 
    sigma=10.0,
    chunk_size=64,        # Frames per chunk (default: 64)
    chunk_overlap=16,     # Overlapping frames (default: 16)
    use_chunking=True     # Use chunked processing
)

# Disable chunking (process entire video at once, extreme memory usage)
denoised = vsrvrt.Denoise(clip, sigma=10.0, use_chunking=False)
```

**Chunk Size Guidelines:**
- **64 frames**: Good balance (default)
- **48-96**: Adjust based on VRAM
- Must be <128 to keep processing on GPU

### Model Info

These are the datasets used in training the models, choose the model based on how closely your video matches the dataset.

| Dataset    | Task            | Resolution    | Content Type                | Best For                               |
|------------|-----------------|---------------|-----------------------------|----------------------------------------|
| REDS       | Super-Resolution| 1280×720      | Real diverse scenes         | General upscaling, natural motion      |
| Vimeo-90K  | Super-Resolution| 448×256       | Real web videos             | Web content, user videos                |
| GoPro      | Deblurring      | 1280×720      | Synthetic blur from real scenes | Camera shake, dynamic motion           |
| DVD        | Deblurring      | ~1280×720     | Hand-held camera blur        | Smartphone videos, hand shake           |
| DAVIS      | Denoising       | 1080p/480p    | High-quality footage         | Tunable denoising (sigma 0-50)         |

## Troubleshooting

### "requires PyTorch with CUDA support, but you have the CPU-only version"

This means pip installed the CPU-only PyTorch from the default PyPI index. Fix:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### "CUDA is not available" (but you have PyTorch with CUDA)

This usually means:
1. You don't have an NVIDIA GPU
2. Your GPU driver is too old
3. CUDA drivers aren't installed

Update your NVIDIA drivers from https://www.nvidia.com/Download/index.aspx

## Performance Tips

1. **Use FP16**: Enable for 2x speedup and 50% VRAM reduction
2. **Preview Mode**: Use `preview_mode=True` in vspreview for instant feedback
3. **Adjust Tiling**: If you get OOM errors, reduce `tile_size`
4. **Chunk Size**: Larger chunks = better quality but more VRAM. Default 64 is a good balance.

## Project Structure

```
vsrvrt/
├── __init__.py             # Package initialization
├── download_models.py      # CLI: vsrvrt-download
├── model_configs.py        # Model configurations
├── rvrt_core.py            # Core inference wrapper
├── rvrt_filter.py          # VapourSynth filter functions
├── _binary/                # Pre-built CUDA extension binaries
│   ├── __init__.py         # Platform/version loader
│   ├── manylinux_x86_64/   # Linux (cp312, cp313, cp314)
│   └── win_amd64/          # Windows (cp312, cp313)
├── models/                 # Placeholder (models cached in ~/.cache/vsrvrt/)
├── utils/                  # Utility functions
└── rvrt_src/               # RVRT source code
    ├── models/
    │   ├── network_rvrt.py
    │   └── op/             # CUDA extension source
    └── utils/
        ├── utils_image.py
        └── utils_video.py
```

## Citation

```bibtex
@article{liang2022rvrt,
    title={Recurrent Video Restoration Transformer with Guided Deformable Attention},
    author={Liang, Jingyun and Fan, Yuchen and Xiang, Xiaoyu and Ranjan, Rakesh and Ilg, Eddy  and Green, Simon and Cao, Jiezhang and Zhang, Kai and Timofte, Radu and Van Gool, Luc},
    journal={arXiv preprint arXiv:2206.02146},
    year={2022}
}
```

## License

This plugin follows the same license as RVRT (CC-BY-NC-4.0 for non-commercial use).
