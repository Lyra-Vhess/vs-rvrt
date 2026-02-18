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

## Requirements

- Python >= 3.8
- Vapoursynth >= 60
- PyTorch >= 1.9.1
- CUDA-capable GPU

## Installation

### Arch Linux

```bash
# Clone the repository
git clone https://github.com/Lyra-Vhess/vs-rvrt/
cd vs-rvrt

# Download models from releases
mkdir vsrvrt/models && cd vsrvrt/models
wget https://github.com/Lyra-Vhess/vs-rvrt/releases/download/v1.0.0/001_RVRT_videosr_bi_REDS_30frames.pth \
wget https://github.com/Lyra-Vhess/vs-rvrt/releases/download/v1.0.0/002_RVRT_videosr_bi_Vimeo_14frames.pth \
wget https://github.com/Lyra-Vhess/vs-rvrt/releases/download/v1.0.0/003_RVRT_videosr_bd_Vimeo_14frames.pth \
wget https://github.com/Lyra-Vhess/vs-rvrt/releases/download/v1.0.0/004_RVRT_videodeblurring_DVD_16frames.pth \
wget https://github.com/Lyra-Vhess/vs-rvrt/releases/download/v1.0.0/005_RVRT_videodeblurring_GoPro_16frames.pth \
wget https://github.com/Lyra-Vhess/vs-rvrt/releases/download/v1.0.0/006_RVRT_videodenoising_DAVIS_16frames.pth

# Create PKGBUILD directory
cd ../..
mkdir ../vsrvrt-pkgbuild
mv PKGBUILD ../vsrvrt-pkgbuild/

# Build and install
cd ../vsrvrt-pkgbuild
makepkg -si
```

### pip
```bash
# Clone the repository
git clone https://github.com/Lyra-Vhess/vs-rvrt/
cd vs-rvrt

# Install
pip install -e .
```

### Environment Setup

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# May be required for CUDA extension compilation
export CC=gcc
export CXX=g++
export CUDA_HOME=/opt/cuda

# Prevent stale cache locks
rm -f ~/.cache/torch_extensions/*/deform_attn/lock 2>/dev/null
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

### Import Hangs / Stuck in Terminal

If importing vsrvrt hangs, the CUDA extension may be compiling or a stale cache lock exists:

```bash
# Clear stale lock files
rm -f ~/.cache/torch_extensions/*/deform_attn/lock
```

### CUDA Compilation Errors

If you see errors about `aocc-clang` or compiler issues:

```bash
# Ensure GCC is used, not AOCC
export CC=gcc
export CXX=g++
export CUDA_HOME=/opt/cuda
```

## Performance Tips

1. **Use FP16**: Enable for 2x speedup and 50% VRAM reduction
3. **Preview Mode**: Use `preview_mode=True` in vspreview for instant feedback
4. **Adjust Tiling**: If you get OOM errors, reduce `tile_size`
6. **Chunk Size**: Larger chunks = better quality but more VRAM. Default 64 is a good balance.

## Project Structure

```
vsrvrt/
├── __init__.py             # Package initialization
├── model_configs.py        # Model configurations
├── rvrt_core.py            # Core inference wrapper
├── rvrt_filter.py          # Vapoursynth filter functions
├── models/                 # Downloaded model weights (auto-populated)
├── utils/                  # Utility functions
└── rvrt_src/               # RVRT source code
    ├── models/
    │   ├── network_rvrt.py
    │   └── op/             # CUDA extensions
    └── utils/
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
