"""Download all RVRT model weights.

Usage:
    python -m vsrvrt.download_models
    vsrvrt-download
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

from .model_configs import TASK_CONFIGS

MODELS_DIR = Path(__file__).parent / "models"

MODEL_FILES = [
    ("001_RVRT_videosr_bi_REDS_30frames.pth", "001_RVRT_videosr_bi_REDS_30frames"),
    ("002_RVRT_videosr_bi_Vimeo_14frames.pth", "002_RVRT_videosr_bi_Vimeo_14frames"),
    ("003_RVRT_videosr_bd_Vimeo_14frames.pth", "003_RVRT_videosr_bd_Vimeo_14frames"),
    (
        "004_RVRT_videodeblurring_DVD_16frames.pth",
        "004_RVRT_videodeblurring_DVD_16frames",
    ),
    (
        "005_RVRT_videodeblurring_GoPro_16frames.pth",
        "005_RVRT_videodeblurring_GoPro_16frames",
    ),
    (
        "006_RVRT_videodenoising_DAVIS_16frames.pth",
        "006_RVRT_videodenoising_DAVIS_16frames",
    ),
]


def download_model(filepath: Path, url: str) -> bool:
    """Download a model file with progress bar.

    Args:
        filepath: Destination path for the model file
        url: URL to download from

    Returns:
        True on success, False on failure
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                desc=filepath.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        return True

    except requests.RequestException as e:
        print(f"  ERROR: Failed to download {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False
    except Exception as e:
        print(f"  ERROR: Unexpected error downloading {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def main():
    """Download all RVRT model weights.

    Exits with code 0 on success, 1 on any failure.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading RVRT model weights...")
    print(f"Target directory: {MODELS_DIR}")
    print()

    failed = []

    for filename, config_key in MODEL_FILES:
        filepath = MODELS_DIR / filename
        url = TASK_CONFIGS[config_key].model_url

        if filepath.exists():
            print(f"[SKIP] {filename} (already exists)")
            continue

        print(f"[DOWNLOAD] {filename}")
        if not download_model(filepath, url):
            failed.append(filename)

    print()

    if failed:
        print(f"FAILED: Could not download: {', '.join(failed)}")
        print("Models will be downloaded on first use.")
        sys.exit(1)
    else:
        print("SUCCESS: All model weights downloaded.")
        sys.exit(0)


if __name__ == "__main__":
    main()
