#!/usr/bin/env python
"""Test vsrvrt by piping frames directly to ffmpeg, bypassing vspipe.

This script allows testing vsrvrt in any Python venv without relying on
the system vspipe binary, which is compiled against a specific Python version.

Usage:
    python run_test.py [output_file]

Example:
    source .venv312/bin/activate && cd test && python run_test.py denoise_output_312.mkv
"""

import subprocess
import sys
import os

os.environ.setdefault("VAPOURSYNTH_PLUGIN_PATH", "/usr/lib/vapoursynth")

import vapoursynth as vs
from vapoursynth import core
import vsrvrt

source = "denoise_input.mkv"

clip = core.ffms2.Source(source=source)

clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709")

denoised = vsrvrt.Denoise(
    clip, sigma=12, tile_size=(64, 256, 256), tile_overlap=(2, 20, 20)
)

denoised = denoised.resize.Bicubic(
    format=vs.YUV420P10, matrix_s="709", dither_type="error_diffusion"
)

output_file = sys.argv[1] if len(sys.argv) > 1 else "denoise_output.mkv"
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-i",
    "-",
    "-c:v",
    "libx264",
    "-preset",
    "placebo",
    "-qp",
    "0",
    output_file,
]

print(f"Processing with Python {sys.version_info.major}.{sys.version_info.minor}")
print(f"vsrvrt version: {vsrvrt.__version__}")
print(f"Output: {output_file}")

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
denoised.output(proc.stdin, y4m=True)
proc.stdin.close()
proc.wait()

if proc.returncode == 0:
    print(f"Success! Output saved to {output_file}")
else:
    print(f"Error: ffmpeg exited with code {proc.returncode}")

sys.exit(proc.returncode)
