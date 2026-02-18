# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
import os
import subprocess
import warnings
import torch
from torch import nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from packaging.version import Version
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)

# Workaround for glog header issues in some PyTorch builds
# Note: GLOG_DEPRECATED must be defined but empty to avoid syntax errors
_extra_cflags = ["-DGLOG_EXPORT=", "-DGLOG_NO_EXPORT=", "-DGLOG_DEPRECATED="]
_extra_cuda_cflags = ["-DGLOG_EXPORT=", "-DGLOG_NO_EXPORT=", "-DGLOG_DEPRECATED="]


def _find_ninja():
    """Find ninja executable in common locations.

    On Windows portable Python setups, ninja may be installed in the Scripts
    folder but not in PATH. This function searches common locations.

    Returns:
        tuple: (ninja_path: str or None, ninja_dir: str or None)
               ninja_path is full path to executable, ninja_dir is directory containing it
    """
    import sys

    # Check NINJA_PATH environment variable first
    env_ninja = os.environ.get("NINJA_PATH")
    if env_ninja and os.path.isfile(env_ninja):
        return env_ninja, os.path.dirname(env_ninja)

    # First check if ninja is in PATH
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["ninja", "--version"], capture_output=True, text=True, shell=False
            )
        else:
            result = subprocess.run(
                ["ninja", "--version"], capture_output=True, text=True
            )
        if result.returncode == 0:
            return "ninja", None  # In PATH, use as-is
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    # On Windows, search in common portable Python locations
    if sys.platform == "win32":
        import sys

        python_exe = sys.executable
        python_dir = os.path.dirname(python_exe)

        possible_paths = [
            # User-specified NINJA_PATH
            os.environ.get("NINJA_PATH"),
            # Python Scripts folder (most common pip install location)
            os.path.join(python_dir, "Scripts", "ninja.exe"),
            # Same directory as python.exe
            os.path.join(python_dir, "ninja.exe"),
            # Parent directory Scripts (VapourSynth portable structure)
            os.path.join(os.path.dirname(python_dir), "Scripts", "ninja.exe"),
            os.path.join(os.path.dirname(python_dir), "ninja.exe"),
            # Site-packages ninja package
            os.path.join(
                python_dir, "Lib", "site-packages", "ninja", "data", "bin", "ninja.exe"
            ),
        ]

        # Also check all paths in sys.path for site-packages
        for site_path in sys.path:
            if "site-packages" in site_path:
                possible_paths.extend(
                    [
                        os.path.join(site_path, "ninja", "data", "bin", "ninja.exe"),
                        os.path.join(
                            os.path.dirname(site_path), "Scripts", "ninja.exe"
                        ),
                    ]
                )

        for ninja_path in possible_paths:
            if ninja_path and os.path.isfile(ninja_path):
                return ninja_path, os.path.dirname(ninja_path)

    return None, None


def _check_cuda_requirements():
    """Check if CUDA requirements are met for compiling the extension.

    Returns:
        tuple: (is_available: bool, error_message: str or None, ninja_path: str or None, ninja_dir: str or None)
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available. This plugin requires CUDA.", None, None

    # Check for CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        # Try to infer from torch
        try:
            from torch.utils.cpp_extension import CUDA_HOME

            cuda_home = CUDA_HOME
        except:
            pass

    if not cuda_home or not os.path.exists(cuda_home):
        return (
            False,
            (
                "CUDA_HOME not found. Please set CUDA_HOME environment variable to your CUDA installation path.\n"
                "Example: export CUDA_HOME=/usr/local/cuda or export CUDA_HOME=/opt/cuda"
            ),
            None,
            None,
        )

    # Check for nvcc
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.exists(nvcc_path):
        # Try to find nvcc in PATH (Windows uses 'where', Linux/Mac use 'which')
        try:
            cmd = "where" if os.name == "nt" else "which"
            result = subprocess.run([cmd, "nvcc"], capture_output=True, text=True)
            if result.returncode != 0:
                return (
                    False,
                    (
                        f"nvcc not found at {nvcc_path} and not in PATH.\n"
                        f"Please ensure CUDA toolkit is properly installed."
                    ),
                    None,
                    None,
                )
        except:
            return (
                False,
                f"nvcc not found. Please ensure CUDA toolkit is properly installed.",
                None,
                None,
            )

    # Check for ninja
    ninja_path, ninja_dir = _find_ninja()
    if ninja_path is None:
        return (
            False,
            (
                "ninja build system not found. Please install ninja.\n"
                "On Ubuntu/Debian: sudo apt-get install ninja-build\n"
                "On Arch: sudo pacman -S ninja\n"
                "On Windows: pip install ninja\n"
                "\n"
                "If you already installed ninja but get this error:\n"
                "  Windows portable Python: Add the Scripts folder to your PATH:\n"
                "    set PATH=%%PATH%%;C:\\path\\to\\python\\Scripts\n"
                "  Or set NINJA_PATH environment variable:\n"
                "    set NINJA_PATH=C:\\path\\to\\ninja.exe"
            ),
            None,
            None,
        )

    return True, None, ninja_path, ninja_dir


def _load_deform_attn_extension():
    """Load or compile the deform_attn CUDA extension.

    The extension is compiled on first use and cached for subsequent runs.
    """
    is_available, error_msg, ninja_path, ninja_dir = _check_cuda_requirements()

    if not is_available:
        raise RuntimeError(
            f"Cannot load deform_attn extension: {error_msg}\n\n"
            "To use this plugin, you need:\n"
            "1. NVIDIA GPU with CUDA support\n"
            "2. CUDA toolkit installed\n"
            "3. ninja build system installed\n"
            "4. C++ compiler (gcc/clang)\n\n"
            "Environment setup:\n"
            "  export CUDA_HOME=/path/to/cuda  # e.g., /usr/local/cuda or /opt/cuda\n"
            '  export PATH="$CUDA_HOME/bin:$PATH"\n'
            '  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"'
        )

    # Add ninja to PATH if found outside of PATH (e.g., Windows portable Python)
    if ninja_dir and ninja_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ninja_dir + os.pathsep + os.environ.get("PATH", "")

    try:
        ext = load(
            "deform_attn",
            sources=[
                os.path.join(module_path, "deform_attn_ext.cpp"),
                os.path.join(
                    module_path,
                    "deform_attn_cuda_pt110.cpp"
                    if Version(torch.__version__) >= Version("1.10.0")
                    else "deform_attn_cuda_pt109.cpp",
                ),
                os.path.join(module_path, "deform_attn_cuda_kernel.cu"),
            ],
            extra_cflags=_extra_cflags,
            extra_cuda_cflags=_extra_cuda_cflags,
            verbose=False,
        )
        return ext
    except Exception as e:
        raise RuntimeError(
            f"Failed to compile deform_attn extension: {e}\n\n"
            "Common solutions:\n"
            "1. Ensure CUDA_HOME is set correctly\n"
            "2. Install ninja: pip install ninja\n"
            "3. Install C++ compiler: sudo apt-get install build-essential (Ubuntu) or sudo pacman -S base-devel (Arch)\n"
            "4. Check that your CUDA version matches PyTorch's CUDA version\n\n"
            f"Current CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}\n"
            f"PyTorch CUDA version: {torch.version.cuda}"
        ) from e


deform_attn_ext = _load_deform_attn_extension()


class Mlp(nn.Module):
    """Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DeformAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        offset,
        kernel_h,
        kernel_w,
        stride=1,
        padding=0,
        dilation=1,
        attention_heads=1,
        deformable_groups=1,
        clip_size=1,
    ):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.attention_heads = attention_heads
        ctx.deformable_groups = deformable_groups
        ctx.clip_size = clip_size
        if q.requires_grad or kv.requires_grad or offset.requires_grad:
            ctx.save_for_backward(q, kv, offset)
        output = q.new_empty(q.shape)
        ctx._bufs = [
            q.new_empty(0),
            q.new_empty(0),
            q.new_empty(0),
            q.new_empty(0),
            q.new_empty(0),
        ]
        deform_attn_ext.deform_attn_forward(
            q,
            kv,
            offset,
            output,
            ctx._bufs[0],
            ctx._bufs[1],
            ctx._bufs[2],
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.attention_heads,
            ctx.deformable_groups,
            ctx.clip_size,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        q, kv, offset = ctx.saved_tensors
        grad_q = torch.zeros_like(q)
        grad_kv = torch.zeros_like(kv)
        grad_offset = torch.zeros_like(offset)
        deform_attn_ext.deform_attn_backward(
            q,
            kv,
            offset,
            ctx._bufs[0],
            ctx._bufs[1],
            ctx._bufs[2],
            ctx._bufs[3],
            ctx._bufs[4],
            grad_q,
            grad_kv,
            grad_offset,
            grad_output,
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.attention_heads,
            ctx.deformable_groups,
            ctx.clip_size,
        )

        return (
            grad_q,
            grad_kv,
            grad_offset,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


deform_attn = DeformAttnFunction.apply


class DeformAttn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        attention_window=[3, 3],
        deformable_groups=12,
        attention_heads=12,
        clip_size=1,
    ):
        super(DeformAttn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = attention_window[0]
        self.kernel_w = attention_window[1]
        self.attn_size = self.kernel_h * self.kernel_w
        self.deformable_groups = deformable_groups
        self.attention_heads = attention_heads
        self.clip_size = clip_size
        self.stride = 1
        self.padding = self.kernel_h // 2
        self.dilation = 1

        self.proj_q = nn.Sequential(
            Rearrange("n d c h w -> n d h w c"),
            nn.Linear(self.in_channels, self.in_channels),
            Rearrange("n d h w c -> n d c h w"),
        )
        self.proj_k = nn.Sequential(
            Rearrange("n d c h w -> n d h w c"),
            nn.Linear(self.in_channels, self.in_channels),
            Rearrange("n d h w c -> n d c h w"),
        )
        self.proj_v = nn.Sequential(
            Rearrange("n d c h w -> n d h w c"),
            nn.Linear(self.in_channels, self.in_channels),
            Rearrange("n d h w c -> n d c h w"),
        )
        self.mlp = nn.Sequential(
            Rearrange("n d c h w -> n d h w c"),
            Mlp(self.in_channels, self.in_channels * 2),
            Rearrange("n d h w c -> n d c h w"),
        )

    def forward(self, q, k, v, offset):
        q = self.proj_q(q)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(
            q,
            kv,
            offset,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding,
            self.dilation,
            self.attention_heads,
            self.deformable_groups,
            self.clip_size,
        )
        v = v + self.mlp(v)
        return v


class DeformAttnPack(DeformAttn):
    """A Deformable Attention Encapsulation that acts as normal attention layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
    """

    def __init__(self, *args, **kwargs):
        super(DeformAttnPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels * (1 + self.clip_size),
            self.clip_size * self.deformable_groups * self.attn_size * 2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            bias=True,
        )
        self.init_weight()

    def init_weight(self):
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, q, k, v):
        out = self.conv_offset(torch.cat([q.flatten(1, 2), k.flatten(1, 2)], 1))
        o1, o2 = torch.chunk(out, 2, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        q = self.proj_q(q)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(
            q,
            kv,
            offset,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding,
            self.dilation,
            self.attention_heads,
            self.deformable_groups,
            self.clip_size,
        )
        v = v + self.mlp(v)
        return v
