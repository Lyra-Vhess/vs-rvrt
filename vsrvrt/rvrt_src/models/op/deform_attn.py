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


def _find_cuda_home():
    """Find CUDA installation directory on various platforms.

    Searches for CUDA installation in common locations, prioritizing
    the version that matches PyTorch's compiled CUDA version.

    Returns:
        str or None: Path to CUDA installation directory, or None if not found
    """
    import sys
    import glob

    # Helper function to validate CUDA home by checking for nvcc
    def _validate_cuda_home(path):
        if not path or not os.path.exists(path):
            return False
        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        nvcc_path = os.path.join(path, "bin", nvcc_name)
        return os.path.exists(nvcc_path)

    # 1. Check environment variables first
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and _validate_cuda_home(cuda_home):
        return cuda_home

    # 2. Check torch's internal CUDA_HOME (but validate it!)
    try:
        from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home

        if torch_cuda_home and _validate_cuda_home(torch_cuda_home):
            return torch_cuda_home
    except (ImportError, AttributeError):
        pass

    # 3. Try to find nvcc in PATH and infer CUDA_HOME
    try:
        cmd = "where" if os.name == "nt" else "which"
        result = subprocess.run([cmd, "nvcc"], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc_path = result.stdout.strip().split("\n")[0].strip()
            # nvcc is typically in CUDA_HOME/bin/nvcc
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
            if _validate_cuda_home(cuda_home):
                return cuda_home
    except (subprocess.SubprocessError, OSError):
        pass

    # 4. Platform-specific searches
    pytorch_cuda_version = None
    if torch.cuda.is_available():
        pytorch_cuda_version = torch.version.cuda

    if sys.platform == "win32":
        # Windows: Search standard NVIDIA CUDA installation paths
        base_paths = [
            os.environ.get("ProgramFiles", "C:\\Program Files"),
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
        ]

        cuda_base_dirs = []
        for base in base_paths:
            nvidia_path = os.path.join(base, "NVIDIA GPU Computing Toolkit", "CUDA")
            if os.path.exists(nvidia_path):
                cuda_base_dirs.append(nvidia_path)

        # Find all CUDA version directories
        cuda_versions = []
        for cuda_base in cuda_base_dirs:
            for entry in os.listdir(cuda_base):
                version_dir = os.path.join(cuda_base, entry)
                nvcc_path = os.path.join(version_dir, "bin", "nvcc.exe")
                if os.path.isdir(version_dir) and os.path.exists(nvcc_path):
                    cuda_versions.append((entry, version_dir))

        # Sort by version (newest first)
        def version_key(item):
            try:
                return tuple(map(int, item[0].lstrip("v").split(".")))
            except ValueError:
                return (0, 0)

        cuda_versions.sort(key=version_key, reverse=True)

        # Prefer version matching PyTorch's CUDA version
        if pytorch_cuda_version and cuda_versions:
            for version_name, cuda_path in cuda_versions:
                # Check if version matches (e.g., "v12.1" matches "12.1")
                if pytorch_cuda_version in version_name:
                    return cuda_path

        # Return newest version if no match
        if cuda_versions:
            return cuda_versions[0][1]

    else:
        # Linux/macOS: Check common paths
        common_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/local/cuda-{}".format(pytorch_cuda_version)
            if pytorch_cuda_version
            else None,
        ]

        # Also check for versioned installations
        for base in ["/usr/local", "/opt"]:
            if os.path.exists(base):
                for entry in os.listdir(base):
                    if entry.startswith("cuda"):
                        common_paths.append(os.path.join(base, entry))

        for path in common_paths:
            if path and os.path.exists(path):
                nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
                nvcc_path = os.path.join(path, "bin", nvcc_name)
                if os.path.exists(nvcc_path):
                    return path

    return None


def _check_cuda_requirements():
    """Check if CUDA requirements are met for compiling the extension.

    Returns:
        tuple: (is_available: bool, error_message: str or None, cuda_home: str or None, ninja_path: str or None, ninja_dir: str or None)
    """
    if not torch.cuda.is_available():
        return (
            False,
            "CUDA is not available. This plugin requires CUDA.",
            None,
            None,
            None,
        )

    # Find CUDA installation
    cuda_home = _find_cuda_home()

    if not cuda_home:
        pytorch_cuda = torch.version.cuda if torch.cuda.is_available() else "unknown"
        return (
            False,
            (
                "CUDA installation not found.\n\n"
                "Searched locations:\n"
                "  - CUDA_HOME / CUDA_PATH environment variables\n"
                "  - PyTorch's internal CUDA_HOME\n"
                "  - nvcc in PATH\n"
                "  - Standard CUDA installation directories\n\n"
                f"PyTorch was built with CUDA {pytorch_cuda}.\n\n"
                "Please install CUDA toolkit or set CUDA_HOME:\n"
                "  Windows: set CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\n"
                "  Linux: export CUDA_HOME=/usr/local/cuda or /opt/cuda"
            ),
            None,
            None,
            None,
        )

    # Verify nvcc exists
    nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    nvcc_path = os.path.join(cuda_home, "bin", nvcc_name)
    if not os.path.exists(nvcc_path):
        return (
            False,
            (
                f"CUDA found at {cuda_home} but nvcc not found at {nvcc_path}.\n"
                f"Your CUDA installation may be incomplete."
            ),
            None,
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
            cuda_home,
            None,
            None,
        )

    return True, None, cuda_home, ninja_path, ninja_dir


def _load_deform_attn_extension():
    """Load or compile the deform_attn CUDA extension.

    The extension is compiled on first use and cached for subsequent runs.
    """
    is_available, error_msg, cuda_home, ninja_path, ninja_dir = (
        _check_cuda_requirements()
    )

    if not is_available:
        raise RuntimeError(
            f"Cannot load deform_attn extension: {error_msg}\n\n"
            "To use this plugin, you need:\n"
            "1. NVIDIA GPU with CUDA support\n"
            "2. CUDA toolkit installed\n"
            "3. ninja build system installed\n"
            "4. C++ compiler (gcc/clang / MSVC)\n\n"
            "Environment setup:\n"
            "  Windows:\n"
            "    set CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\n"
            "    set PATH=%CUDA_HOME%\\bin;%PATH%\n"
            "  Linux:\n"
            "    export CUDA_HOME=/usr/local/cuda\n"
            "    export PATH=$CUDA_HOME/bin:$PATH\n"
            "    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        )

    # At this point, cuda_home should always be set since is_available is True
    assert cuda_home is not None, (
        "cuda_home should not be None when is_available is True"
    )

    # Set CUDA_HOME if auto-detected and not already set
    if not os.environ.get("CUDA_HOME"):
        os.environ["CUDA_HOME"] = cuda_home

    # Add CUDA bin to PATH if not already there
    cuda_bin = os.path.join(cuda_home, "bin")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

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
            "1. Ensure CUDA toolkit is installed\n"
            "2. Install ninja: pip install ninja\n"
            "3. Install C++ compiler:\n"
            "   - Windows: Visual Studio Build Tools with C++ workload\n"
            "   - Ubuntu: sudo apt-get install build-essential\n"
            "   - Arch: sudo pacman -S base-devel\n"
            "4. Check that your CUDA version matches PyTorch's CUDA version\n\n"
            f"Detected CUDA_HOME: {cuda_home}\n"
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
