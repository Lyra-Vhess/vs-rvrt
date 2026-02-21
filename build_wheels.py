#!/usr/bin/env python
"""Build script for vsrvrt pre-compiled wheels.

This script compiles the deform_attn CUDA extension and builds a wheel
for the current Python version. Run separately with each Python version
to build all wheels.

Requirements:
    - Python 3.12, 3.13, or 3.14
    - PyTorch 2.10+ with CUDA 12.8
    - CUDA Toolkit 12.8+ (for compilation)
    - C++ compiler (MSVC on Windows, GCC on Linux)
    - ninja build system

Usage:
    Windows:
        py -3.12 build_wheels.py    # Build wheel for Python 3.12
        py -3.13 build_wheels.py    # Build wheel for Python 3.13

    Linux:
        python3.12 build_wheels.py    # Build wheel for Python 3.12
        python3.13 build_wheels.py    # Build wheel for Python 3.13

Output:
    dist/vsrvrt-1.1.0-cp312-none-win_amd64.whl
    dist/vsrvrt-1.1.0-cp313-none-win_amd64.whl
    dist/vsrvrt-1.1.0-cp314-none-win_amd64.whl
    dist/vsrvrt-1.1.0-cp312-none-manylinux_x86_64.whl
    dist/vsrvrt-1.1.0-cp313-none-manylinux_x86_64.whl
    dist/vsrvrt-1.1.0-cp314-none-manylinux_x86_64.whl

Why separate wheels?
    - Simpler maintenance: adding a new Python version requires only one build
    - No need to rebuild old versions when adding new ones
    - Each wheel is standalone and independent
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()
VSRVRT_DIR = PROJECT_ROOT / "vsrvrt"
BINARY_DIR = VSRVRT_DIR / "_binary"
DIST_DIR = PROJECT_ROOT / "dist"
STAGING_DIR = PROJECT_ROOT / "staging"

OP_DIR = VSRVRT_DIR / "rvrt_src" / "models" / "op"
SOURCES = [
    OP_DIR / "deform_attn_ext.cpp",
    OP_DIR / "deform_attn_cuda_pt110.cpp",
    OP_DIR / "deform_attn_cuda_kernel.cu",
]

EXTRA_CFLAGS = ["-DGLOG_EXPORT=", "-DGLOG_NO_EXPORT=", "-DGLOG_DEPRECATED="]
EXTRA_CUDA_CFLAGS = ["-DGLOG_EXPORT=", "-DGLOG_NO_EXPORT=", "-DGLOG_DEPRECATED="]

SUPPORTED_PYTHON_VERSIONS = ["cp312", "cp313", "cp314"]


def get_platform_id():
    """Get the platform identifier for internal directory naming."""
    if sys.platform == "win32":
        return "win_amd64"
    elif sys.platform.startswith("linux"):
        return "manylinux_x86_64"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def get_wheel_platform_tag():
    """Get the platform tag for wheel naming (PyPI-compatible)."""
    if sys.platform == "win32":
        return "win-amd64"
    elif sys.platform.startswith("linux"):
        return "manylinux2014_x86_64"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def get_python_version_id():
    """Get the Python version identifier (e.g., 'cp312')."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def check_requirements():
    """Check that all build requirements are met."""
    print("Checking build requirements...")

    py_version = get_python_version_id()
    if py_version not in SUPPORTED_PYTHON_VERSIONS:
        print(
            f"ERROR: Python {sys.version_info.major}.{sys.version_info.minor} is not supported."
        )
        print(f"Supported versions: {', '.join(SUPPORTED_PYTHON_VERSIONS)}")
        sys.exit(1)
    print(f"  Python: {py_version} [OK]")

    try:
        import torch

        print(f"  PyTorch: {torch.__version__} [OK]")

        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available in PyTorch.")
            print(
                "Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
            )
            sys.exit(1)
        print(f"  CUDA: {torch.version.cuda} [OK]")

        from packaging.version import Version

        if Version(torch.__version__) < Version("2.10.0"):
            print(f"WARNING: PyTorch {torch.__version__} is older than 2.10.0.")
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        print(
            "Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        )
        sys.exit(1)

    try:
        result = subprocess.run(["ninja", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ninja: {result.stdout.strip()} [OK]")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.SubprocessError):
        print("ERROR: ninja is not installed.")
        print("Install: pip install ninja")
        sys.exit(1)

    if sys.platform == "win32":
        try:
            result = subprocess.run(["where", "cl"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  MSVC: Found [OK]")
            else:
                raise FileNotFoundError
        except (FileNotFoundError, subprocess.SubprocessError):
            print("ERROR: MSVC compiler (cl.exe) not found.")
            print("Install Visual Studio Build Tools with C++ workload.")
            sys.exit(1)
    else:
        try:
            result = subprocess.run(
                ["gcc", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  GCC: {result.stdout.split(chr(10))[0]} [OK]")
            else:
                raise FileNotFoundError
        except (FileNotFoundError, subprocess.SubprocessError):
            print("ERROR: GCC compiler not found.")
            print(
                "Install: sudo apt-get install build-essential (Ubuntu) or sudo pacman -S base-devel (Arch)"
            )
            sys.exit(1)

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        nvcc_name = "nvcc.exe" if sys.platform == "win32" else "nvcc"
        nvcc_path = Path(cuda_home) / "bin" / nvcc_name
        if nvcc_path.exists():
            print(f"  CUDA Toolkit: {cuda_home} [OK]")
    print()


def compile_extension():
    """Compile the deform_attn extension using PyTorch's cpp_extension."""
    print("Compiling deform_attn extension...")

    import torch
    from torch.utils.cpp_extension import load

    py_version = get_python_version_id()
    build_dir = PROJECT_ROOT / "build" / f"torch_ext_{py_version}"
    build_dir.mkdir(parents=True, exist_ok=True)

    ext = load(
        "deform_attn",
        sources=[str(s) for s in SOURCES],
        extra_cflags=EXTRA_CFLAGS,
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        build_directory=str(build_dir),
        verbose=True,
    )

    ext_file = (
        build_dir / "deform_attn.pyd"
        if sys.platform == "win32"
        else build_dir / f"deform_attn.cpython-{py_version}-x86_64-linux-gnu.so"
    )

    if not ext_file.exists():
        candidates = (
            list(build_dir.glob("*.pyd"))
            if sys.platform == "win32"
            else list(build_dir.glob("*.so"))
        )
        if candidates:
            ext_file = candidates[0]
        else:
            print("ERROR: Could not find compiled extension.")
            print(f"Searched in: {build_dir}")
            sys.exit(1)

    print(f"  Compiled: {ext_file}")
    return ext_file


def copy_binary_to_source(ext_file, platform_id, py_version):
    """Copy the compiled binary to the source tree _binary directory."""
    dest_dir = BINARY_DIR / platform_id / py_version
    dest_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        dest_file = dest_dir / "deform_attn.pyd"
    else:
        dest_file = dest_dir / f"deform_attn.cpython-{py_version}-x86_64-linux-gnu.so"

    shutil.copy2(ext_file, dest_file)
    print(f"  Copied to source: {dest_file}")
    return dest_file


def create_staging_directory(platform_id, py_version):
    """Create a staging directory with only the current version's binary."""
    print("Creating staging directory...")

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    staging_vsrvrt = STAGING_DIR / "vsrvrt"

    for item in VSRVRT_DIR.iterdir():
        if item.name == "_binary":
            continue
        dest = staging_vsrvrt / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

    staging_binary = staging_vsrvrt / "_binary" / platform_id / py_version
    staging_binary.mkdir(parents=True, exist_ok=True)

    src_binary_dir = BINARY_DIR / platform_id / py_version
    if src_binary_dir.exists():
        for item in src_binary_dir.iterdir():
            shutil.copy2(item, staging_binary / item.name)

    staging_binary_init = staging_vsrvrt / "_binary" / "__init__.py"
    staging_binary_init.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BINARY_DIR / "__init__.py", staging_binary_init)

    staging_platform_init = staging_vsrvrt / "_binary" / platform_id / "__init__.py"
    staging_platform_init.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BINARY_DIR / platform_id / "__init__.py", staging_platform_init)

    print(f"  Staging dir: {STAGING_DIR}")
    return STAGING_DIR


def build_wheel(platform_id, py_version):
    """Build a wheel for the specific Python version using staging directory."""
    print("Building wheel...")
    print(f"  Platform: {platform_id}")
    print(f"  Python: {py_version}")

    create_staging_directory(platform_id, py_version)

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    plat_name = get_wheel_platform_tag()

    setup_py = PROJECT_ROOT / "setup.py"
    pyproject_toml = PROJECT_ROOT / "pyproject.toml"
    manifest_in = PROJECT_ROOT / "MANIFEST.in"

    shutil.copy2(setup_py, STAGING_DIR / "setup.py")
    shutil.copy2(pyproject_toml, STAGING_DIR / "pyproject.toml")
    if manifest_in.exists():
        shutil.copy2(manifest_in, STAGING_DIR / "MANIFEST.in")
    shutil.copy2(PROJECT_ROOT / "README.md", STAGING_DIR / "README.md")
    shutil.copy2(PROJECT_ROOT / "LICENSE", STAGING_DIR / "LICENSE")

    result = subprocess.run(
        [
            sys.executable,
            str(STAGING_DIR / "setup.py"),
            "bdist_wheel",
            "-p",
            plat_name,
            "--python-tag",
            py_version,
            "-d",
            str(DIST_DIR),
        ],
        capture_output=True,
        text=True,
        cwd=str(STAGING_DIR),
    )

    if result.returncode != 0:
        print("ERROR: Failed to build wheel.")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    wheels = list(DIST_DIR.glob("*.whl"))
    if not wheels:
        print("ERROR: No wheel found in dist directory.")
        sys.exit(1)

    newest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    return newest_wheel


def cleanup_staging():
    """Remove staging directory after build."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
        print(f"  Cleaned staging: {STAGING_DIR}")


def main():
    """Main build process - builds binary and wheel for current Python version."""
    print("=" * 60)
    print("vsrvrt Wheel Builder")
    print("=" * 60)
    print()

    platform_id = get_platform_id()
    py_version = get_python_version_id()

    print(f"Platform: {platform_id}")
    print(f"Python: {py_version}")
    print()

    check_requirements()

    ext_file = compile_extension()
    print()

    copy_binary_to_source(ext_file, platform_id, py_version)
    print()

    wheel_file = build_wheel(platform_id, py_version)
    print()

    cleanup_staging()
    print()

    print("=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"Wheel: {wheel_file}")
    print()

    remaining = [v for v in SUPPORTED_PYTHON_VERSIONS if v != py_version]
    if remaining:
        print("To build wheels for other Python versions:")
        for v in remaining:
            if sys.platform == "win32":
                print(f"  py -{v[2]}.{v[3:]} build_wheels.py")
            else:
                print(f"  python{v[2]}.{v[3:]} build_wheels.py")
    else:
        print("All wheels built!")
    print()

    print("To upload to PyPI:")
    print(f"  twine upload {wheel_file}")
    print()


if __name__ == "__main__":
    main()
