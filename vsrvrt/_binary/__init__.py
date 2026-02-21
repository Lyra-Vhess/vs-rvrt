"""Pre-built binary loader for the deform_attn CUDA extension.

This module loads the appropriate pre-compiled binary based on:
- Platform (Windows/Linux)
- Python version (3.12, 3.13)

Pre-built binaries are distributed in the wheel and require no compilation.
"""

import importlib
import sys
import platform


def get_platform_id():
    """Get the platform identifier for binary selection.

    Returns:
        str: Platform identifier ('win_amd64' or 'manylinux_x86_64')
    """
    if sys.platform == "win32":
        return "win_amd64"
    elif sys.platform.startswith("linux"):
        return "manylinux_x86_64"
    else:
        raise RuntimeError(
            f"Unsupported platform: {sys.platform}. "
            f"vsrvrt only supports Windows and Linux x86_64."
        )


def get_python_version_id():
    """Get the Python version identifier for binary selection.

    Returns:
        str: Python version identifier (e.g., 'cp312', 'cp313')
    """
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_binary_module_name():
    """Get the full module name for the pre-built binary.

    Returns:
        str: Full dotted module name
    """
    platform_id = get_platform_id()
    py_version = get_python_version_id()
    return f"vsrvrt._binary.{platform_id}.{py_version}.deform_attn"


def load_deform_attn():
    """Load the pre-built deform_attn extension.

    Returns:
        module: The loaded deform_attn extension module

    Raises:
        RuntimeError: If no compatible binary is found
    """
    py_version = get_python_version_id()
    supported_versions = ["cp312", "cp313", "cp314"]

    if py_version not in supported_versions:
        raise RuntimeError(
            f"Unsupported Python version: {sys.version_info.major}.{sys.version_info.minor}. "
            f"vsrvrt requires Python 3.12 or 3.13.\n"
            f"Supported versions: {', '.join(supported_versions)}"
        )

    platform_id = get_platform_id()
    module_name = get_binary_module_name()

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise RuntimeError(
            f"Failed to load pre-built deform_attn extension.\n\n"
            f"Platform: {platform_id}\n"
            f"Python: {py_version}\n"
            f"Module: {module_name}\n\n"
            f"This usually means you're installing from source instead of a pre-built wheel.\n\n"
            f"Please install vsrvrt from PyPI:\n"
            f"    pip install vsrvrt\n\n"
            f"If you need to install from source, please use the development branch "
            f"which supports JIT compilation, or build the wheels locally using:\n"
            f"    python build_wheels.py\n\n"
            f"Original error: {e}"
        ) from e


def get_binary_info():
    """Get information about the binary that would be loaded.

    Returns:
        dict: Information about platform, Python version, and module name
    """
    return {
        "platform": get_platform_id(),
        "python_version": get_python_version_id(),
        "module_name": get_binary_module_name(),
        "supported_python": ["cp312", "cp313", "cp314"],
        "supported_platforms": ["win_amd64", "manylinux_x86_64"],
    }
