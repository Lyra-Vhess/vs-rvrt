"""Setup script for vsrvrt package."""

from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "RVRT Vapoursynth Plugin for video restoration"

setup(
    name="vsrvrt",
    version="1.1.1",
    author="Lyra Vhess",
    author_email="auxilliary.email@protonmail.com",
    description="Vapoursynth plugin for RVRT video restoration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lyra-Vhess/vs-rvrt",
    packages=find_packages(),
    package_data={
        "vsrvrt": ["models/*.pth"],
        "vsrvrt.rvrt_src.models.op": ["*.cpp", "*.cu"],
        "vsrvrt._binary": ["**/*.pyd", "**/*.so"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.12,<3.15",
    install_requires=[
        "torch>=2.10.0",
        "torchvision",
        "numpy",
        "requests",
        "tqdm",
        "einops",
        "vapoursynth>=60",
        "packaging",
    ],
    include_package_data=True,
    zip_safe=False,
)
