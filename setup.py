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
    version="1.0.0",
    author="RVRT Vapoursynth Plugin",
    description="Vapoursynth plugin for RVRT video restoration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JingyunLiang/RVRT",
    packages=find_packages(),
    package_data={
        "vsrvrt": ["models/*.pth"],
        "vsrvrt.rvrt_src.models.op": ["*.cpp", "*.cu"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
