#!/bin/zsh
if [ ! -t 1 ]; then
    konsole --hold -e "$0"
    exit
fi

python3.12 -m venv ../.venv312
source ../.venv312/bin/activate
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging
export CC=gcc CXX=g++ CUDA_HOME=/opt/cuda
python ../build_wheels.py
deactivate

python3.13 -m venv ../.venv313
source ../.venv313/bin/activate
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging
export CC=gcc CXX=g++ CUDA_HOME=/opt/cuda
python ../build_wheels.py
deactivate

python3.14 -m venv ../.venv314
source ../.venv314/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging
export CC=gcc CXX=g++ CUDA_HOME=/opt/cuda
python ../build_wheels.py
deactivate
