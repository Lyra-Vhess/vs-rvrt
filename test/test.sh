#!/bin/zsh
if [ ! -t 1 ]; then
    konsole --hold -e "$0"
    exit
fi

source ../.venv312/bin/activate
pip install ../dist/vsrvrt-*-cp312-none-manylinux2014_x86_64.whl --no-deps vapoursynth requests tqdm einops urllib3 certifi charset_normalizer idna
python ../test/run_test.py output312.mkv

source ../.venv313/bin/activate
pip install ../dist/vsrvrt-*-cp313-none-manylinux2014_x86_64.whl --no-deps vapoursynth requests tqdm einops urllib3 certifi charset_normalizer idna
python ../test/run_test.py output313.mkv

source ../.venv314/bin/activate
pip install ../dist/vsrvrt-*-cp314-none-manylinux2014_x86_64.whl --no-deps vapoursynth requests tqdm einops urllib3 certifi charset_normalizer idna
python ../test/run_test.py output314.mkv
