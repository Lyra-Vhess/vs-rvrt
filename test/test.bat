set PATH=C:\Program Files\ShareX;%PATH%
for %%f in (..\dist\vsrvrt-*-cp312-none-win_amd64.whl) do (
    ..\.venv312\Scripts\pip install "%%f" --no-deps vapoursynth requests tqdm einops urllib3 certifi charset_normalizer idna
)
..\.venv312\Scripts\python run_test.py output312.mkv
for %%f in (..\dist\vsrvrt-*-cp313-none-win_amd64.whl) do (
    ..\.venv313\Scripts\pip install "%%f" --no-deps vapoursynth requests tqdm einops urllib3 certifi charset_normalizer idna
)
..\.venv312\Scripts\python run_test.py output313.mkv
pause