import os
import glob
from pathlib import Path

import pandas as pd

from src.env import PACKAGE_DIR
from src.utils.file_io import extract_id_from_filename
from src.utils.preprocess.cleans import replace_symbols, remove_header, remove_footer

def preprocess_novel(text: str) -> str:
    text = remove_header(text)
    text = replace_symbols(text)
    text = remove_footer(text)
    return text

def run(cfg):
    if cfg.mode == 'valid':
        raw_novel_dir = PACKAGE_DIR.joinpath('data/raw/valid_sets/novels')
        processed_novel_dir = PACKAGE_DIR.joinpath('data/processed/valide_sets/novels')
        encoding = 'utf-8'
    elif cfg.mode == 'test':
        raw_novel_dir = PACKAGE_DIR.joinpath('data/raw/test_sets/novels')
        processed_novel_dir = PACKAGE_DIR.joinpath('data/processed/test_sets/novels')
        encoding = 'shift-jis'

    os.makedirs(processed_novel_dir, exist_ok=True)
    raw_files = glob.glob(str(raw_novel_dir.joinpath('*')))
    novel_meta_data = {'novel_id': [], 'title': [], 'author': []}

    for raw_file in raw_files:
        with open(raw_file, mode='r', encoding=encoding) as f:
            novel_meta_data['novel_id'].append(extract_id_from_filename(raw_file))
            novel_meta_data['title'].append(next(f).strip())
            novel_meta_data['author'].append(next(f).strip())

            text = f.read()
            text = preprocess_novel(text)

        processed_novel_file = processed_novel_dir.joinpath(Path(raw_file).name)
        with open(processed_novel_file, mode='w', encoding='utf-8') as f:
            f.write(text)

    meta_data = pd.DataFrame(novel_meta_data)
    meta_data.to_csv(processed_novel_dir.parent.joinpath('metadata.csv'), index=None)

