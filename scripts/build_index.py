import os
from fire import Fire

from src.env import PACKAGE_DIR
from configulator.config import default_config
from src.index_builder.text_splitter import CharTextSplitter
from src.index_builder.index_augmentator import IndexAugmentator


def run(mode: str, chunksize: int, overlap: int):
    if mode == 'valid':
        novel_dir = PACKAGE_DIR.joinpath('data/processed/valide_sets/novels')
        database_dir = PACKAGE_DIR.joinpath(f'data/database/valid_sets/chartext_chunk{chunksize}_lap{overlap}')
    elif mode == 'test':
        novel_dir = PACKAGE_DIR.joinpath('data/processed/test_sets/novels')
        database_dir = PACKAGE_DIR.joinpath(f'data/database/valid_sets/chartext_chunk{chunksize}_lap{overlap}')

    splitter = CharTextSplitter(novel_dir, chunksize, overlap, from_tiktoken=False)
    index_df = splitter.split_novels(return_df=True)

    augmentator = IndexAugmentator(index_df=index_df, request_params=default_config().index_augmentation_params)
    augmented_index_df = augmentator.augment_index(num_expands=10)

    augmented_index_df.merge(index_df, on=['novel_id', 'chunk_id'])
    os.makedirs(database_dir, exist_ok=True)
    augmented_index_df.to_csv(database_dir.joinpath('base_chunked_index.csv'), index=False)

    return augmented_index_df


if __name__ == '__main__':
    Fire(run)
