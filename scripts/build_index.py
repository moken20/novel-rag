from fire import Fire
import logging

import pandas as pd

from src.env import PACKAGE_DIR
from configulator.config import default_config
from src.index_builder.text_splitter import CharTextSplitter
from src.index_builder.index_augmentator import IndexAugmentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index(mode: str, chunksize: int, overlap: int) -> pd.DataFrame:
    novel_dir = PACKAGE_DIR.joinpath(f'data/processed/{mode}_sets/novels')

    splitter = CharTextSplitter(novel_dir, chunksize, overlap, from_tiktoken=False)
    index_df = splitter.split_novels(return_df=True)
    logger.info(f'splited index size is {len(index_df)}')


    augmentator = IndexAugmentator(index_df=index_df, request_params=default_config().index_augmentation_params)
    augmented_index_df = augmentator.augment_index(num_expands=10)
    logger.info(f'augmented index size is {len(augmented_index_df)}')

    augmented_index_df = augmented_index_df.merge(index_df, on=['novel_id', 'chunk_id'])

    return augmented_index_df


if __name__ == '__main__':
    Fire(build_index)
