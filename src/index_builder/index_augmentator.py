import logging

import pandas as pd
from tqdm import tqdm

from src.utils.call_api.util_openai import (
    update_params_with_dict,
    prepare_input_messages,
    call_chatgpt_api,
)
from configulator.config import OpenAISettings

logging.getLogger(__name__)

class IndexAugmentator:
    def __init__(
        self,
        index_df: pd.DataFrame,
        request_params: dict
    ):
        self.index_df = index_df
        self.openai_params = update_params_with_dict(
            base_model=OpenAISettings().openai_params, update_params=request_params
        )

    
    def generate_indexes(self, chunk_text: str, num_expands: int) -> list[str]:
        messages = prepare_input_messages(prompt=self.openai_params.messages, num_expands=num_expands, text=chunk_text)

        augmented_text = call_chatgpt_api(messages=messages, model=self.openai_params.model, request_params=self.openai_params)
        comma_splited_text = augmented_text.strip().split(',')
        ten_splited_text = augmented_text.strip().split('、')
        return comma_splited_text if len(comma_splited_text) >= len(ten_splited_text) else ten_splited_text


    def augment_index(self, num_expands: int = 10) -> pd.DataFrame:
        augmented_indexes = {'novel_id': [], 'chunk_id': [], 'augmented_text': []}

        for _, row in tqdm(self.index_df.iterrows()):
            new_indexes = self.generate_indexes(row['text'], num_expands)
            augmented_indexes['novel_id'].extend([row['novel_id']] * len(new_indexes))
            augmented_indexes['chunk_id'].extend([row['chunk_id']] * len(new_indexes))
            augmented_indexes['augmented_text'].extend(new_indexes)

        augmented_index_df = pd.DataFrame(augmented_indexes)
        print(f'base index df size is {len(self.index_df)}, augmented index df size is {len(augmented_index_df)}')
        
        return augmented_index_df