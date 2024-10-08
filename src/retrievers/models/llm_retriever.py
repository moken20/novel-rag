import json
import logging

from configulator.config import OpenAISettings
from src.retrievers.models.retriver_base import Retriever
from src.utils.call_api.util_openai import (
    update_params_with_dict,
    prepare_input_messages,
    call_chatgpt_api,
)

logger = logging.getLogger(__name__)


class GPTRetriever(Retriever):
    def __init__(self, request_params: dict):
        self.openai_params = update_params_with_dict(
                    base_model=OpenAISettings().openai_params, update_params=request_params
                )
        
    def retrieve(self, query: str, top_k: int, references: list[str]) -> list[int]:
        contexts = [f'{idx} : {context}' for idx, context in enumerate(references)]

        messages = prepare_input_messages(
            prompt=self.openai_params.messages,
            top_k=top_k,
            novel_info='\n'.join(contexts),
            query=query,
        )
        retrieved_indexes = call_chatgpt_api(
            messages=messages, model=self.openai_params.model, request_params=self.openai_params
        )
        try:
            selected_indexes = json.loads(f'[{retrieved_indexes}]')
            selected_indexes = [
                selected_index for selected_index in selected_indexes if (0 <= int(selected_index) < len(references))
            ]
        except json.decoder.JSONDecodeError:
            selected_indexes = []

        return selected_indexes