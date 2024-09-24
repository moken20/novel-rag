from typing import Any, Literal, Optional

from pydantic_settings import BaseSettings
import openai
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.utils.call_api.schema import EmbeddingEngineEnum
from src.env import PACKAGE_DIR

OpenAI()

@retry(
    retry=retry_if_exception_type(
        (openai.APIConnectionError, openai.RateLimitError, openai.APIError)
    ),
    wait=wait_random_exponential(multiplier=1, min=1, max=240),
    stop=stop_after_attempt(7),
)
def get_embedding(text: str, model: EmbeddingEngineEnum) -> list[float]:
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding