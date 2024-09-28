from typing import Any

from pydantic_settings import BaseSettings
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.utils.call_api.schema import (
    EmbeddingEngineEnum,
    Message,
    ModelEnum,
    OpenAIBasicParameter,
)
from src.env import PACKAGE_DIR

OpenAI()


def format_input_for_chatgpt(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    formatted_messages: list[ChatCompletionMessageParam] = []

    for msg in messages:
        formatted_messages.append({'role': msg.role, 'content': msg.content})
    return formatted_messages


def prepare_input_messages(prompt: list, **kwargs: Any) -> list[Message]:
    messages = [
        Message(
            role=turn.role,
            content=turn.content.format(**kwargs),
        )
        for turn in prompt
    ]
    return messages


@retry(
    retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError, openai.APIError)),
    wait=wait_random_exponential(multiplier=1, min=1, max=240),
    stop=stop_after_attempt(7),
)
def call_chatgpt_api(
    model: ModelEnum,
    request_params: OpenAIBasicParameter,
    messages: list[Message],
) -> ChatCompletion:

    completion = openai.chat.completions.create(
        temperature=request_params.temperature,
        top_p=request_params.top_p,
        frequency_penalty=request_params.frequency_penalty,
        presence_penalty=request_params.presence_penalty,
        max_tokens=request_params.max_tokens,
        messages=format_input_for_chatgpt(messages),
        user=request_params.user or '',
        model=model.value,
        seed=request_params.seed,
    )

    return content if (content := completion.choices[0].message.content) is not None else ''


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


def update_params_with_dict(base_model: OpenAIBasicParameter, update_params: dict) -> OpenAIBasicParameter:
    base_params = base_model.model_dump()
    base_params.update({key: update_params[key] for key in base_params.keys() & update_params.keys()})
    return base_model.model_validate(base_params)