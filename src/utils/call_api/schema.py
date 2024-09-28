from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    System = 'system'
    User = 'user'
    Assistant = 'assistant'


class Message(BaseModel):
    role: Role
    content: str


class EmbeddingEngineEnum(str, Enum):
    Small = 'text-embedding-3-small'
    Large = 'text-embedding-3-large'
    ADA002 = 'text-embedding-ada-002'


class ModelEnum(str, Enum):
    GPT4 = 'gpt-4'
    GPT4_32K = 'gpt-4-32k'
    GPT4T = 'gpt-4-turbo'
    GPT4O = 'gpt-4o'
    GPT4O_MINI = 'gpt-4o-mini'


class OpenAIBasicParameter(BaseModel):
    model: ModelEnum = ModelEnum('gpt-4o-mini')
    temperature: float = Field(ge=0.0, le=1.0, default=0.0)
    top_p: float = Field(ge=0.0, le=1.0, default=1.0)
    frequency_penalty: float = Field(ge=0.0, default=0.0)
    presence_penalty: float = Field(ge=0.0, default=0.0)
    n: int = Field(ge=0, default=1)
    seed: Optional[int] = None
    max_tokens: Optional[int] = Field(ge=0, default=None)
    token_dependent_model_selection: bool = False
    output_token_num_buffer: int = Field(ge=0, default=0)
    limit_input_token_num: Optional[int] = Field(ge=0, default=None)


class OpenAIChatParameter(OpenAIBasicParameter):
    messages: list[Message]
    prompt_role: Optional[str]