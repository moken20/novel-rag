from enum import Enum


class EmbeddingEngineEnum(str, Enum):
    Small = 'text-embedding-3-small'
    Large = 'text-embedding-3-large'
    ADA002 = 'text-embedding-ada-002'