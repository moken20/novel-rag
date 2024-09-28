from ml_collections import ConfigDict

from src.utils.call_api.schema import (
    ModelEnum,
    OpenAIChatParameter,
)

def default_config():
    cfg = ConfigDict()
    cfg.name = 'rag_1'
    cfg.verbose = False
    cfg.mode = 'valid'

    # text splitter
    cfg.chunksize = 200
    cfg.overlap = 0.4

    # OpenAi
    cfg.openai_setting = OpenAIChatParameter(
        messages=[],
        model=ModelEnum.GPT4O_MINI,
        temperature=0,
        top_p=0.2,
        frequency_penalty=0,
        presence_penalty=0,
        prompt_role='user',
    )

    return cfg