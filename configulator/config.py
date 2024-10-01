from ml_collections import ConfigDict
import yaml

from src.utils.call_api.schema import (
    ModelEnum,
    OpenAIChatParameter,
)
from src.env import PACKAGE_DIR


with open(PACKAGE_DIR/'src/prompt/prompts.yml') as fin:
    prompt_template = yaml.safe_load(fin)


def default_config():
    cfg = ConfigDict()
    cfg.name = 'rag_1'
    cfg.verbose = False
    cfg.mode = 'valid'

    # text splitter
    cfg.chunksize = 200
    cfg.overlap = 0.4

    # index augmentation
    cfg.index_augmentation_params = {'messages': prompt_template['IndexExpansion']['messages']}

    return cfg


class OpenAISettings:
    openai_params = OpenAIChatParameter(
        messages=[],
        model=ModelEnum.GPT4O_MINI,
        temperature=0,
        top_p=0.2,
        frequency_penalty=0,
        presence_penalty=0,
        prompt_role='user',
    )