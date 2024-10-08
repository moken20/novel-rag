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
    cfg.mode = 'test'

    # text preoprocess
    cfg.metadata_path = PACKAGE_DIR.joinpath(f'data/processed/{cfg.mode}_sets/metadata.csv')

    # text split
    cfg.chunksize = 400
    cfg.overlap = 0.4
    cfg.re_split_chunksize = 200
    cfg.re_split_overlap = 0.2

    # index augmentation
    cfg.index_augmentation_params = {'messages': prompt_template['IndexExpansion']['messages']}

    # retrieve
    cfg.index_path = PACKAGE_DIR.joinpath(
        f'data/database/{cfg.mode}_sets/chartext_chunk{cfg.chunksize}_lap{cfg.overlap}/openai_large.csv'
    )
    cfg.query_path = PACKAGE_DIR.joinpath(
        f'data/processed/{cfg.mode}_sets/query.csv'
    )
    cfg.topk = 10
    cfg.vec_rate = 0.5
    cfg.retrieve_method = 'rrf'
    cfg.rank_impact_mitigator = 0
    cfg.llm_retieving_params = {'messages': prompt_template['LLMRetrieving']['messages']}


    # generation
    cfg.generation_params = [
        {'messages': prompt_template['GenerateStep1']['messages']},
        {'messages': prompt_template['GenerateStep2']['messages']},
        {'messages': prompt_template['GenerateStep3']['messages']},
        {'messages': prompt_template['GenerateStep4']['messages']},
        {'messages': prompt_template['GenerateStep5']['messages']},
        {'messages': prompt_template['Reasoning']['messages']}
    ]

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