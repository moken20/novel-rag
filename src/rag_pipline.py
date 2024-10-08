from src.retrievers.novel_retriever import NovelRetriever
from src.generator.generate_pipline import Generator

import ast
import pandas as pd
from tqdm import tqdm

class NovelRag:
    def __init__(self, cfg):
        self.retriever = NovelRetriever(cfg)
        self.generator = Generator(cfg.generation_params)
        self.novel_metadata =  pd.read_csv(cfg.metadata_path)

    def format_retrieved_info(self, retrieved_df: pd.DataFrame) -> str:
        novel_info = ''
        retrieved_df = retrieved_df.sort_values(by=['novel_id', 'chunk_id'])
        retrieved_df_groupby = retrieved_df.groupby('novel_id')['text'].apply(lambda x: '\n'.join(x)).reset_index()

        for novel_id, text in zip(retrieved_df_groupby["novel_id"], retrieved_df_groupby["text"]):
            novel_title = self.novel_metadata[self.novel_metadata['novel_id']==novel_id]['title'].values[0]
            novel_info += f'**小説タイトル: {novel_title}\n'
            novel_info += f'**小説の内容の一部抜粋: \n{text}\n'
            novel_info += '-------------------------------\n\n'
        
        return novel_info.strip()

    def generate(self, query: str, novel_id: list[int]) -> tuple[str, str]:
        retrieved_df = self.retriever.retrieve_with_llm_retriever(query, novel_id)
        retrieved_info = self.format_retrieved_info(retrieved_df)

        answer = self.generator.generate_answer(user_input=query,
                                                novel_info=retrieved_info,
                                               )
        reason = self.generator.generate_reason(ans=answer,
                                                user_input=query,
                                                novel_info=retrieved_info,
                                                )
        return answer, reason

    
    def generate_sequentially(self, query_df: pd.DataFrame) -> pd.DataFrame:
        answers = []
        reasons = []
        query_df['relevant_novel_id'] = query_df['relevant_novel_id'].map(ast.literal_eval)
        for query, novel_ids in tqdm(zip(query_df['problem'].values, query_df['relevant_novel_id'].values)):
            if not isinstance(novel_ids[0], int):
                print(novel_ids)
                raise ValueError(f'novel_id must be int, your type is {type(novel_ids[0])}')
            answer, reason = self.generate(query, novel_ids)
            answers.append(answer)
            reasons.append(reason)
    
        query_df['answer'] = answers
        query_df['reason'] = reasons
        return query_df