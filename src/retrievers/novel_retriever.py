import pandas as pd
from langchain.text_splitter import CharacterTextSplitter

from src.utils.call_api.schema import EmbeddingEngineEnum
from src.retrievers.models.bm25_retriever import KeywordRetriever
from src.retrievers.models.vector_retriever import VectorRetriever
from src.retrievers.models.ensemble_retriever import EnsembleRetriever
from src.retrievers.models.llm_retriever import GPTRetriever

class NovelRetriever:
    def __init__(self, cfg: dict):
        # params
        self.topk = cfg.topk
        self.vec_rate = cfg.vec_rate
        self.ensemble_method = cfg.retrieve_method
        self.rank_impact_mitigator = cfg.rank_impact_mitigator

        # database
        self.index_df = pd.read_csv(cfg.index_path, dtype={'novel_id': int, 'chunk_id': int})

        # llm retirever
        self.llmretriever = GPTRetriever(cfg.llm_retieving_params)

        # re-splitter
        self.splitter = CharacterTextSplitter(
                separator = "。",                            
                chunk_size = cfg.re_split_chunksize,   
                chunk_overlap = int(cfg.re_split_chunksize * cfg.re_split_overlap),
            )
        
    def build_retriever(self, index_df: pd.DataFrame):
        keyretriever = KeywordRetriever(index_df, tokenized_column_name='augmented_tokenized_text', target_column_name='augmented_text')
        vecretriever = VectorRetriever(index_df, emb_column_name='augmented_embedding', target_column_name='augmented_text', model=EmbeddingEngineEnum.Large)
        ensretriever = EnsembleRetriever(vecretriever, keyretriever)
        return ensretriever


    def retrieve(self, query: str, novel_id: list[int], topk = None):
        topk = topk or self.topk
        ensretriever = self.build_retriever(self.index_df[self.index_df['novel_id'].isin(novel_id)].reset_index(drop=True))
        retrieved_df = ensretriever.retrieve(query=query,
                                             top_k=topk * 5,
                                             weights=[self.vec_rate, 1-self.vec_rate],
                                             ensemble_method=self.ensemble_method,
                                             rank_impact_mitigator=self.rank_impact_mitigator,
                                             require_columns=['text', 'novel_id', 'chunk_id'])
        retrieved_df = retrieved_df.drop_duplicates(subset='text')

        return retrieved_df[:topk]
    

    def retrieve_with_llm_retriever(self, query: str, novel_id: list[int]):
        retrieved_df = self.retrieve(query, novel_id, topk=self.topk * 2)

        retrieved_df = retrieved_df.sort_values(by=['novel_id', 'chunk_id'])
        retrieved_df_groupby = retrieved_df.groupby('novel_id')['text'].apply(lambda x: '\n\n'.join(x)).reset_index()

        re_splited_indexes = {'novel_id' : [], 'chunk_id' : [], 'text': []}
        for novel_id, text in zip(retrieved_df_groupby['novel_id'], retrieved_df_groupby['text']):
            splited_texts = self.splitter.split_text(text)
            re_splited_indexes['text'].extend(splited_texts)
            re_splited_indexes['novel_id'].extend([novel_id] * len(splited_texts))
            re_splited_indexes['chunk_id'].extend([i for i in range(1, len(splited_texts)+1)])
            re_splited_index_df = pd.DataFrame(re_splited_indexes)

        indexes_of_filtrered_indexes_by_llm = self.llmretriever.retrieve(query, self.topk, re_splited_indexes['text'])
        if indexes_of_filtrered_indexes_by_llm == []:
            return retrieved_df

        llm_retrieved_df = re_splited_index_df.iloc[indexes_of_filtrered_indexes_by_llm]
        return llm_retrieved_df