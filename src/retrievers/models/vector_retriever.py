import logging
from pathlib import Path

import faiss
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retrievers.models.retriver_base import Retriever
from src.utils.call_api.schema import EmbeddingEngineEnum
from src.utils.call_api.util_openai import get_embedding

logger = logging.getLogger(__name__)

tqdm.pandas()


class VectorRetriever(Retriever):
    def __init__(
        self,
        index_df: pd.DataFrame,
        target_column_name: str = 'text',
        emb_column_name: str = 'embedding',
        model: EmbeddingEngineEnum = EmbeddingEngineEnum.Small,
    ) -> None:
        self.model = model
        self.emb_column_name = emb_column_name
        dim_embedding = 3072 if self.model == EmbeddingEngineEnum.Large else 1536

        super().__init__(
            index_df=index_df,
            target_column_name=target_column_name,
            model=model,
            dim_embedding=dim_embedding,
        )

    def _build_index(
            self,
            model: EmbeddingEngineEnum,
            dim_embedding: int,
        ) -> None:
        if self.emb_column_name not in self.index_df.columns:
            logger.info(f'Embedding column {self.emb_column_name} not found. Generating embeddings...')
            self.index_df[self.emb_column_name] = self.index_df[self.target_column_name].progress_apply(
                lambda x: get_embedding(x, model=model)
            )
        self.index = faiss.IndexFlatIP(dim_embedding)
        embeddings = np.array(list(self.index_df[self.emb_column_name].values))
        self.index.add(embeddings.reshape(-1, dim_embedding).astype('float32'))


    def _preprocess_index_df(self, index_df: pd.DataFrame) -> pd.DataFrame:
        if self.emb_column_name in index_df.columns:
            index_df[self.emb_column_name] = index_df[self.emb_column_name].map(eval)
        return index_df


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        require_columns: list[str] | None = ['chunk_id'],
    ) -> pd.DataFrame:
        query_emb = get_embedding(query, model=self.model)
        distances, indices = self.index.search(np.array([query_emb]).astype('float32'), top_k)
        result_df = self.index_df.iloc[indices[0]].reset_index()

        result_df['similar_document'] = result_df[self.target_column_name]
        similarity = distances[0]
        result_df['similarity'] = similarity
        require_columns = ['similar_document', 'similarity'] + (require_columns or [])

        return result_df[require_columns]