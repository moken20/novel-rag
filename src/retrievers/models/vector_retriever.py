import logging
from pathlib import Path

import faiss
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retrievers.models.retriver_base import IndexesRetrieverBase
from src.utils.call_api.schema import EmbeddingEngineEnum
from src.utils.call_api.util_openai import get_embedding

logger = logging.getLogger(__name__)

tqdm.pandas()


class VectorRetriever(IndexesRetrieverBase):
    EMB_COLUMN_NAME = 'embedding'
    TARGET_COLUMN_NAME = 'text'

    def __init__(
        self,
        index_df: pd.DataFrame,
        target_column_name: str | None = None,
        model: EmbeddingEngineEnum = EmbeddingEngineEnum.Small,
        dim_embedding: int = 1536,
    ) -> None:
        self.model = model
        self.dim_embedding = dim_embedding
        self.index = None
        self.index_df = self._preprocess_index_df(index_df.reset_index(drop=True))

        if self.model == EmbeddingEngineEnum.Large:
            self.dim_embedding = 3072

        if target_column_name:
            self.index_df[self.TARGET_COLUMN_NAME] = self.index_df[target_column_name]
        elif self.TARGET_COLUMN_NAME not in self.index_df.columns:
            raise ValueError(
                f'Provide a target_column_name or include a column named {self.TARGET_COLUMN_NAME} in the index_df.'
            )
        self._build_index()


    def _build_index(self) -> None:
        if self.EMB_COLUMN_NAME not in self.index_df.columns:
            logger.info(f'Embedding column {self.EMB_COLUMN_NAME} not found. Generating embeddings...')
            self.index_df[self.EMB_COLUMN_NAME] = self.index_df[self.TARGET_COLUMN_NAME].progress_apply(
                lambda x: get_embedding(x, model=self.model)
            )
        self.index = faiss.IndexFlatIP(self.dim_embedding)
        embeddings = np.array(list(self.index_df[self.EMB_COLUMN_NAME].values))
        self.index.add(embeddings.reshape(-1, self.dim_embedding).astype('float32'))


    def _preprocess_index_df(self, index_df: pd.DataFrame) -> pd.DataFrame:
        if self.EMB_COLUMN_NAME in index_df.columns:
            index_df[self.EMB_COLUMN_NAME] = index_df[self.EMB_COLUMN_NAME].map(eval)
        return index_df


    def to_csv(self, index_df_path: str | Path) -> None:
        self.index_df.to_csv(index_df_path, index=False)


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        require_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        query_emb = get_embedding(query, model=self.model)
        distances, indices = self.index.search(np.array([query_emb]).astype('float32'), top_k)
        result_df = self.index_df.iloc[indices[0]].reset_index()

        result_df['similar_document'] = result_df[self.TARGET_COLUMN_NAME]
        similarity = distances[0]
        result_df['similarity'] = similarity
        require_columns = ['similar_document', 'similarity'] + (require_columns or [])

        return result_df[require_columns]