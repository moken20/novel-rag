import logging

import numpy as np
import pandas as pd
from sklearn import preprocessing

from src.retrievers.models.retriver_base import Retriever

logger = logging.getLogger(__name__)


class EnsembleRetriever(Retriever):
    def __init__(
        self,
        *retrievers: Retriever,
    ) -> None:
        """Wrapper class for ensemble retriever.

        Args:
            *retrievers (VectorRetriever | KeywordRetriever): A list of retrievers to be ensembled.
        """
        if any(len(retriever.index_df) != len(retrievers[0].index_df) for retriever in retrievers):
            raise ValueError('Not all retriever have same index')
        self.retrievers = retrievers
        super().__init__(
            index_df=retrievers[0].index_df,
            target_column_name=retrievers[0].target_column_name,
        )

    def _build_index(self) -> None:
        pass

    def retrieve(
        self,
        query: str,
        weights: list[float] | None,
        ensemble_method: str = 'rrf',
        rank_impact_mitigator: int | None = 60,
        top_k: int = 5,
        require_columns: list[str] | None =['chunk_id'],
    ) -> pd.DataFrame:
        """Execute a search across multiple retrievers and fuse these rank.

        Args:
            query (str): Query text.
            weirhts (list(float) | None) : A list of weights corresponding to the retrievers. Defaults to equal
                weighting for all retrievers.
            ensemble_method (str) : Ensemble method which can be selected from ['rrf', 'combsum']. Defaults to 'rrf'.
                rrf (Reciprocal Rank Fusion) :
                    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
                combsum (Combination of Multiple Searches):
                    https://www.khoury.northeastern.edu/home/jaa/IS4200.10X1/resources/fox94combination.pdf
            rank_impact_mitigator (int | None): A constant added to the rank, controlling the balance between
                the importance of high-ranked items and the consideration given to lower-ranked items.
                Default is 60, which is found to be near-optimal experimentally in thesis.
                Only used when rrf ensemble method is choosed.
            top_k (int): Number of results to return. Defaults to 5.
            require_columns (list[str] | None): Columns to return. Defaults to None.

        """
        if not weights:
            weights = [1 / len(self.retrievers)] * len(self.retrievers)

        if len(self.retrievers) != len(weights):
            raise ValueError(
                f'number of weights {len(weights)} is not match number of retrievers {len(self.retrievers)}'
            )

        ensembled_scores = np.zeros(
            len(self.index_df),
        )
        for i in range(len(self.retrievers)):
            ranked_index = self.retrievers[i].retrieve(query=query, top_k=len(self.index_df), require_columns=['index'])

            if ensemble_method == 'rrf':
                ranked_index['score'] = weights[i] * (1 / (ranked_index.index + 1 + rank_impact_mitigator))
            elif ensemble_method == 'combsum':
                std_sim = preprocessing.scale(ranked_index['similarity'].astype(float))
                ranked_index['score'] = weights[i] * std_sim
            else:
                raise ValueError('Invalid ensemble method is specified. Please select the method from [rrf, combsum].')

            score_dic = dict(zip(ranked_index['index'], ranked_index['score']))
            for i in range(len(ensembled_scores)):
                ensembled_scores[i] += score_dic[i]

        top_k_indices = np.argsort(ensembled_scores)[::-1][:top_k]
        result_df = self.index_df.iloc[top_k_indices].reset_index()

        result_df['similar_document'] = result_df[self.target_column_name]
        result_df['similarity'] = [ensembled_scores[i] for i in top_k_indices]
        require_columns = ['similar_document', 'similarity'] + (require_columns or [])

        return result_df[require_columns]