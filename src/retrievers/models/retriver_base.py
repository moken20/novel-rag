import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class Retriever(metaclass=ABCMeta):

    def __init__(
        self,
        index_df: pd.DataFrame,
        target_column_name: str,
        **build_params,
    ) -> None:
        """Wrapper class for faiss index.

        Args:
            index_df (pd.DataFrame): DataFrame containing text columns.
            target_column_name (str | None, optional): Name of the text column.
                If None, the text column is assumed to be named 'text'. Defaults to None.
            **build_params: Parameters for building the index.
        """
        self.index_df = index_df.reset_index(drop=True)
        self.target_column_name = target_column_name
        if self.target_column_name not in self.index_df.columns:
            raise ValueError(
                f'Provide a target_column_name or include a column named {self.target_column_name} in the index_df.'
            )

        self.index = None
        self.index_df = self._preprocess_index_df(self.index_df)
        self._build_index(**build_params)

    @classmethod
    def from_csv(
        cls,
        index_df_path: str | Path,
        **kwargs,
    ) -> 'Retriever':
        """Load index from csv file.

        Args:
            index_df_path (str | Path): Path to the csv file.
            **kwargs: Parameters for the constructor.
        """
        index_df = pd.read_csv(index_df_path)
        return cls(index_df=index_df, **kwargs)

    def _build_index(self, **build_params) -> None:
        """Build the index."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> pd.DataFrame:
        """Search the index for the query.

        Args:
            query (str): Query text.
            top_k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing the search results.
        """
        pass

    def _preprocess_index_df(self, index_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess index dataframe before building index.

        Args:
            index_df (pd.DataFrame): DataFrame to be processed.

        Returns:
            pd.DataFrame: processed DataFrame.
        """
        return index_df

    def to_csv(self, index_df_path: str | Path) -> None:
        """Save the index to csv file.

        Args:
            index_df_path (str | Path): Path to the csv file.
        """
        self.index_df.to_csv(index_df_path, index=False)

    def add_column(self, values: ArrayLike | list, column_name: str) -> None:
        """Add a column to the index.

        Args:
            values (ArrayLike | list): Values to add.
            column_name (str): Name of the column.
        """
        self.index_df[column_name] = values