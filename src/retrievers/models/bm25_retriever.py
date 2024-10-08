import logging

import MeCab
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.retrievers.models.retriver_base import Retriever
from src.env import PACKAGE_DIR

logger = logging.getLogger(__name__)

with open(PACKAGE_DIR/'src/retrievers/config/stopwords-ja.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = [line.strip() for line in f.readlines()]


class KeywordRetriever(Retriever):
    def __init__(
        self,
        index_df: pd.DataFrame,
        target_column_name: str = 'text',
        tokenized_column_name: str = 'tokenized_text',
        mecab_dic_type: str | None = 'unidic_lite',
        mecab_dic_path: str | None = None,
        mecab_other_option: str = '',
        target_pos_list: list[str] = ['名詞', '動詞'],
        filter_str_type: str | None = None,
    ) -> None:
        """Wrapper class BM250kapi index.

        Args:
            index_df (pd.DataFrame): DataFrame containing text columns.
            target_column_name (str | None, optional): Name of the target column for searching.
            mecab_dic_type (str | None, optional): Type of mein dicitionary name used in morphological analysis.
                Can be selected from: [None, 'unidic_lite']. Defaults to 'unidic_lite'.
                None can be selected when mecab_dic_path is specified.
            mecab_dic_path (str | None, optional) : Path to a base dictionary. Mainly used when using NEologd.
            mecab_other_option (str): Other options for MeCab.
            target_pos_list (list[str]): A list of parts of speech to extract from the input text. Default to
                ['名詞', '動詞'].
            filter_str_type (str | None, optional): Type of filtering to apply to the extracted keywords.
                Possible values are [None, 'digit', 'numeric', 'ascii']. Defaults to None.
        """
        self._build_tokenizer(
            mecab_dic_type=mecab_dic_type, mecab_dic_path=mecab_dic_path, mecab_other_option=mecab_other_option
        )
        self.filter_str_type = filter_str_type
        self.target_pos_list = target_pos_list
        self.tokenized_column_name = tokenized_column_name
        super().__init__(
            index_df=index_df,
            target_column_name=target_column_name,
            target_pos_list=target_pos_list,
            filter_str_type=filter_str_type,
        )


    def _build_tokenizer(
        self,
        mecab_dic_type: str | None = 'unidic_lite',
        mecab_dic_path: str | None = None,
        mecab_other_option: str = '',
    ) -> None:
        """Build MeCab tokenizer.

        Args:
            mecab_dic_type (str | None, optional): Type of mein dicitionary name used in morphological analysis.
                Can be selected from: [None, 'unidic_lite']. Defaults to 'unidic_lite'.
                None can be selected when mecab_dic_path is specified.
            mecab_dic_path (str | None, optional) : Path to a base dictionary. Mainly used when using NEologd.
            mecab_other_option (str): Other options for MeCab.
        """
        if mecab_dic_type and not mecab_dic_path:
            mecab_dic_type = mecab_dic_type.lower()
            if mecab_dic_type == 'unidic_lite':
                import unidic_lite

                mecab_dic_path = unidic_lite.DICDIR
            else:
                raise ValueError('Invalid mecab_dic_type is specified. Please select the type from [unidic_lite].')
        elif not mecab_dic_path:
            raise FileNotFoundError('Neither `mecab_dic_type` nor `mecab_dic_path` were input.')

        self.mecab = MeCab.Tagger(f'-d {mecab_dic_path}' + mecab_other_option)


    def _build_index(
            self,
            target_pos_list: list[str] = ['名詞', '動詞'],
            filter_str_type: str | None = None
    ) -> None:
        """Build the index.

        Args:
            target_pos_list (list[str]): A list of parts of speech to extract from the input text. Default to
                ['名詞', '動詞'].
            filter_str_type (str | None, optional): The type of filtering to apply to the extracted keywords.
                Possible values are [None, 'digit', 'numeric', 'ascii']. Defaults to None.
        """
        if self.tokenized_column_name not in self.index_df.columns:
            logger.info(
                f'Keywords column {self.tokenized_column_name} not found.'
                f'Generating kewwords by extract nouns and verbs from target column {self.target_column_name}...'
            )
            self.index_df[self.tokenized_column_name] = self.index_df[self.target_column_name].apply(
                lambda x: self.tokenize(x, filter_str_type=filter_str_type, target_pos_list=target_pos_list)
            )

        keywords = list(self.index_df[self.tokenized_column_name].values)
        self.index = BM25Okapi(keywords)


    def _preprocess_index_df(self, index_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess index dataframe before building index.

        Args:
            index_df (pd.DataFrame): DataFrame to be processed.

        Returns:
            pd.DataFrame: processed DataFrame.
        """
        if self.tokenized_column_name in index_df.columns:
            index_df[self.tokenized_column_name] = index_df[self.tokenized_column_name].map(eval)
        return index_df


    def tokenize(
        self,
        text: str,
        target_pos_list: list[str] = ['名詞', '動詞'],
        filter_str_type: str | None = None,
    ) -> list[str]:
        """Processes the given text to extract keywords based on specified filter criteria

        Args:
            text (str): The text to be processed.
            target_pos_list (list[str]): A list of parts of speech to extract from the input text. Default to
                ['名詞', '動詞'].
            filter_str_type (str | None, optional): The type of filtering to apply to the extracted keywords.
                Possible values are [None, 'digit', 'numeric', 'ascii']. Defaults to None.

        Returns:
            list[str]: A list of extracted and filtered keywords.
        """
        if filter_str_type not in [None, 'digit', 'numeric', 'ascii']:
            raise ValueError(
                'Invalid filter_str_type is specified.' 'Please select the type from [None, digit, numeric, ascii].'
            )

        parsed_text = self.mecab.parse(text)
        lines = parsed_text.split('\n')
        keywords = []

        for line in lines:
            if any(pos in line for pos in target_pos_list):
                parts = line.split('\t')
                word = parts[0]

                if word in STOPWORDS:
                    continue
                if filter_str_type is None:
                    keywords.append(word)
                    continue

                if filter_str_type == 'digit' and not word.isdigit():
                    keywords.append(word)
                elif filter_str_type == 'numeric' and not word.isnumeric():
                    keywords.append(word)
                elif filter_str_type == 'ascii' and not word.isascii():
                    keywords.append(word)

        return keywords


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        require_columns: list[str] | None = ['chunk_id'],
    ) -> pd.DataFrame:
        """Search the index for the query.

        Args:
            query (str): Query text.
            top_k (int): Number of results to return. Defaults to 5.
            require_columns (list[str] | None, optional): Columns to return. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the search results.
        """
        query = self.tokenize(query, self.target_pos_list, self.filter_str_type)
        similarities = np.array(self.index.get_scores(query))
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        result_df = self.index_df.iloc[top_k_indices].reset_index()

        result_df['similar_document'] = result_df[self.target_column_name]
        result_df['similarity'] = [similarities[i] for i in top_k_indices]
        require_columns = ['similar_document', 'similarity'] + (require_columns or [])

        return result_df[require_columns]