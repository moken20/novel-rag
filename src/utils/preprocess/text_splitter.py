import glob
from pathlib import PosixPath

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import pandas as pd

from src.env import PACKAGE_DIR
from src.utils.file_io import extract_id_from_filename

class NovelSplitter:
    def __init__(self, novels_dir: PosixPath, **build_params):
        self.novels = self._load_novel_data(novels_dir)
        self._build_splitter(**build_params)

    def _build_splitter(**build_parameter) -> None:
        pass

    def _load_novel_data(self, novels_dir: PosixPath) -> list[str]:
        novel_path_list = glob.glob(str(novels_dir.joinpath('*')))
        if novel_path_list == []:
            raise ValueError('novel file is empty')

        novels = []
        for novel_path in novel_path_list:
            loader = TextLoader(novel_path)
            novel = loader.load()
            novels.append(novel)
        return novels
    
    def split_novels(self, return_df: bool = True) -> pd.DataFrame:
        splited_novels = []
        chunk_id = []
        for i in range(len(self.novels)):
            splited_novel = self.novel_splitter.split_documents(self.novels[i])
            splited_novels.extend(splited_novel)
            chunk_id.extend([i for i in range(1, len(splited_novel)+1)])

        if return_df:
            return self._to_dataframe(splited_novels, chunk_id)
        return splited_novels
    
    def _to_dataframe(self, document_list, chunk_id):
        splited_novel_dic = {
            'text': [doc.page_content for doc in document_list],
            'novel_id': [extract_id_from_filename(doc.metadata['source']) for doc in document_list],
            'chunk_id': chunk_id
        }
        return pd.DataFrame(splited_novel_dic)


class CharTextSplitter(NovelSplitter):
    def __init__(self, novels_dir: PosixPath, chunksize: int, overlap: float, from_tiktoken: bool = False):
        self.output_data_path = PACKAGE_DIR.joinpath(f'data/database/chartext_sp/df_{chunksize}_{overlap*100}.csv')
        super().__init__(
            novels_dir=novels_dir,
            chunksize=chunksize,
            overlap=overlap,
            from_tiktoken=from_tiktoken)

    def _build_splitter(self, chunksize, overlap, from_tiktoken: bool = False) -> None:
        if from_tiktoken:
            self.novel_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator = "。",           
                chunk_size = chunksize, 
                chunk_overlap = int(chunksize * overlap),
            )
        else:
            self.novel_splitter = CharacterTextSplitter(
                separator = "。",                                  # セパレータ
                chunk_size = chunksize,                            # チャンクの文字数
                chunk_overlap = int(chunksize * overlap),          # チャンクオーバーラップの文字数
            )


class RecurCharTextSplitter(NovelSplitter):
    def __init__(self, novels_dir: PosixPath, chunksize: int, overlap: float):
        self.output_data_path = PACKAGE_DIR.joinpath(f'data/database/recurchartext_sp/df_{chunksize}_{overlap*100}.csv')
        super().__init__(
            novels_dir=novels_dir,
            chunksize=chunksize,
            overlap=overlap)

    def _build_splitter(self, chunksize, overlap) -> None:
        self.novel_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunksize,                            # チャンクの文字数
            chunk_overlap = int(chunksize * overlap),          # チャンクオーバーラップの文字数
        )
        print(int(chunksize * overlap))