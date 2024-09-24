from typing import Any


class IndexesRetrieverBase:
    def retrieve(self, query: Any, top_k: int, **serach_param) -> list[int]:
        raise NotImplementedError