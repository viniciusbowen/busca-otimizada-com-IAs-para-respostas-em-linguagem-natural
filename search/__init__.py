"""Módulos de busca semântica (cosseno exaustivo, FAISS e HNSW)."""

from tf.search.base import BaseSearcher, SearchResult
from tf.search.cosine_search import CosineSearcher
from tf.search.faiss_search import FaissSearcher
from tf.search.hnsw_search import HNSWSearcher

__all__ = [
    "BaseSearcher",
    "SearchResult",
    "CosineSearcher",
    "FaissSearcher",
    "HNSWSearcher",
]
