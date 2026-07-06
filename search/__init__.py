"""Módulos de busca semântica (cosseno exaustivo, FAISS e HNSW)."""

from busca.search.base import BaseSearcher, SearchResult
from busca.search.cosine_search import CosineSearcher
from busca.search.faiss_search import FaissSearcher
from busca.search.hnsw_search import HNSWSearcher

__all__ = [
    "BaseSearcher",
    "SearchResult",
    "CosineSearcher",
    "FaissSearcher",
    "HNSWSearcher",
]
