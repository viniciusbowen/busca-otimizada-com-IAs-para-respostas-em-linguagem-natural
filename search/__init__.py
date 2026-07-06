"""Módulos de busca semântica usados pela integração atual."""

from search.base import BaseSearcher, SearchResult
from search.cosine_search import CosineSearcher

__all__ = ["BaseSearcher", "SearchResult", "CosineSearcher"]
