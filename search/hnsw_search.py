"""Busca aproximada de vizinhos com HNSW (hnswlib).

HNSW (Hierarchical Navigable Small World) é um grafo hierárquico que
permite busca aproximada de vizinhos em tempo sublinear, tipicamente
~O(log N), com alta taxa de recall.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tf import config
from tf.search.base import BaseSearcher, SearchResult


class HNSWSearcher(BaseSearcher):
    """Busca aproximada de vizinhos usando hnswlib."""

    name = "hnsw"

    def __init__(
        self,
        metric: str = config.DISTANCE_METRIC,
        m: int = config.HNSW_M,
        ef_construction: int = config.HNSW_EF_CONSTRUCTION,
        ef_search: int = config.HNSW_EF_SEARCH,
    ) -> None:
        """Configura os hiperparâmetros do índice HNSW.

        Args:
            metric: Espaço de distância ("cosine", "l2", "ip").
            m: Número de conexões por nó no grafo.
            ef_construction: Tamanho da lista dinâmica na construção.
            ef_search: Tamanho da lista dinâmica na busca (recall vs velocidade).
        """
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index = None
        self._metadata: list[dict[str, Any]] | None = None

    @property
    def complexity(self) -> str:
        return "~O(log N) por consulta; O(N log N) para construir"

    def build(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "HNSWSearcher":
        """Inicializa e popula o índice HNSW com os vetores do corpus.

        Complexidade: ~O(N log N) para a construção do grafo.
        """
        raise NotImplementedError("Construir o índice HNSW e adicionar vetores.")

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Consulta o índice HNSW pelos ``top_k`` vizinhos aproximados.

        Complexidade: ~O(log N) por consulta.
        """
        raise NotImplementedError("Consultar o índice HNSW (knn_query).")

    def save(self, path=config.INDEX_DIR) -> None:
        raise NotImplementedError("Persistir o índice HNSW (save_index).")

    def load(self, path=config.INDEX_DIR) -> "HNSWSearcher":
        raise NotImplementedError("Carregar o índice HNSW (load_index).")
