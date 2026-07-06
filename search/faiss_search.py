"""Busca vetorial com FAISS.

Usa um índice FAISS para acelerar a busca de vizinhos mais próximos. Pode
usar índice plano (exato) ou índices aproximados (IVF, HNSW do FAISS).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from busca import config
from busca.search.base import BaseSearcher, SearchResult


class FaissSearcher(BaseSearcher):
    """Busca de vizinhos mais próximos usando FAISS."""

    name = "faiss"

    def __init__(self, metric: str = config.DISTANCE_METRIC) -> None:
        """Configura o índice FAISS.

        Args:
            metric: Métrica de distância ("cosine", "l2", "ip").
        """
        self.metric = metric
        self._index = None
        self._metadata: list[dict[str, Any]] | None = None

    @property
    def complexity(self) -> str:
        return "O(N*d) (IndexFlat) ou sublinear (IVF/HNSW)"

    def build(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "FaissSearcher":
        """Cria o índice FAISS e adiciona os vetores do corpus.

        Complexidade: O(N*d) para IndexFlat; treino adicional para IVF.
        """
        raise NotImplementedError("Construir o índice FAISS e adicionar vetores.")

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Consulta o índice FAISS pelos ``top_k`` vizinhos.

        Complexidade: depende do tipo de índice (exato vs aproximado).
        """
        raise NotImplementedError("Consultar o índice FAISS.")

    def save(self, path=config.INDEX_DIR) -> None:
        raise NotImplementedError("Persistir o índice FAISS (faiss.write_index).")

    def load(self, path=config.INDEX_DIR) -> "FaissSearcher":
        raise NotImplementedError("Carregar o índice FAISS (faiss.read_index).")
