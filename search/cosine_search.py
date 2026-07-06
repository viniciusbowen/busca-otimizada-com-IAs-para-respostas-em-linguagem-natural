"""Busca exaustiva por similaridade de cosseno.

Baseline simples: compara o vetor da consulta com todos os vetores do
corpus. Serve de referência de qualidade e de custo para as buscas
aproximadas (FAISS/HNSW).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tf.search.base import BaseSearcher, SearchResult


class CosineSearcher(BaseSearcher):
    """Busca exaustiva usando similaridade de cosseno (scikit-learn/NumPy)."""

    name = "cosine"

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] | None = None

    @property
    def complexity(self) -> str:
        return "O(N*d) por consulta"

    def build(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "CosineSearcher":
        """Armazena os vetores (opcionalmente normalizados) e metadados.

        Complexidade: O(N*d) para normalização, O(1) sem ela.
        """
        self._vectors = np.asarray(vectors, dtype=np.float32)
        if self._vectors.ndim != 2:
            raise ValueError(
                "`vectors` deve ser uma matriz 2-D (N x d); "
                f"recebido ndim={self._vectors.ndim}."
            )
        self._metadata = metadata
        return self

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Calcula a similaridade de cosseno com todo o corpus e ordena.

        Complexidade: O(N*d) para o produto + O(N log k) para o top-k.
        """
        if self._vectors is None:
            raise RuntimeError("Índice não construído: chame `build` antes de `search`.")

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        # Mesma mecânica do projeto de referência (TF-IDF-Analise-Recomedacao):
        # similaridade de cosseno da consulta contra todos os vetores do corpus.
        scores = cosine_similarity(query, self._vectors).flatten()

        # Seleção eficiente do top-k: particiona em O(N) e ordena apenas os k.
        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []
        partitioned = np.argpartition(-scores, k - 1)[:k]
        top_indices = partitioned[np.argsort(-scores[partitioned])]

        results: list[SearchResult] = []
        for idx in top_indices:
            i = int(idx)
            payload = self._metadata[i] if self._metadata is not None else {}
            results.append(SearchResult(index=i, score=float(scores[i]), payload=payload))
        return results
