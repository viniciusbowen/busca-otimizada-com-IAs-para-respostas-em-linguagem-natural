"""Busca exaustiva por similaridade de cosseno.

Baseline simples: compara o vetor da consulta com todos os vetores do
corpus. Serve de referência de qualidade e de custo para as buscas
aproximadas (FAISS/HNSW).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from base import BaseSearcher, SearchResult

EPS = 1e-12


class CosineSearcher(BaseSearcher):
    """Busca exaustiva usando similaridade de cosseno (scikit-learn/NumPy)."""

    name = "cosine"

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] | None = None
        self._n_comparisons: int = 0 

    @property
    def complexity(self) -> str:
        return "O(N*d) por consulta"
    
    @property
    def n_comparisons(self) -> int:
        return self._n_comparisons

    def build(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "CosineSearcher":
        """Armazena os vetores (opcionalmente normalizados) e metadados.

        Complexidade: O(N*d) para normalização, O(1) sem ela.
        """
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                "`vectors` deve ser uma matriz 2-D (N x d); "
                f"recebido ndim={arr.ndim}."
            )
 
        norms = np.linalg.norm(arr, axis=1, keepdims=True)  # O(N*d)
        norms = np.maximum(norms, EPS)                      # protege vetores nulos
        self._vectors = arr / norms                          # O(N*d)
        self._metadata = metadata
        self._n_comparisons = 0
        return self

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Calcula a similaridade de cosseno da consulta contra todo o corpus.
 
        Passos e custo:
            1. Normalização da consulta:              O(d)
            2. Produto escalar consulta x corpus:      O(N*d)  <- domina o custo
            3. Seleção do top-k (argpartition):        O(N)
            4. Ordenação apenas dos k selecionados:     O(k log k)
 
        Total por consulta: O(N*d).
        """
        if self._vectors is None:
            raise RuntimeError("Índice não construído: chame `build` antes de `search`.")
 
        query = np.asarray(query_vector, dtype=np.float64).reshape(-1)
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"Dimensão da consulta ({query.shape[0]}) difere da "
                f"dimensão do corpus ({self._vectors.shape[1]})."
            )
 
        q_norm = np.linalg.norm(query)  # O(d)
        if q_norm < EPS:
            scores = np.zeros(self._vectors.shape[0], dtype=np.float64)
        else:
            q_unit = query / q_norm            # O(d)
            scores = self._vectors @ q_unit     # O(N*d) -- custo dominante
 
        self._n_comparisons += self._vectors.shape[0]
 
        n = scores.shape[0]
        k = min(top_k, n)
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
