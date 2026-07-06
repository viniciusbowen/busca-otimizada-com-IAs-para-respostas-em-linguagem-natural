from __future__ import annotations

from typing import Any

import numpy as np

from busca.search.base import BaseSearcher, SearchResult

EPS = 1e-12


# a similaridade de cosseno entre dois vetores u e v é:
#
#       cos(u, v) = (u . v) / (||u|| * ||v||)
#
# normalizando os vetores antes da busca : (||u|| = ||v|| = 1)
# o cálculo se reduz ao produto escalar, o
# que torna mais eficiente para a busca em um mesmo banco

# analise de complexidade:
#   1. Normalização da consulta: O(d)
#   2. Produto escalar consulta x corpus: O(N*d)  
#   3. Seleção do top-k (argpartition): O(N)
#   4. Ordenação apenas dos k selecionados: O(k log k)

#   Total por consulta: O(N*d).

class CosineSearcher(BaseSearcher):

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
        
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                "`vectors` deve ser uma matriz 2-D (N x d); "
                f"recebido ndim={arr.ndim}."
            )
 
        norma = np.linalg.norm(arr, axis=1, keepdims=True)  # O(N*d)
        norma = np.maximum(norma, EPS)                      # protege vetores nulos
        self._vectors = arr /norma                          # O(N*d)
        self._metadata = metadata
        self._n_comparisons = 0
        return self

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
 
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
            scores = self._vectors @ q_unit     # O(N*d)
 
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


if __name__ == "__main__":

    buscador = CosineSearcher()

    print("=" * 60)
    print("TESTE 1 - Vetor idêntico")
    print("=" * 60)

    corpus = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=float)

    consulta = np.array([1, 0], dtype=float)

    buscador.build(corpus)
    resultados = buscador.search(consulta, top_k=3)

    for r in resultados:
        print(f"Índice: {r.index} | Similaridade: {r.score:.4f}")

    print("\nEsperado aproximadamente:")
    print("Índice 0 -> 1.0000")
    print("Índice 2 -> 0.7071")
    print("Índice 1 -> 0.0000")


    print("\n" + "=" * 60)
    print("TESTE 2 - Vetor oposto")
    print("=" * 60)

    corpus = np.array([
        [-1, 0],
        [1, 0]
    ], dtype=float)

    consulta = np.array([1, 0], dtype=float)

    buscador.build(corpus)
    resultados = buscador.search(consulta, top_k=2)

    for r in resultados:
        print(f"Índice: {r.index} | Similaridade: {r.score:.4f}")

    print("\nEsperado aproximadamente:")
    print("Índice 1 -> 1.0000")
    print("Índice 0 -> -1.0000")


    print("\n" + "=" * 60)
    print("TESTE 3 - Vetores perpendiculares")
    print("=" * 60)

    corpus = np.array([
        [0, 1],
        [1, 0]
    ], dtype=float)

    consulta = np.array([1, 0], dtype=float)

    buscador.build(corpus)
    resultados = buscador.search(consulta, top_k=2)

    for r in resultados:
        print(f"Índice: {r.index} | Similaridade: {r.score:.4f}")

    print("\nEsperado aproximadamente:")
    print("Índice 1 -> 1.0000")
    print("Índice 0 -> 0.0000")


    print("\n" + "=" * 60)
    print("TESTE 4 - Ordenação")
    print("=" * 60)

    corpus = np.array([
        [1, 0],
        [0.9, 0.1],
        [0, 1]
    ], dtype=float)

    consulta = np.array([1, 0], dtype=float)

    buscador.build(corpus)
    resultados = buscador.search(consulta, top_k=3)

    for r in resultados:
        print(f"Índice: {r.index} | Similaridade: {r.score:.4f}")

    print("\nEsperado aproximadamente:")
    print("Índice 0 -> 1.0000")
    print("Índice 1 -> 0.9939")
    print("Índice 2 -> 0.0000")