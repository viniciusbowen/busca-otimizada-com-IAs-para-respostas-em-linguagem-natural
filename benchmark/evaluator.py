"""Comparação das técnicas de busca semântica.

Reúne as métricas exigidas pelo projeto para comparar as abordagens
(cosseno, Word2Vec avg, Sentence Embeddings, FAISS, HNSW):
- Complexidade computacional (teórica, vinda de cada searcher).
- Tempo de execução (medido empiricamente).
- Qualidade dos resultados (Precision@K, Recall@K, MRR).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from tf.search.base import BaseSearcher, SearchResult


def precision_at_k(results: Sequence[SearchResult], relevant: set[int], k: int) -> float:
    """Precision@K: fração dos top-k resultados que são relevantes.

    Complexidade: O(k).
    """
    raise NotImplementedError("Calcular Precision@K.")


def recall_at_k(results: Sequence[SearchResult], relevant: set[int], k: int) -> float:
    """Recall@K: fração dos itens relevantes recuperados no top-k.

    Complexidade: O(k).
    """
    raise NotImplementedError("Calcular Recall@K.")


def reciprocal_rank(results: Sequence[SearchResult], relevant: set[int]) -> float:
    """Reciprocal Rank: 1/posição do primeiro resultado relevante.

    Complexidade: O(len(results)).
    """
    raise NotImplementedError("Calcular o reciprocal rank.")


def measure_query_time(searcher: BaseSearcher, query_vector, top_k: int, repeats: int = 1) -> float:
    """Mede o tempo médio de execução de uma consulta.

    Args:
        searcher: Índice de busca já construído.
        query_vector: Vetor da consulta.
        top_k: Número de resultados.
        repeats: Repetições para estabilizar a medição.

    Returns:
        Tempo médio por consulta, em segundos.
    """
    raise NotImplementedError("Medir o tempo de execução da consulta.")


def benchmark_searchers(
    searchers: Sequence[BaseSearcher],
    query_vectors: Sequence,
    ground_truth: Sequence[set[int]],
    top_k: int,
) -> pd.DataFrame:
    """Executa o benchmark completo sobre um conjunto de searchers.

    Para cada searcher e cada consulta, coleta tempo e métricas de qualidade,
    agrega os resultados e anexa a coluna de complexidade teórica de cada
    técnica. Serve de base para os itens 6-12 da apresentação.

    Args:
        searchers: Lista de índices de busca já construídos.
        query_vectors: Vetores das consultas de avaliação.
        ground_truth: Conjunto de índices relevantes para cada consulta.
        top_k: Valor de K para as métricas.

    Returns:
        DataFrame com uma linha por técnica e colunas de tempo, métricas e
        complexidade.
    """
    raise NotImplementedError("Executar o benchmark e agregar os resultados.")
