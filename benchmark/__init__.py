"""Módulo de benchmark das técnicas de busca semântica."""

from busca.benchmark.evaluator import (
    benchmark_searchers,
    measure_query_time,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "benchmark_searchers",
    "measure_query_time",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
