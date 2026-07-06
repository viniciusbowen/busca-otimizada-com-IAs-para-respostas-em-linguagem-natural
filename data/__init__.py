"""Módulos de acesso e preparação do dataset CMU Movie Summary Corpus."""

from busca.data.download import download_dataset, ensure_dataset
from busca.data.loader import (
    build_corpus,
    load_corpus,
    load_metadata,
    load_plot_summaries,
)

__all__ = [
    "ensure_dataset",
    "download_dataset",
    "build_corpus",
    "load_corpus",
    "load_metadata",
    "load_plot_summaries",
]
