"""Módulos de acesso e preparação do dataset CMU Movie Summary Corpus."""

from tf.data.download import download_dataset, ensure_dataset
from tf.data.loader import (
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
