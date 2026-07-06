"""Módulos de pré-processamento textual (NLTK)."""

from busca.preprocessing.text_cleaner import (
    normalize,
    preprocess_corpus,
    remove_stopwords,
    tokenize,
)

__all__ = ["normalize", "tokenize", "remove_stopwords", "preprocess_corpus"]
