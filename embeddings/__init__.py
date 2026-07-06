"""Módulos de geração de embeddings (TF-IDF, Word2Vec avg e Sentence Transformers)."""

from busca.embeddings.base import BaseEmbedder
from busca.embeddings.sentence_embed import SentenceEmbedder
from busca.embeddings.tfidf_embed import TfidfEmbedder
from busca.embeddings.word2vec_avg import Word2VecAverageEmbedder

__all__ = [
    "BaseEmbedder",
    "SentenceEmbedder",
    "TfidfEmbedder",
    "Word2VecAverageEmbedder",
]
