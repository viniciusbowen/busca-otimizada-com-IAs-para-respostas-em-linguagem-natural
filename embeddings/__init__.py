"""Módulos de geração de embeddings do projeto."""

from embeddings.base import BaseEmbedder
from embeddings.sentence_embed import SentenceEmbedder

__all__ = ["BaseEmbedder", "SentenceEmbedder"]
