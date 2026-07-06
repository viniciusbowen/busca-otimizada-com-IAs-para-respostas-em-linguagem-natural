"""Embeddings via Sentence Transformers.

Estratégia "Sentence Embeddings": usa um modelo pré-treinado que mapeia
sentenças inteiras para vetores densos, capturando melhor a semântica do
que a simples média de vetores de palavras.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from config import SENTENCE_MODEL_NAME
from embeddings.base import BaseEmbedder


class SentenceEmbedder(BaseEmbedder):
    """Gera embeddings de sentenças com Sentence Transformers."""

    name = "sentence_embeddings"

    def __init__(self, model_name: str = SENTENCE_MODEL_NAME) -> None:
        """Configura o modelo de embeddings.

        Args:
            model_name: Nome do modelo no Hugging Face / Sentence Transformers.
        """
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    def fit(self, corpus: Iterable[str] | None = None) -> "SentenceEmbedder":
        """Carrega o modelo pré-treinado (não há treinamento próprio).

        Args:
            corpus: Ignorado; presente apenas por compatibilidade de interface.

        Returns:
            A própria instância.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Dependência opcional ausente: 'sentence-transformers'. Instale com `pip install sentence-transformers`."
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        return self

    def embed_text(self, text: str) -> np.ndarray:
        """Gera o embedding de um único texto.

        Complexidade: O(m) no comprimento do texto (custo do forward do modelo).
        """
        if self._model is None:
            raise RuntimeError("Modelo não carregado; chame fit() primeiro.")

        embedding = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embedding[0], dtype=np.float32)

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Gera embeddings para um lote de textos (encode em batch)."""
        if self._model is None:
            raise RuntimeError("Modelo não carregado; chame fit() primeiro.")

        documents = list(texts)
        embeddings = self._model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Modelo ainda não carregado; chame fit() primeiro.")
        return self._dim
