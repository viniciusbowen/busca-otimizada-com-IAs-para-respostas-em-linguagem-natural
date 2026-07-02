"""Embeddings via média de vetores Word2Vec (Gensim).

Estratégia "Word2Vec Average": treina (ou carrega) um modelo Word2Vec e
representa cada documento pela média dos vetores das suas palavras.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from tf import config
from tf.embeddings.base import BaseEmbedder


class Word2VecAverageEmbedder(BaseEmbedder):
    """Representa textos pela média dos vetores Word2Vec das palavras."""

    name = "word2vec_avg"

    def __init__(
        self,
        vector_size: int = config.WORD2VEC_VECTOR_SIZE,
        window: int = config.WORD2VEC_WINDOW,
        min_count: int = config.WORD2VEC_MIN_COUNT,
    ) -> None:
        """Configura os hiperparâmetros do Word2Vec.

        Args:
            vector_size: Dimensão dos vetores de palavra.
            window: Tamanho da janela de contexto.
            min_count: Frequência mínima para uma palavra entrar no vocabulário.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self._model = None

    def fit(self, corpus: Iterable[str]) -> "Word2VecAverageEmbedder":
        """Treina o modelo Word2Vec sobre o corpus tokenizado.

        Complexidade: O(épocas * N * janela) no treinamento.
        """
        raise NotImplementedError("Treinar Word2Vec com Gensim.")

    def embed_text(self, text: str) -> np.ndarray:
        """Gera o embedding médio de um texto.

        Complexidade: O(m) no número de palavras do texto.
        """
        raise NotImplementedError("Calcular a média dos vetores das palavras.")

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Gera embeddings médios para um lote de textos."""
        raise NotImplementedError("Gerar embeddings médios em lote.")

    @property
    def dim(self) -> int:
        return self.vector_size
