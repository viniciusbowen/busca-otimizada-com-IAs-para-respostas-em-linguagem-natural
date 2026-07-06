"""Embeddings via média de vetores Word2Vec (Gensim).

Estratégia "Word2Vec Average": treina (ou carrega) um modelo Word2Vec e
representa cada documento pela média dos vetores das suas palavras.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from gensim.models import Word2Vec

from busca import config
from busca.embeddings.base import BaseEmbedder


class Word2VecAverageEmbedder(BaseEmbedder):
    """Representa textos pela média dos vetores Word2Vec das palavras."""

    name = "word2vec_avg"
    complexity = "embed: O(m·d) por texto; treino: O(épocas·N·janela·d)"

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
    
        sentences = [text.lower().split() for text in corpus]

        #criar e treinar o modelo Word2Vec com Gensim
        self._model = Word2Vec(
                                sentences=sentences,
                                vector_size=self.vector_size,
                                window=self.window,
                                min_count=self.min_count,
                                )
        
        return self
        
        #raise NotImplementedError("Treinar Word2Vec com Gensim.")

    def embed_text(self, text: str) -> np.ndarray:
        """Gera o embedding médio de um texto.

        Complexidade: O(m) no número de palavras do texto.
        """

        if self._model is None:
            raise RuntimeError("Moldelo deve ser treinado antes de gerar 'embeddings'.")

        #tokeniza
        tokens = text.lower().split()

        #cria vetor
        vectors = [self._model.wv[token] for token in tokens if token in self._model.wv]
        
        #caso vazio
        if not vectors:
            return np.zeros(self.vector_size)
        
        #media é suficiente para Word2Vec
        return np.mean(vectors, axis=0)
        
        #raise NotImplementedError("Calcular a média dos vetores das palavras.")

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Gera embeddings médios para um lote de textos."""

        embeddings = [self.embed_text(text) for text in texts]

        return np.vstack(embeddings)
        #raise NotImplementedError("Gerar embeddings médios em lote.")

    @property
    def dim(self) -> int:
        return self.vector_size
