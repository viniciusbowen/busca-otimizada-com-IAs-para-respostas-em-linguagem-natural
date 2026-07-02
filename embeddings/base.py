"""Interface comum para geradores de embeddings.

Definir uma interface única permite trocar a técnica de vetorização
(Word2Vec médio, Sentence Embeddings, etc.) sem alterar o restante do
pipeline, e facilita a comparação de desempenho entre elas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class BaseEmbedder(ABC):
    """Classe base abstrata para geradores de embeddings.

    Atributos esperados nas implementações:
    - ``name``: identificador legível da técnica (para relatórios/benchmark).
    """

    name: str = "base"

    @abstractmethod
    def fit(self, corpus: Iterable[str]) -> "BaseEmbedder":
        """Treina/ajusta o embedder ao corpus, quando aplicável.

        Modelos pré-treinados (ex.: Sentence Transformers) podem apenas
        carregar os pesos; modelos como Word2Vec são treinados aqui.

        Args:
            corpus: Iterável de documentos de texto.

        Returns:
            A própria instância (para encadeamento).
        """
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Gera o vetor de embedding de um único texto.

        Args:
            text: Texto de entrada.

        Returns:
            Vetor 1-D de dimensão :attr:`dim`.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Gera embeddings para um lote de textos.

        Args:
            texts: Iterável de textos.

        Returns:
            Matriz 2-D (n_textos x :attr:`dim`).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensão dos vetores gerados."""
        raise NotImplementedError
