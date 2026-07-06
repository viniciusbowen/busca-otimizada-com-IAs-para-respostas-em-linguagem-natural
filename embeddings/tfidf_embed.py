"""Embeddings via TF-IDF (scikit-learn).

Estratégia "TF-IDF": representa cada documento por um vetor esparso em que
cada dimensão é um termo do vocabulário, ponderado por *Term Frequency -
Inverse Document Frequency*. É a mesma técnica usada no projeto de referência
``TF-IDF-Analise-Recomedacao`` (``TfidfVectorizer`` + similaridade de cosseno).

Diferente do Word2Vec/Sentence Transformers, o TF-IDF não captura semântica
por contexto: a similaridade vem da sobreposição de termos entre a consulta e
os documentos. Ainda assim, é um baseline forte e barato de calcular.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from busca import config
from busca.embeddings.base import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    """Representa textos por vetores TF-IDF densificados."""

    name = "tfidf"

    def __init__(
        self,
        max_features: int | None = config.TFIDF_MAX_FEATURES,
        min_df: int = config.TFIDF_MIN_DF,
        use_stopwords: bool = config.TFIDF_USE_STOPWORDS,
        language: str = config.LANGUAGE,
    ) -> None:
        """Configura os hiperparâmetros do ``TfidfVectorizer``.

        Args:
            max_features: Tamanho máximo do vocabulário (``None`` = todos).
            min_df: Frequência mínima de documentos para reter um termo.
            use_stopwords: Se ``True``, remove stopwords do ``language``.
            language: Idioma das stopwords (aceito pelo scikit-learn: ``"english"``).
        """
        self.max_features = max_features
        self.min_df = min_df
        self.use_stopwords = use_stopwords
        self.language = language
        self._vectorizer: TfidfVectorizer | None = None
        self._dim: int | None = None

    def fit(self, corpus: Iterable[str]) -> "TfidfEmbedder":
        """Ajusta o vocabulário e os pesos IDF ao corpus.

        Espelha ``tfidf.fit_transform(...)`` do projeto de referência: aprende o
        vocabulário e as ponderações IDF a partir de todos os documentos.

        Args:
            corpus: Iterável de documentos de texto (strings brutas).

        Returns:
            A própria instância.

        Complexidade: O(N * m) no total de tokens do corpus.
        """
        documents = [doc if isinstance(doc, str) else "" for doc in corpus]

        stop_words = self.language if self.use_stopwords else None
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            stop_words=stop_words,
        )
        self._vectorizer.fit(documents)
        self._dim = len(self._vectorizer.vocabulary_)
        return self

    def _ensure_fitted(self) -> None:
        if self._vectorizer is None:
            raise RuntimeError("Embedder não ajustado; chame fit() primeiro.")

    def embed_text(self, text: str) -> np.ndarray:
        """Gera o vetor TF-IDF (denso) de um único texto.

        Complexidade: O(m) no número de termos do texto.
        """
        self._ensure_fitted()
        matrix = self._vectorizer.transform([text])
        return matrix.toarray()[0].astype(np.float32)

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Gera a matriz TF-IDF (densa) para um lote de textos.

        Complexidade: O(N * m) no total de termos do lote.
        """
        self._ensure_fitted()
        documents = [doc if isinstance(doc, str) else "" for doc in texts]
        matrix = self._vectorizer.transform(documents)
        return matrix.toarray().astype(np.float32)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Embedder ainda não ajustado; chame fit() primeiro.")
        return self._dim
