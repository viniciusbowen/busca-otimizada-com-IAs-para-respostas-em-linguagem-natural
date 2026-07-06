"""Pré-processamento de texto para as sinopses e consultas.

Usa o NLTK para normalização, tokenização e remoção de stopwords. O mesmo
pipeline deve ser aplicado tanto às sinopses (na indexação) quanto às
perguntas do usuário (na consulta), garantindo consistência.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from busca import config


def normalize(text: str) -> str:
    """Normaliza um texto (minúsculas, remoção de pontuação/acentos, etc.).

    Args:
        text: Texto bruto.

    Returns:
        Texto normalizado.

    Complexidade: O(n) no comprimento do texto.
    """
    raise NotImplementedError("Normalizar o texto.")


def tokenize(text: str) -> list[str]:
    """Divide o texto em tokens (NLTK ``word_tokenize``).

    Args:
        text: Texto (idealmente já normalizado).

    Returns:
        Lista de tokens.

    Complexidade: O(n) no comprimento do texto.
    """
    raise NotImplementedError("Tokenizar o texto com NLTK.")


def remove_stopwords(tokens: Iterable[str], language: str = config.LANGUAGE) -> list[str]:
    """Remove stopwords da lista de tokens.

    Args:
        tokens: Tokens de entrada.
        language: Idioma das stopwords do NLTK.

    Returns:
        Tokens sem stopwords.

    Complexidade: O(n) com lookup O(1) em conjunto de stopwords.
    """
    raise NotImplementedError("Remover stopwords com NLTK.")


def preprocess_text(text: str) -> list[str]:
    """Aplica o pipeline completo a um único texto.

    Encadeia :func:`normalize`, :func:`tokenize` e :func:`remove_stopwords`.

    Args:
        text: Texto bruto.

    Returns:
        Lista de tokens limpos.
    """
    raise NotImplementedError("Aplicar normalize + tokenize + remove_stopwords.")


def preprocess_corpus(corpus: pd.DataFrame, text_column: str = "plot") -> pd.DataFrame:
    """Aplica o pré-processamento à coluna de texto do corpus inteiro.

    Args:
        corpus: DataFrame do corpus (ver :mod:`busca.data.loader`).
        text_column: Nome da coluna com o texto a processar.

    Returns:
        DataFrame com uma coluna adicional de tokens pré-processados.

    Complexidade: O(N * m), N documentos de comprimento médio m.
    """
    raise NotImplementedError("Pré-processar a coluna de texto do corpus.")
