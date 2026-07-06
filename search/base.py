"""Interface comum para os mecanismos de busca semântica.

Todas as técnicas (cosseno exaustivo, FAISS, HNSW) implementam a mesma
interface para que possam ser trocadas no pipeline e comparadas de forma
justa no benchmark (tempo, complexidade e qualidade).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    """Um resultado individual da busca.

    Attributes:
        index: Índice do documento no corpus original.
        score: Pontuação de similaridade (quanto maior, mais similar).
        payload: Metadados opcionais do documento (título, sinopse, etc.).
    """

    index: int
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class BaseSearcher(ABC):
    """Classe base abstrata para índices de busca vetorial."""

    name: str = "base"

    @property
    @abstractmethod
    def complexity(self) -> str:
        """Descrição da complexidade da busca
        """
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "BaseSearcher":
        """Constrói o índice a partir da matriz de embeddings do corpus.

        Args:
            vectors: Matriz 2-D (N x d) com os embeddings dos documentos.
            metadata: Metadados por documento (mesma ordem de ``vectors``).

        Returns:
            A própria instância (para encadeamento).
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Retorna os ``top_k`` documentos mais similares à consulta.

        Args:
            query_vector: Vetor 1-D de embedding da consulta.
            top_k: Número de resultados a retornar.

        Returns:
            Lista de :class:`SearchResult` ordenada por similaridade decrescente.
        """
        raise NotImplementedError

    def save(self, path) -> None:
        """Persiste o índice em disco (opcional)."""
        raise NotImplementedError("Persistir o índice em disco.")

    def load(self, path) -> "BaseSearcher":
        """Carrega o índice do disco (opcional)."""
        raise NotImplementedError("Carregar o índice do disco.")
