"""Orquestração do fluxo de busca semântica ponta a ponta.

Une os componentes: carrega o corpus, pré-processa, gera embeddings,
constrói o índice de busca escolhido e, dada uma pergunta, recupera as
sinopses mais relevantes e gera a resposta final com o LLM local.

Fluxo:
    pergunta -> embedding -> busca (top-k) -> LLM -> resposta formatada
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import config
from data.loader import load_corpus
from embeddings.base import BaseEmbedder
from llm.generator import LocalLLM
from search.base import BaseSearcher, SearchResult


@dataclass
class PipelineResponse:
    """Resposta completa do pipeline para uma pergunta.

    Attributes:
        question: Pergunta original do usuário.
        search_query: Consulta de busca usada (possivelmente reescrita pelo LLM).
        results: Sinopses recuperadas pela busca.
        answer: Resposta em linguagem natural gerada pelo LLM.
        elapsed: Tempos por etapa (reescrita, busca, geração), em segundos.
    """

    question: str
    search_query: str = ""
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""
    elapsed: dict[str, float] = field(default_factory=dict)


class SemanticSearchPipeline:
    """Encapsula o fluxo completo de busca semântica + geração de resposta."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        searcher: BaseSearcher,
        llm: LocalLLM | None = None,
        top_k: int = config.TOP_K,
    ) -> None:
        """Configura o pipeline com os componentes escolhidos.

        Args:
            embedder: Estratégia de embeddings (Word2Vec avg / Sentence).
            searcher: Mecanismo de busca (cosseno / FAISS / HNSW).
            llm: LLM local para gerar a resposta (opcional).
            top_k: Número de sinopses a recuperar.
        """
        self.embedder = embedder
        self.searcher = searcher
        self.llm = llm
        self.top_k = top_k
        self._corpus = None

    def build(
        self,
        corpus: pd.DataFrame | None = None,
        max_docs: int | None = config.MAX_CORPUS_DOCS,
    ) -> "SemanticSearchPipeline":
        """Prepara o pipeline: corpus -> embeddings -> índice.

        Passos:
        1. Carregar/receber o corpus (opcionalmente limitado a ``max_docs``).
        2. Ajustar o embedder às sinopses e gerar os embeddings do corpus.
        3. Construir o índice de busca com os metadados por documento.

        Args:
            corpus: DataFrame do corpus; se ``None``, carrega via loader.
            max_docs: Limite de documentos (para viabilizar TF-IDF denso em
                memória); ``None`` usa o corpus inteiro.

        Returns:
            A própria instância.
        """
        if corpus is None:
            corpus = load_corpus()

        if max_docs is not None and len(corpus) > max_docs:
            corpus = corpus.head(max_docs).reset_index(drop=True)

        self._corpus = corpus

        texts = corpus["plot"].fillna("").astype(str).tolist()
        # Metadados por documento (título/sinopse/gêneros) para o payload e o LLM.
        metadata = corpus.to_dict("records")

        self.embedder.fit(texts)
        vectors = self.embedder.embed_batch(texts)
        self.searcher.build(vectors, metadata)

        return self

    def rewrite_query(self, question: str) -> str:
        """Reescreve a pergunta como consulta de busca (em inglês) via LLM.

        Se não houver LLM configurado, retorna a pergunta original.

        Args:
            question: Pergunta em linguagem natural.

        Returns:
            Consulta usada na busca vetorial.
        """
        if self.llm is None:
            return question
        return self.llm.rewrite_query(question)

    def search(
        self,
        question: str,
        top_k: int | None = None,
        rewrite: bool = True,
    ) -> list[SearchResult]:
        """Recupera as sinopses mais relevantes para a pergunta.

        Args:
            question: Pergunta em linguagem natural.
            top_k: Sobrescreve o ``top_k`` padrão, se informado.
            rewrite: Se ``True`` e houver LLM, reescreve a pergunta como
                consulta de busca em inglês antes de gerar o embedding.

        Returns:
            Lista de :class:`SearchResult`.
        """
        if self._corpus is None:
            raise RuntimeError("Pipeline não construído: chame `build` antes de `search`.")

        k = top_k or self.top_k
        search_query = self.rewrite_query(question) if rewrite else question
        query_vector = self.embedder.embed_text(search_query)
        return self.searcher.search(query_vector, k)

    def answer(self, question: str, top_k: int | None = None) -> PipelineResponse:
        """Executa o fluxo completo e retorna a resposta formatada.

        Etapas:
        1. Reescrita: o LLM transforma a pergunta em uma consulta de busca
           (idealmente em inglês, para casar com o corpus).
        2. Busca: embedding da consulta + recuperação dos ``top_k`` documentos.
        3. Geração: o LLM formata a resposta final a partir do contexto.

        Args:
            question: Pergunta do usuário.
            top_k: Número de sinopses a recuperar.

        Returns:
            :class:`PipelineResponse` com a consulta, resultados, resposta e tempos.
        """
        if self._corpus is None:
            raise RuntimeError("Pipeline não construído: chame `build` antes de `answer`.")

        k = top_k or self.top_k

        t0 = time.perf_counter()
        search_query = self.rewrite_query(question)
        t1 = time.perf_counter()

        query_vector = self.embedder.embed_text(search_query)
        results = self.searcher.search(query_vector, k)
        t2 = time.perf_counter()

        answer_text = ""
        if self.llm is not None:
            retrieved_docs = [result.payload for result in results]
            answer_text = self.llm.generate_answer(question, retrieved_docs)
        t3 = time.perf_counter()

        return PipelineResponse(
            question=question,
            search_query=search_query,
            results=results,
            answer=answer_text,
            elapsed={
                "reescrita": t1 - t0,
                "busca": t2 - t1,
                "geracao": t3 - t2,
            },
        )
