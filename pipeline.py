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

from tf import config
from tf.data.loader import load_corpus
from tf.embeddings.base import BaseEmbedder
from tf.llm.generator import LocalLLM
from tf.search.base import BaseSearcher, SearchResult


@dataclass
class PipelineResponse:
    """Resposta completa do pipeline para uma pergunta.

    Attributes:
        question: Pergunta original do usuário.
        results: Sinopses recuperadas pela busca.
        answer: Resposta em linguagem natural gerada pelo LLM.
        elapsed: Tempos por etapa (embedding, busca, geração), em segundos.
    """

    question: str
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

    def search(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        """Recupera as sinopses mais relevantes para a pergunta.

        Args:
            question: Pergunta em linguagem natural.
            top_k: Sobrescreve o ``top_k`` padrão, se informado.

        Returns:
            Lista de :class:`SearchResult`.
        """
        if self._corpus is None:
            raise RuntimeError("Pipeline não construído: chame `build` antes de `search`.")

        k = top_k or self.top_k
        query_vector = self.embedder.embed_text(question)
        return self.searcher.search(query_vector, k)

    def answer(self, question: str, top_k: int | None = None) -> PipelineResponse:
        """Executa o fluxo completo e retorna a resposta formatada.

        Args:
            question: Pergunta do usuário.
            top_k: Número de sinopses a recuperar.

        Returns:
            :class:`PipelineResponse` com resultados, resposta e tempos.
        """
        t0 = time.perf_counter()
        results = self.search(question, top_k)
        t1 = time.perf_counter()

        answer_text = ""
        if self.llm is not None:
            retrieved_docs = [result.payload for result in results]
            answer_text = self.llm.generate_answer(question, retrieved_docs)
        t2 = time.perf_counter()

        return PipelineResponse(
            question=question,
            results=results,
            answer=answer_text,
            elapsed={"busca": t1 - t0, "geracao": t2 - t1},
        )
