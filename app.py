"""Interface web (Streamlit) do sistema de busca semântica de filmes.

Permite ao usuário: digitar uma pergunta em linguagem natural, escolher a
técnica de embedding e de busca, visualizar as sinopses recuperadas e a
resposta gerada pelo LLM local, além de rodar o benchmark comparativo.

Execução:
    streamlit run tf/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Garante que a raiz do repositório esteja no sys.path para permitir
# `from tf import ...` ao rodar via `streamlit run tf/app.py` (o Streamlit
# coloca apenas a pasta do script, tf/, no path por padrão).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from tf import config
from tf.embeddings.sentence_embed import SentenceEmbedder
from tf.embeddings.tfidf_embed import TfidfEmbedder
from tf.embeddings.word2vec_avg import Word2VecAverageEmbedder
from tf.llm.generator import LocalLLM
from tf.pipeline import SemanticSearchPipeline
from tf.search.cosine_search import CosineSearcher
from tf.search.faiss_search import FaissSearcher
from tf.search.hnsw_search import HNSWSearcher

# Mapas de nome -> classe para os seletores da interface. A primeira entrada
# de cada mapa é o padrão e corresponde ao caminho já integrado ponta a ponta.
EMBEDDERS = {
    "TF-IDF": TfidfEmbedder,
    "Sentence Embeddings": SentenceEmbedder,
    "Word2Vec Average": Word2VecAverageEmbedder,
}
SEARCHERS = {
    "Similaridade de Cosseno": CosineSearcher,
    "FAISS": FaissSearcher,
    "HNSW": HNSWSearcher,
}


@st.cache_resource(show_spinner="Construindo pipeline (corpus + embeddings + LLM)...")
def build_pipeline(embedder_name: str, searcher_name: str) -> SemanticSearchPipeline:
    """Constrói (e cacheia) o pipeline para a combinação escolhida.

    O cache evita reconstruir embeddings/índices a cada interação.

    Args:
        embedder_name: Chave em :data:`EMBEDDERS`.
        searcher_name: Chave em :data:`SEARCHERS`.

    Returns:
        Pipeline pronto para responder consultas.
    """
    embedder = EMBEDDERS[embedder_name]()
    searcher = SEARCHERS[searcher_name]()
    llm = LocalLLM()
    pipeline = SemanticSearchPipeline(embedder=embedder, searcher=searcher, llm=llm)
    return pipeline.build()


def render_results(response) -> None:
    """Renderiza as sinopses recuperadas e a resposta do LLM.

    Args:
        response: Objeto :class:`tf.pipeline.PipelineResponse`.
    """
    if response.search_query:
        if response.search_query != response.question:
            st.info(f"Consulta reescrita pelo LLM: **{response.search_query}**")
        else:
            st.caption(
                f"Consulta de busca (LLM manteve a pergunta): _{response.search_query}_"
            )

    st.subheader("Resposta")
    st.write(response.answer or "_(sem resposta gerada)_")

    if response.elapsed:
        cols = st.columns(len(response.elapsed))
        for col, (etapa, segundos) in zip(cols, response.elapsed.items()):
            col.metric(etapa.capitalize(), f"{segundos:.2f} s")

    st.subheader(f"Sinopses recuperadas ({len(response.results)})")
    for rank, result in enumerate(response.results, start=1):
        payload = result.payload or {}
        title = payload.get("title") or "Título desconhecido"
        with st.expander(f"{rank}. {title}  ·  score {result.score:.4f}"):
            genres = payload.get("genres")
            if genres is not None and len(genres) > 0:
                st.caption("Gêneros: " + ", ".join(map(str, genres)))
            st.write(payload.get("plot", ""))


def render_benchmark_tab() -> None:
    """Renderiza a aba de benchmark comparando as técnicas de busca."""
    st.info(
        "Benchmark comparativo (tempo, complexidade e qualidade) ainda não "
        "implementado nesta integração mínima."
    )


def main() -> None:
    """Monta a interface Streamlit."""
    st.set_page_config(page_title="Busca Semântica de Filmes (CMU)", layout="wide")
    st.title("Busca Semântica de Filmes — CMU Movie Summary Corpus")
    st.caption(
        "Faça uma pergunta em linguagem natural e compare técnicas de busca semântica."
    )

    with st.sidebar:
        st.header("Configuração")
        embedder_name = st.selectbox("Técnica de embedding", list(EMBEDDERS))
        searcher_name = st.selectbox("Técnica de busca", list(SEARCHERS))
        top_k = st.slider("Top-K resultados", 1, 20, config.TOP_K)

    tab_search, tab_benchmark = st.tabs(["Busca", "Benchmark"])

    with tab_search:
        question = st.text_input(
            "Sua pergunta",
            placeholder="Ex.: um filme sobre viagem no tempo e paradoxos",
        )
        if st.button("Buscar", type="primary") and question.strip():
            pipeline = build_pipeline(embedder_name, searcher_name)
            response = pipeline.answer(question, top_k=top_k)
            render_results(response)

    with tab_benchmark:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
