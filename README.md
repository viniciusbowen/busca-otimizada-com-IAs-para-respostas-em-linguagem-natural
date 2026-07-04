# Busca Semântica de Filmes — CMU Movie Summary Corpus

Projeto da disciplina **Projeto e Análise de Algoritmos (PAA) — 1/2026**.

Sistema em que o usuário faz uma pergunta em linguagem natural sobre filmes e
o sistema responde de forma formatada. A resposta é construída a partir de uma
**busca semântica** sobre as sinopses do *CMU Movie Summary Corpus*, seguida da
geração de texto por um **LLM local**.

> Este repositório contém apenas o **esqueleto** do projeto. Os algoritmos ainda
> não estão implementados: cada função levanta `NotImplementedError` e está
> pronta para ser preenchida.

## Integrantes

- Vinícius Bowen — 180079239
- Miguel Carvalho de Medeiros — 211068341
- Luca Valderramos Cirino — 211066140
- Luisa Ribeiro de Oliveira — 241024197
- Gustavo de Paula Ávila - 212006871

## Fluxo do sistema

1. O usuário faz uma pergunta em linguagem natural.
2. A pergunta é convertida em vetor (embedding).
3. Busca semântica nas sinopses usando diferentes técnicas:
   - Word2Vec Average
   - Sentence Embeddings
   - Similaridade de Cosseno (exaustiva)
   - HNSW Search (e FAISS)
4. Comparação das técnicas quanto a complexidade, tempo de execução e qualidade.
5. As sinopses recuperadas alimentam um LLM local que gera a resposta.
6. A resposta formatada é exibida na interface web (Streamlit).

## Estrutura

```
tf/
├── app.py                # Interface Streamlit (entrypoint)
├── config.py             # Caminhos, modelos e constantes
├── pipeline.py           # Orquestra pergunta -> embedding -> busca -> LLM
├── data/
│   ├── download.py       # Localiza/extrai o dataset
│   └── loader.py         # Une metadados + sinopses no corpus
├── preprocessing/
│   └── text_cleaner.py   # Normalização, tokenização, stopwords (NLTK)
├── embeddings/
│   ├── base.py           # Interface BaseEmbedder
│   ├── word2vec_avg.py   # Word2Vec Average (Gensim)
│   └── sentence_embed.py # Sentence Transformers
├── search/
│   ├── base.py           # Interface BaseSearcher + SearchResult
│   ├── cosine_search.py  # Busca exaustiva por cosseno
│   ├── faiss_search.py   # Índice FAISS
│   └── hnsw_search.py    # Índice HNSW (hnswlib)
├── llm/
│   └── generator.py      # LLM local (Transformers) — RAG
└── benchmark/
    └── evaluator.py      # Tempo, complexidade e qualidade (P@K, R@K, MRR)
```

## Dataset

*CMU Movie Summary Corpus*, composto por:

- `movie.metadata.tsv` — metadados dos filmes (id, título, gêneros).
- `plot_summaries.txt` — sinopses indexadas pelo Wikipedia movie ID.

O arquivo `cmu-movie-summary-corpus.zip` deve estar na raiz do repositório. A
função `tf.data.download.ensure_dataset()` cuida da extração para
`tf/data_files/`.

## Instalação

Requer **Python 3.12**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r tf/requirements.txt
```

## Execução

A partir da raiz do repositório:

```bash
streamlit run tf/app.py
```

## Tecnologias e licenças

| Categoria | Biblioteca | Licença |
|---|---|---|
| Linguagem | Python 3.12 | PSF |
| Dados | NumPy, Pandas | BSD |
| IA/NLP | scikit-learn | BSD |
| IA/NLP | NLTK | Apache 2.0 |
| IA/NLP | Gensim | LGPL |
| IA/NLP | Sentence Transformers | Apache 2.0 |
| IA/NLP | PyTorch | BSD-style |
| IA/NLP | Hugging Face Transformers | Apache 2.0 |
| Busca | FAISS | MIT |
| Busca | hnswlib | Apache 2.0 |
| Interface | Streamlit | Apache 2.0 |
