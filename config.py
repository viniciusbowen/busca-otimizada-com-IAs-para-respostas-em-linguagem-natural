"""Configurações centrais do projeto.

Reúne caminhos de arquivos, nomes de modelos e constantes usadas pelos
demais módulos. Manter tudo aqui facilita ajustar o comportamento do
sistema sem tocar na lógica.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# Diretório raiz do pacote / repositório.
BASE_DIR = Path(__file__).resolve().parent

# Raiz do repositório (mesma pasta do pacote ``busca``).
PROJECT_ROOT = BASE_DIR

# Diretório onde os dados brutos/processados ficam armazenados.
DATA_DIR = BASE_DIR / "data_files"

# Diretório onde os índices vetoriais (FAISS/HNSW) são persistidos.
INDEX_DIR = BASE_DIR / "indexes"

# Arquivo compactado do dataset (CMU Movie Summary Corpus).
DATASET_ZIP = PROJECT_ROOT / "cmu-movie-summary-corpus.zip"

# Diretório onde o dataset é extraído.
DATASET_DIR = DATA_DIR / "cmu-movie-summary-corpus"

# Arquivos do dataset após a extração.
METADATA_FILE = DATASET_DIR / "movie.metadata.tsv"
PLOT_SUMMARIES_FILE = DATASET_DIR / "plot_summaries.txt"

# Corpus unificado (metadados + sinopses) em formato Parquet, gerado pelo loader.
CORPUS_PARQUET = DATA_DIR / "corpus.parquet"

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
# Modelo de Sentence Embeddings (Sentence Transformers / Hugging Face).
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM local para geração da resposta final (Transformers).
# Alternativas simples: SmolLM, TinyLlama, Phi-3 Mini, Mistral.
LLM_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Parâmetros do Word2Vec (Gensim).
WORD2VEC_VECTOR_SIZE = 300
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 2

# Parâmetros do TF-IDF (scikit-learn / TfidfVectorizer).
# Limita o tamanho do vocabulário (dimensão dos vetores) mantendo os termos
# mais frequentes; ``None`` usa todo o vocabulário.
TFIDF_MAX_FEATURES = 50000
# Frequência mínima de documentos para um termo entrar no vocabulário.
TFIDF_MIN_DF = 2
# Se ``True``, remove stopwords do idioma configurado em ``LANGUAGE``.
TFIDF_USE_STOPWORDS = True

# ---------------------------------------------------------------------------
# Constantes de busca / geração
# ---------------------------------------------------------------------------
# Quantidade padrão de resultados retornados pela busca semântica.
TOP_K = 5

# Limite de documentos carregados no corpus para a integração mínima.
# O corpus completo tem ~42 mil sinopses; densificar a matriz TF-IDF sobre
# todas estouraria a memória. ``None`` usa o corpus inteiro (não recomendado
# com TF-IDF denso).
MAX_CORPUS_DOCS = 5000

# Parâmetros do índice HNSW (hnswlib).
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 50

# Métrica de distância padrão dos índices vetoriais ("cosine", "l2", "ip").
DISTANCE_METRIC = "cosine"

# Número máximo de tokens gerados pela resposta do LLM.
LLM_MAX_NEW_TOKENS = 256

# Idioma usado no pré-processamento (stopwords do NLTK).
LANGUAGE = "english"
