"""Carregamento e unificação do CMU Movie Summary Corpus.

O corpus é composto por dois arquivos:
- ``movie.metadata.tsv``: metadados dos filmes (id, título, gêneros, etc.).
- ``plot_summaries.txt``: sinopses indexadas pelo id do filme (Wikipedia ID).

Este módulo une ambos em um único :class:`pandas.DataFrame` pronto para o
pré-processamento e a indexação vetorial.
"""

from __future__ import annotations

import json

import pandas as pd

from busca import config
from busca.data.download import ensure_dataset

# Colunas esperadas no DataFrame unificado gerado por ``build_corpus``.
CORPUS_COLUMNS = ["wiki_id", "title", "genres", "plot"]

# Ordem das colunas do arquivo movie.metadata.tsv (sem cabeçalho).
_METADATA_COLUMNS = [
    "wiki_id",
    "freebase_id",
    "title",
    "release_date",
    "revenue",
    "runtime",
    "languages",
    "countries",
    "genres",
]


def _parse_freebase_map(raw: str) -> list[str]:
    """Extrai os nomes de um mapa Freebase (``{"/m/...": "Nome"}``).

    Args:
        raw: String JSON serializada como aparece no dataset.

    Returns:
        Lista com os nomes (valores) do mapa; vazia se o parsing falhar.
    """
    if not raw:
        return []
    try:
        return list(json.loads(raw).values())
    except (json.JSONDecodeError, TypeError):
        return []


def load_metadata(path=config.METADATA_FILE) -> pd.DataFrame:
    """Carrega os metadados dos filmes (``movie.metadata.tsv``).

    O arquivo é um TSV sem cabeçalho; as colunas relevantes são o Wikipedia
    movie ID, o título e o mapa de gêneros (JSON serializado).

    Args:
        path: Caminho do arquivo de metadados.

    Returns:
        DataFrame com, ao menos, ``wiki_id``, ``title`` e ``genres``.

    Complexidade: O(N) no número de filmes.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=_METADATA_COLUMNS,
        dtype={"wiki_id": "int64"},
    )
    df["genres"] = df["genres"].apply(_parse_freebase_map)
    return df[["wiki_id", "title", "genres"]]


def load_plot_summaries(path=config.PLOT_SUMMARIES_FILE) -> pd.DataFrame:
    """Carrega as sinopses (``plot_summaries.txt``).

    Cada linha contém ``<wiki_id>\\t<sinopse>``.

    Args:
        path: Caminho do arquivo de sinopses.

    Returns:
        DataFrame com ``wiki_id`` e ``plot``.

    Complexidade: O(N) no número de sinopses.
    """
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["wiki_id", "plot"],
        dtype={"wiki_id": "int64", "plot": "string"},
        quoting=3,  # csv.QUOTE_NONE: sinopses contêm aspas livremente.
    )


def build_corpus(save: bool = True) -> pd.DataFrame:
    """Une metadados e sinopses em um único corpus.

    Passos:
    1. Garantir o dataset via :func:`busca.data.download.ensure_dataset`.
    2. Carregar metadados e sinopses.
    3. Fazer o *join* por ``wiki_id``.
    4. Opcionalmente persistir em ``config.CORPUS_PARQUET``.

    Args:
        save: Se ``True``, salva o corpus em Parquet para reuso.

    Returns:
        DataFrame com as colunas em :data:`CORPUS_COLUMNS`.

    Complexidade: O(N) para o join por chave (hash join).
    """
    ensure_dataset()

    metadata = load_metadata()
    plots = load_plot_summaries()

    # Join interno: só interessam filmes que possuem sinopse.
    corpus = plots.merge(metadata, on="wiki_id", how="inner")

    corpus["title"] = corpus["title"].fillna("").astype("string")
    corpus["genres"] = corpus["genres"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    corpus = corpus[CORPUS_COLUMNS].reset_index(drop=True)

    if save:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        corpus.to_parquet(config.CORPUS_PARQUET, index=False)

    return corpus


def load_corpus(path=config.CORPUS_PARQUET) -> pd.DataFrame:
    """Carrega o corpus já unificado a partir do Parquet, se existir.

    Se o arquivo ainda não foi gerado, delega a :func:`build_corpus`.

    Args:
        path: Caminho do arquivo Parquet do corpus.

    Returns:
        DataFrame do corpus.
    """
    if path.exists():
        return pd.read_parquet(path)
    return build_corpus(save=True)
