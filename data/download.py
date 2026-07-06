"""Localização e extração do dataset CMU Movie Summary Corpus.

O dataset já vem compactado no repositório (``cmu-movie-summary-corpus.zip``).
Este módulo garante que ele esteja extraído em ``config.DATASET_DIR`` antes do
carregamento. Também deixa espaço para um download remoto caso o arquivo local
não exista.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from tf import config

# URL de fallback caso o zip não esteja presente localmente. O CMU Movie
# Summary Corpus é distribuído pela Carnegie Mellon University.
DATASET_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"


def ensure_dataset(force: bool = False) -> Path:
    """Garante que o dataset esteja disponível e extraído localmente.

    Passos:
    1. Se ``config.DATASET_DIR`` já existir com os arquivos e ``force`` for
       ``False``, retorná-lo sem trabalho adicional.
    2. Caso o zip não exista localmente, baixá-lo (download remoto).
    3. Extrair o zip em ``config.DATA_DIR``.

    Args:
        force: Se ``True``, força a re-extração mesmo que os arquivos existam.

    Returns:
        Caminho do diretório com os arquivos do dataset.

    Complexidade: O(T) no tamanho total dos arquivos extraídos.
    """
    already_extracted = (
        config.METADATA_FILE.exists() and config.PLOT_SUMMARIES_FILE.exists()
    )
    if already_extracted and not force:
        return config.DATASET_DIR

    zip_path = config.DATASET_ZIP
    if not zip_path.exists():
        download_dataset(DATASET_URL, zip_path)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.DATA_DIR)

    return config.DATASET_DIR


def download_dataset(url: str, destination: Path) -> Path:
    """Baixa o zip do dataset de uma URL remota (fallback).

    Args:
        url: URL de origem do arquivo compactado.
        destination: Caminho local onde salvar o arquivo.

    Returns:
        Caminho do arquivo baixado.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination
