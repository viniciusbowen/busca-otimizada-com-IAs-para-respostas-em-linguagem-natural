"""Localização e extração do dataset CMU Movie Summary Corpus.

O dataset já vem compactado no repositório (``cmu-movie-summary-corpus.zip``).
Este módulo garante que ele esteja extraído em ``config.DATASET_DIR`` antes do
carregamento. Também deixa espaço para um download remoto caso o arquivo local
não exista.
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from busca import config

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
    _extract_archive(zip_path, config.DATA_DIR)
    _normalize_dataset_dir()

    return config.DATASET_DIR


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extrai um arquivo compactado, detectando se é ``tar``/``tar.gz`` ou ``zip``.

    O dataset distribuído pela CMU vem como ``.tar.gz`` (mesmo quando o arquivo
    local tem extensão ``.zip``), então detectamos o formato pelo conteúdo em vez
    de confiar na extensão.

    Args:
        archive_path: Caminho do arquivo compactado.
        destination: Diretório de destino da extração.
    """
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(destination)
        return

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination)
        return

    raise ValueError(
        f"Formato de arquivo não reconhecido (nem tar nem zip): {archive_path}"
    )


def _normalize_dataset_dir() -> None:
    """Garante que os arquivos fiquem em ``config.DATASET_DIR``.

    O tarball da CMU extrai para uma pasta ``MovieSummaries``. Se o diretório
    esperado ainda não existir, movemos/renomeamos a pasta extraída para o nome
    configurado.
    """
    if config.METADATA_FILE.exists() and config.PLOT_SUMMARIES_FILE.exists():
        return

    extracted = config.DATA_DIR / "MovieSummaries"
    if extracted.exists() and extracted != config.DATASET_DIR:
        if config.DATASET_DIR.exists():
            shutil.rmtree(config.DATASET_DIR)
        shutil.move(str(extracted), str(config.DATASET_DIR))


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
