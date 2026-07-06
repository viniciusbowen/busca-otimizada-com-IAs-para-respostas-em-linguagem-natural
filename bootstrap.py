"""Registra o pacote ``busca`` para importação estável.

O nome da pasta do repositório contém hífens
(``busca-otimizada-com-IAs-para-respostas-em-linguagem-natural``), inválidos
em identificadores Python. Este módulo expõe o mesmo código sob o alias
``busca`` (abreviação de *busca otimizada*).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_NAME = "busca"
ROOT = Path(__file__).resolve().parent


def ensure_package() -> None:
    """Garante que ``busca`` aponta para a raiz deste repositório."""
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if PACKAGE_NAME in sys.modules:
        return

    init_path = ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        init_path,
        submodule_search_locations=[root_str],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar o pacote {PACKAGE_NAME!r}")

    module = importlib.util.module_from_spec(spec)
    module.__path__ = [root_str]  # type: ignore[attr-defined]
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)


ensure_package()
