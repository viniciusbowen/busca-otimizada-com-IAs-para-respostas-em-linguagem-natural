"""Busca aproximada de vizinhos com HNSW (implementação própria).

HNSW (Hierarchical Navigable Small World) é um grafo de navegação em
múltiplas camadas que permite busca aproximada de vizinhos em tempo
tipicamente ~O(log N), com alta taxa de recall.

Estrutura do grafo:
    - Cada nó recebe uma camada máxima ``l`` sorteada de uma distribuição
      exponencial decrescente (poucos nós em camadas altas).
    - A camada 0 contém todos os nós; camadas superiores são cada vez mais
      esparsas e funcionam como "atalhos" de longa distância.
    - A busca desce das camadas altas (grosseiras) para a camada 0 (fina),
      refinando o conjunto de candidatos a cada nível.
"""

from __future__ import annotations

import heapq
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from tf import config
from tf.search.base import BaseSearcher, SearchResult


class HNSWSearcher(BaseSearcher):
    """Busca aproximada de vizinhos usando um HNSW"""

    name = "hnsw"

    _STATE_FILE = "hnsw_state.pkl"

    def __init__(
        self,
        metric: str = config.DISTANCE_METRIC,
        m: int = config.HNSW_M,
        ef_construction: int = config.HNSW_EF_CONSTRUCTION,
        ef_search: int = config.HNSW_EF_SEARCH,
        seed: int = 42,
    ) -> None:
        """Configura os hiperparâmetros do índice HNSW.

        Args:
            metric: Espaço de distância ("cosine", "l2", "ip").
            m: Número de conexões por nó (Mmax nas camadas > 0; Mmax0 = 2*m
                na camada 0). Também define ``mL = 1/ln(m)``.
            ef_construction: Tamanho da lista dinâmica de candidatos usada
                durante a inserção (recall/qualidade do grafo vs custo).
            ef_search: Tamanho da lista dinâmica na busca (recall vs
                velocidade da consulta).
            seed: Semente do gerador aleatório (sorteio das camadas).
        """
        if metric not in ("cosine", "l2", "ip"):
            raise ValueError(
                f"Métrica não suportada: {metric!r}. Use 'cosine', 'l2' ou 'ip'."
            )
        self.metric = metric
        self.m = m
        self.m_max = m  # máx. de conexões por nó nas camadas > 0
        self.m_max0 = 2 * m  # máx. de conexões na camada 0 (mais densa)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        # Fator de normalização do sorteio de camadas (nível ~ Exp(1/mL)).
        self._m_l = 1.0 / math.log(m) if m > 1 else 1.0
        self._rng = np.random.default_rng(seed)

        # Estado do grafo (preenchido em `build`).
        self._data: np.ndarray | None = None  # vetores (possivelmente normalizados)
        self._normalize = metric == "cosine"
        # _neighbors[node][layer] -> lista de ids de vizinhos naquela camada.
        self._neighbors: list[list[list[int]]] = []
        self._node_level: list[int] = []  # camada máxima de cada nó
        self._entry_point: int | None = None  # id do nó de entrada (topo)
        self._max_level: int = -1
        self._metadata: list[dict[str, Any]] | None = None

    @property
    def complexity(self) -> str:
        return "~O(log N) por consulta; O(N log N) para construir"

    # ------------------------------------------------------------------
    # Distâncias
    # ------------------------------------------------------------------
    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        """Converte para ``float32`` e normaliza L2 quando a métrica é cosseno."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors

    def _dist_many(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        """Distância entre a consulta ``q`` e um conjunto de nós (por id).

        Convenção: quanto MENOR a distância, mais próximo (mais similar).
        """
        X = self._data[ids]  # (k, d)
        if self.metric == "l2":
            diff = X - q
            return np.einsum("ij,ij->i", diff, diff)  # L2 ao quadrado
        # cosine (vetores normalizados) e ip.
        dots = X @ q
        if self.metric == "cosine":
            return 1.0 - dots
        return -dots  # ip: menor distância = maior produto interno

    def _dist_one(self, q: np.ndarray, node_id: int) -> float:
        return float(self._dist_many(q, [node_id])[0])

    def _dist_to_score(self, distance: float) -> float:
        """Converte distância em score (maior = mais similar)."""
        if self.metric == "cosine":
            return 1.0 - float(distance)
        return -float(distance)  # l2 e ip

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------
    def _random_level(self) -> int:
        """Sorteia a camada máxima de um nó (distribuição exponencial)."""
        u = self._rng.random()
        # Evita log(0); u está em [0, 1).
        return int(math.floor(-math.log(u + 1e-12) * self._m_l))

    def build(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "HNSWSearcher":
        """Constrói o grafo HNSW inserindo cada vetor do corpus.

        Complexidade: ~O(N log N) inserções, cada uma com buscas gulosas nas
        camadas superiores + refinamento (efConstruction) na camada 0.
        """
        vectors = self._prepare(vectors)
        if vectors.ndim != 2:
            raise ValueError(
                "`vectors` deve ser uma matriz 2-D (N x d); "
                f"recebido ndim={vectors.ndim}."
            )

        num_elements = vectors.shape[0]
        self._data = vectors
        self._metadata = metadata
        self._neighbors = [[] for _ in range(num_elements)]
        self._node_level = [0] * num_elements
        self._entry_point = None
        self._max_level = -1

        for node_id in range(num_elements):
            self._insert(node_id)

        return self

    def _insert(self, node_id: int) -> None:
        """Insere um nó no grafo"""
        q = self._data[node_id]
        level = self._random_level()
        self._node_level[node_id] = level
        # Uma lista de vizinhos por camada de 0 até `level`.
        self._neighbors[node_id] = [[] for _ in range(level + 1)]

        # Primeiro nó do grafo: vira o ponto de entrada.
        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = level
            return

        ep = self._entry_point
        top = self._max_level

        # Fase 1: desce das camadas altas até `level+1` com busca gulosa (ef=1),
        # apenas para posicionar o ponto de entrada mais próximo.
        for lc in range(top, level, -1):
            W = self._search_layer(q, [ep], ef=1, layer=lc)
            ep = self._nearest(W)

        # Fase 2: das camadas min(top, level) até 0, insere conexões.
        entry_points = [ep]
        for lc in range(min(top, level), -1, -1):
            W = self._search_layer(q, entry_points, ef=self.ef_construction, layer=lc)
            m_max = self.m_max0 if lc == 0 else self.m_max
            neighbors = self._select_neighbors(W, self.m)

            # Conexões bidirecionais entre `node_id` e os vizinhos escolhidos.
            self._neighbors[node_id][lc] = list(neighbors)
            for e in neighbors:
                self._neighbors[e][lc].append(node_id)
                # Poda as conexões de `e` se ultrapassar o limite da camada.
                if len(self._neighbors[e][lc]) > m_max:
                    e_vec = self._data[e]
                    self._neighbors[e][lc] = self._select_neighbors_by_ids(
                        e_vec, self._neighbors[e][lc], m_max
                    )

            # Candidatos desta camada viram pontos de entrada da próxima.
            entry_points = [n for (_neg_d, n) in W]

        # Se o novo nó alcança uma camada mais alta, ele passa a ser a entrada.
        if level > top:
            self._entry_point = node_id
            self._max_level = level

    def _search_layer(
        self, q: np.ndarray, entry_points: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """Busca gulosa dentro de uma única camada (SEARCH-LAYER).

        Retorna a lista dinâmica ``W`` como um max-heap de tuplas
        ``(-distância, id)`` — o topo (``W[0]``) é o candidato mais distante.
        """
        visited: set[int] = set(entry_points)
        candidates: list[tuple[float, int]] = []  # min-heap (dist, id)
        W: list[tuple[float, int]] = []  # max-heap (-dist, id)

        d0 = self._dist_many(q, entry_points)
        for ep, d in zip(entry_points, d0):
            d = float(d)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(W, (-d, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            furthest = -W[0][0]
            # Se o melhor candidato já é pior que o pior de W, encerra.
            if d_c > furthest:
                break

            # Vizinhos de `c` nesta camada (se a camada existir para o nó).
            c_neighbors = self._neighbors[c]
            if layer >= len(c_neighbors):
                continue
            unvisited = [e for e in c_neighbors[layer] if e not in visited]
            if not unvisited:
                continue

            dists = self._dist_many(q, unvisited)
            for e, d_e in zip(unvisited, dists):
                visited.add(e)
                d_e = float(d_e)
                furthest = -W[0][0]
                if d_e < furthest or len(W) < ef:
                    heapq.heappush(candidates, (d_e, e))
                    heapq.heappush(W, (-d_e, e))
                    if len(W) > ef:
                        heapq.heappop(W)  # remove o mais distante

        return W

    @staticmethod
    def _nearest(W: list[tuple[float, int]]) -> int:
        """Id do nó mais próximo em ``W`` (max de ``-dist`` = menor dist)."""
        return max(W, key=lambda item: item[0])[1]

    @staticmethod
    def _select_neighbors(W: list[tuple[float, int]], m: int) -> list[int]:
        """Seleção simples: os ``m`` candidatos mais próximos de ``W``."""
        # W guarda (-dist, id); ordena por dist crescente = -(-dist).
        ordered = sorted(W, key=lambda item: -item[0])
        return [node for (_neg_d, node) in ordered[:m]]

    def _select_neighbors_by_ids(
        self, base_vec: np.ndarray, ids: list[int], m: int
    ) -> list[int]:
        """Mantém os ``m`` ids mais próximos de ``base_vec`` (usado na poda)."""
        dists = self._dist_many(base_vec, ids)
        order = np.argsort(dists)[:m]
        return [ids[i] for i in order]

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Consulta o grafo pelos ``top_k`` vizinhos aproximados (K-NN-SEARCH).

        Complexidade: ~O(log N) por consulta.
        """
        if self._data is None or self._entry_point is None:
            raise RuntimeError("Índice não construído: chame `build` antes de `search`.")

        q = self._prepare(np.asarray(query_vector).reshape(1, -1))[0]

        k = min(top_k, self._data.shape[0])
        if k <= 0:
            return []

        # Desce das camadas altas até a camada 1 com busca gulosa (ef=1).
        ep = self._entry_point
        for lc in range(self._max_level, 0, -1):
            W = self._search_layer(q, [ep], ef=1, layer=lc)
            ep = self._nearest(W)

        # Camada 0: busca ampla com ef = max(ef_search, k).
        ef = max(self.ef_search, k)
        W = self._search_layer(q, [ep], ef=ef, layer=0)

        # Ordena por distância crescente e monta os top-k resultados.
        ordered = sorted(W, key=lambda item: -item[0])[:k]
        results: list[SearchResult] = []
        for neg_d, node in ordered:
            distance = -neg_d
            payload = self._metadata[node] if self._metadata is not None else {}
            results.append(
                SearchResult(
                    index=int(node),
                    score=self._dist_to_score(distance),
                    payload=payload,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    def save(self, path=config.INDEX_DIR) -> None:
        """Persiste o grafo e os metadados em disco."""
        if self._data is None:
            raise RuntimeError("Índice não construído: nada para salvar.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / self._STATE_FILE, "wb") as f:
            pickle.dump(
                {
                    "metric": self.metric,
                    "m": self.m,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "data": self._data,
                    "neighbors": self._neighbors,
                    "node_level": self._node_level,
                    "entry_point": self._entry_point,
                    "max_level": self._max_level,
                    "metadata": self._metadata,
                },
                f,
            )

    def load(self, path=config.INDEX_DIR) -> "HNSWSearcher":
        """Carrega o grafo e os metadados do disco."""
        path = Path(path)
        with open(path / self._STATE_FILE, "rb") as f:
            state = pickle.load(f)

        self.metric = state["metric"]
        self.m = state["m"]
        self.m_max = self.m
        self.m_max0 = 2 * self.m
        self.ef_construction = state["ef_construction"]
        self.ef_search = state["ef_search"]
        self._m_l = 1.0 / math.log(self.m) if self.m > 1 else 1.0
        self._normalize = self.metric == "cosine"
        self._data = state["data"]
        self._neighbors = state["neighbors"]
        self._node_level = state["node_level"]
        self._entry_point = state["entry_point"]
        self._max_level = state["max_level"]
        self._metadata = state["metadata"]
        return self
