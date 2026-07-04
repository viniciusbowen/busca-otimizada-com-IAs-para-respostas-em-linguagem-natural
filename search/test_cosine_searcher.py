"""Teste rápido / smoke test do CosineSearcher.

"""
from __future__ import annotations

import numpy as np

from cosine_search import CosineSearcher

def test_vetor_identico_tem_score_1():
    rng = np.random.default_rng(42)
    corpus = rng.normal(size=(50, 8))
    metadata = [{"id": i} for i in range(50)]

    searcher = CosineSearcher().build(corpus, metadata)
    consulta = corpus[10]  

    resultados = searcher.search(consulta, top_k=3)

    assert resultados[0].index == 10
    assert abs(resultados[0].score - 1.0) < 1e-5
    print("OK: vetor idêntico -> score ~1.0, índice correto")


def test_top_k_respeitado_e_ordenado():
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(200, 16))
    searcher = CosineSearcher().build(corpus)

    consulta = rng.normal(size=16)
    resultados = searcher.search(consulta, top_k=5)

    assert len(resultados) == 5
    scores = [r.score for r in resultados]
    assert scores == sorted(scores, reverse=True)
    print("OK: top_k=5 retornado em ordem decrescente de score")


def test_vetor_nulo_nao_quebra():
    corpus = np.random.default_rng(1).normal(size=(10, 4))
    searcher = CosineSearcher().build(corpus)

    resultados = searcher.search(np.zeros(4), top_k=3)

    assert all(r.score == 0.0 for r in resultados)
    print("OK: consulta com vetor nulo não quebra e retorna score 0.0")


def test_erro_sem_build():
    searcher = CosineSearcher()
    try:
        searcher.search(np.zeros(4), top_k=1)
    except RuntimeError:
        print("OK: search sem build levanta RuntimeError")
    else:
        raise AssertionError("Deveria ter levantado RuntimeError")


def test_contador_de_comparacoes():
    corpus = np.random.default_rng(2).normal(size=(30, 4))
    searcher = CosineSearcher().build(corpus)
    assert searcher.n_comparisons == 0

    searcher.search(np.random.default_rng(3).normal(size=4), top_k=2)
    searcher.search(np.random.default_rng(4).normal(size=4), top_k=2)

    assert searcher.n_comparisons == 60 
    print("OK: n_comparisons acumula N por consulta, como esperado em O(N*d)")


if __name__ == "__main__":
    test_vetor_identico_tem_score_1()
    test_top_k_respeitado_e_ordenado()
    test_vetor_nulo_nao_quebra()
    test_erro_sem_build()
    test_contador_de_comparacoes()
    print("\nTodos os testes passaram.")