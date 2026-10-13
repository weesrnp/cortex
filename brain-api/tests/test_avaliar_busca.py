"""Contas do avaliar_busca.py (toda decisão da busca sai delas): posição com alvo único ou duas notas aceitas, MRR,
erro padrão e placar por pergunta."""
import pytest

from tests.avaliar_busca import PERGUNTAS, aceitas, erro_padrao, placar, posicao, rotulo, rr


def test_posicao_com_alvo_unico_e_com_duas_notas_aceitas():
    top5 = ["a", "irma", "b", "alvo", "c"]
    assert posicao(top5, "alvo") == 4
    assert posicao(top5, ("alvo", "irma")) == 2       # vale a primeira aceita que aparecer
    assert posicao(top5, "fora") is None
    assert rotulo(("alvo", "irma")) == "alvo (ou irma)" and rotulo("alvo") == "alvo"


def test_mrr_erro_padrao_e_placar():
    pos = [1, 2, None, 4]
    assert [rr(p) for p in pos] == [1.0, 0.5, 0.0, 0.25]
    assert erro_padrao([1, 1, 1]) == 0.0 and erro_padrao([1]) == 0.0
    assert erro_padrao([1, None]) == pytest.approx(0.5)  # desvio 0,707 / √2
    assert placar([1, None, 3], [2, 2, 3]) == "melhor em 1, pior em 1 de 3 perguntas"


def test_perguntas_sem_alvo_nem_texto_repetido():
    textos = [p for p, _ in PERGUNTAS]
    alvos = [a for _, alvo in PERGUNTAS for a in aceitas(alvo)]
    assert len(set(textos)) == len(textos) and len(set(alvos)) == len(alvos)
