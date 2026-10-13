"""Expansão de siglas na pergunta (src/siglas.py): nos dois sentidos, sem falso positivo em palavra comum.
Usa um glossário próprio do teste, para valer com qualquer perfil ativo."""
from src.siglas import SIGLAS, expandir_siglas

GLOSSARIO = {
    "LGPD": "Lei Geral de Proteção de Dados",
    "SLA": "Acordo de Nível de Serviço",
    "OKR": "Objetivos e Resultados-Chave",
    "SIM": "Sistema de Informações sobre Mortalidade",
}


def test_sigla_ganha_extenso():
    assert expandir_siglas("pedido de exclusão pela LGPD", GLOSSARIO) == \
        "pedido de exclusão pela LGPD (LGPD: Lei Geral de Proteção de Dados)"


def test_extenso_ganha_sigla_sem_acento_nem_caixa():
    out = expandir_siglas("prazo do acordo de nivel de servico", GLOSSARIO)
    assert out.endswith("(SLA: Acordo de Nível de Serviço)")


def test_varias_siglas_em_ordem_e_sem_repetir():
    out = expandir_siglas("SLA e OKR do trimestre, acordo de nível de serviço", GLOSSARIO)
    assert out.count("SLA: ") == 1 and "OKR: " in out


def test_sem_falso_positivo():
    for pergunta in ("sim, o prazo vale", "o SIMPLES nacional", "okr em minúsculas", "SLAvic", "ticket WEB-12"):
        assert expandir_siglas(pergunta, GLOSSARIO) == pergunta, pergunta


def test_glossario_vazio_nao_mexe():
    assert expandir_siglas("qualquer LGPD", {}) == "qualquer LGPD"


def test_siglas_do_perfil_validas():
    assert all(isinstance(s, str) and s.strip() and str(e).strip() for s, e in SIGLAS.items())
