"""CPF e CNS com dígito verificador válido são mascarados antes de ir para o vault (brain_store e
brain_record_decision). Telefone, protocolo e código com o mesmo número de dígitos ficam como estão."""
import asyncio
from pathlib import Path

import pytest

from src import mcp_server
from src.search import _cns_valido, _cpf_valido, mascarar_gravacao

CPF = "12345678909"            # dígitos verificadores certos
CPF_FMT = "529.982.247-25"     # idem, formatado


def _cns(prefixo: str) -> str:
    """CNS válido a partir de 14 dígitos: o último fecha a soma ponderada em múltiplo de 11."""
    soma = sum(int(d) * p for d, p in zip(prefixo, range(15, 1, -1)))
    ultimo = (11 - soma % 11) % 11
    assert ultimo < 10, "troque o prefixo"
    return prefixo + str(ultimo)


CNS = _cns("70050012345671")


@pytest.fixture(autouse=True)
def sem_auto_linker(monkeypatch):
    async def identidade(title, content, rel_path):
        return content, 0
    monkeypatch.setattr(mcp_server, "_with_auto_links", identidade)


def test_validadores():
    assert _cpf_valido(CPF) and _cpf_valido("52998224725")
    assert not _cpf_valido("12345678900") and not _cpf_valido("11111111111")
    assert _cns_valido(CNS) and not _cns_valido(CNS[:-1] + str((int(CNS[-1]) + 1) % 10))
    assert not _cns_valido("3" + CNS[1:])                    # CNS começa com 1, 2, 7, 8 ou 9


def test_mascara_so_documento_valido():
    texto, n = mascarar_gravacao(f"CPF {CPF_FMT}, cpf={CPF}; CNS {CNS[:3]} {CNS[3:7]} {CNS[7:11]} {CNS[11:]}")
    assert n == 3
    assert texto == f"CPF 529.xxx.xxx-25, cpf=123.xxx.xxx-09; CNS {CNS[:3]} xxxx xxxx {CNS[11:]}"
    assert "*" not in texto  # no Obsidian "***" entre pontos vira itálico e some


@pytest.mark.parametrize("escrito", ["123.456.789/09", "123456789-09", "123 456 789 09", "123.456.789-09"])
def test_cpf_com_outros_separadores(escrito):
    assert mascarar_gravacao(f"cpf {escrito}.") == ("cpf 123.xxx.xxx-09.", 1)


def test_cns_com_ponto_ou_hifen():
    for sep in (".", "-", " "):
        texto = f"cns {CNS[:3]}{sep}{CNS[3:7]}{sep}{CNS[7:11]}{sep}{CNS[11:]}"
        assert mascarar_gravacao(texto) == (f"cns {CNS[:3]} xxxx xxxx {CNS[11:]}", 1)


def test_telefone_protocolo_e_codigos_ficam():
    texto = "Celular 46999887766, protocolo 700123456789012, INE 0000386340, CNES 5829992, CPF 123.456.789-00"
    assert mascarar_gravacao(texto) == (texto, 0)
    assert mascarar_gravacao("") == ("", 0) and mascarar_gravacao(None) == (None, 0)


def test_brain_store_grava_mascarado_e_avisa(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    msg = asyncio.run(mcp_server.brain_store("Caso C3", f"Paciente CNS {CNS}, CPF {CPF}, não contou no C3.",
                                             fonte=f"ficha do CPF {CPF_FMT}"))
    texto = (tmp_path / "00_Inbox" / "Caso C3.md").read_text(encoding="utf-8")
    assert CPF not in texto and CNS not in texto and CPF_FMT not in texto
    assert f"{CNS[:3]} xxxx xxxx {CNS[11:]}" in texto and "529.xxx.xxx-25" in texto
    assert "3 CPF/CNS mascarado(s)" in msg
    again = asyncio.run(mcp_server.brain_store("Caso C3", f"Paciente CNS {CNS}, CPF {CPF}, não contou no C3.",
                                               fonte=f"ficha do CPF {CPF_FMT}"))
    assert "nada mudou" in again and "mascarado" in again  # quem chamou fica sabendo da máscara


def test_titulo_com_documento_e_recusado(tmp_path: Path, monkeypatch):
    """O nome mascarado juntaria pacientes com os mesmos 3 primeiros e 2 últimos dígitos num arquivo só."""
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    assert asyncio.run(mcp_server.brain_store(f"Caso {CPF}", "texto da nota")).startswith("Erro")
    assert not list(tmp_path.rglob("*.md"))


def test_historico_guarda_a_versao_anterior_mascarada(tmp_path: Path, monkeypatch):
    """A versão anterior (escrita à mão com CPF) vai para .cortex_historico, que também fica no vault."""
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    (tmp_path / "00_Inbox").mkdir()
    (tmp_path / "00_Inbox" / "Nota.md").write_text(f"antiga com CPF {CPF}", encoding="utf-8")
    asyncio.run(mcp_server.brain_store("Nota", "texto novo da nota"))
    [versao] = [p for p in (tmp_path / ".cortex_historico").rglob("*.md") if p.is_file()]  # pasta "Nota.md/"
    assert versao.read_text(encoding="utf-8") == "antiga com CPF 123.xxx.xxx-09"


def test_brain_record_decision_tambem_mascara(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    asyncio.run(mcp_server.brain_record_decision("Caso", f"Corrigir o CNS {CNS}", "nada", "erro de digitação"))
    [adr] = (tmp_path / "00_Agent_Decisions").glob("*.md")
    assert CNS not in adr.read_text(encoding="utf-8")


def test_sem_documento_nao_muda_nada(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    msg = asyncio.run(mcp_server.brain_store("Nota", "Texto da regra sem dado pessoal."))
    assert "mascarado" not in msg
    assert (tmp_path / "00_Inbox" / "Nota.md").read_text(encoding="utf-8") == "Texto da regra sem dado pessoal."
