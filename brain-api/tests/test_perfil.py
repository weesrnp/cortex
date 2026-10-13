"""Perfil da área (src/perfil.py): o núcleo roda neutro sem perfil e o perfil de exemplo carrega o que a área define."""
from pathlib import Path

import pytest

from src.config import Settings
from src.mcp_server import montar_instrucoes
from src.perfil import Perfil, aplicar_limiares, carregar_perfil

PERFIS = Path(__file__).resolve().parents[2] / "perfis"
if not PERFIS.is_dir():  # no container, perfis/ fica montado em /app/perfis
    PERFIS = Path("/app/perfis")


def test_sem_perfil_fica_neutro():
    p = carregar_perfil("", "/nao/existe")
    assert p.siglas == {} and p.metadados == {} and p.perguntas("principais") == [] and p.lista("fora") == []
    texto = montar_instrucoes(p)
    assert "PASTAS" not in texto and "COMO USAR" in texto and "REGRAS" in texto


def test_perfil_inexistente_para():
    with pytest.raises(FileNotFoundError):
        carregar_perfil("nao_existe", str(PERFIS))


def test_siglas_com_chave_booleana_param(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "siglas.yaml").write_text("NO: Nota Oficial\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aspas"):
        carregar_perfil("x", str(tmp_path))


def test_perfil_exemplo_carregado():
    p = carregar_perfil("exemplo", str(PERFIS))
    assert p.nome_mcp == "cortex-exemplo" and p.siglas["SLA"] == "Acordo de Nível de Serviço"
    assert set(p.metadados) == {"tickets", "datas"}
    assert p.metadados["tickets"].findall("ver WEB-12 e INF-1") == ["WEB-12", "INF-1"]
    assert p.perguntas("principais")
    assert any(isinstance(a, tuple) for _, a in p.perguntas("conferencia")), "alvo com mais de uma nota vira tupla"
    assert p.lista("fora"), "perguntas que o vault não responde, para calibrar_confianca.py"
    texto = montar_instrucoes(p)
    assert "PASTAS" in texto and "Decisões em 00_Agent_Decisions" in texto and p.dominio in texto


def test_limiar_do_perfil_nao_vence_variavel_de_ambiente(monkeypatch):
    p = Perfil(nome="t", limiares={"low_confidence_threshold": 0.5, "auto_link_threshold": 0.9})
    monkeypatch.setenv("AUTO_LINK_THRESHOLD", "0.8")
    s = Settings()
    aplicar_limiares(p, s)
    assert s.low_confidence_threshold == 0.5 and s.auto_link_threshold == 0.8


def test_limiar_desconhecido_para():
    with pytest.raises(ValueError):
        aplicar_limiares(Perfil(nome="t", limiares={"limiar_inventado": 1.0}), Settings())
