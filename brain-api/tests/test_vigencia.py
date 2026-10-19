"""Vigência pelo frontmatter: parser (_vigencia), perfil (chaves trocadas), estado da nota, rebaixamento no RRF
(aplicar_vigencia), aviso do MCP e as colunas no Postgres (integração com ROLLBACK, só no container)."""
import os
from datetime import date
from pathlib import Path

import psycopg
import pytest

from src.config import perfil
from src.mcp_server import _aviso_vigencia
from src.parser import VaultParser
from src.perfil import VIGENCIA_PADRAO, carregar_perfil
from src.search import SQL_VIGENCIA, aplicar_vigencia, cap_per_note, estado_vigencia

HOJE = date(2026, 10, 18)


@pytest.fixture(autouse=True)
def chaves_padrao(monkeypatch):
    """O resultado não depende das chaves que o perfil ativo usar."""
    monkeypatch.setattr(perfil, "vigencia", {k: list(v) for k, v in VIGENCIA_PADRAO.items()})


def _nota(tmp_path: Path, frontmatter: str):
    nota = tmp_path / "nota.md"
    nota.write_text(f"---\n{frontmatter}\n---\n# Regra\nTexto da regra.\n", encoding="utf-8")
    return VaultParser.parse_file(nota, tmp_path)


def test_parser_le_datas_yaml_texto_e_link(tmp_path):
    v = _nota(tmp_path, 'vigente_desde: 2024-12-01\nrevogada_em: "2026-03-15"\n'
                        'substituida_por: "[[Portaria Nova|a nova]]"').vigencia
    assert v == {"vigente_desde": date(2024, 12, 1), "revogada_em": date(2026, 3, 15),
                 "substituida_por": "Portaria Nova"}


VAZIA = {"vigente_desde": None, "revogada_em": None, "substituida_por": None}


def test_parser_sem_campos_fica_none(tmp_path):
    assert _nota(tmp_path, "tags: [x]").vigencia == VAZIA


def test_data_invalida_anula_a_vigencia_para_nao_revogar_hoje(tmp_path, caplog):
    """Sem isso, data ilegível + substituida_por caía na regra "substituída sem data" e revogava a nota hoje."""
    assert _nota(tmp_path, 'revogada_em: "31/13/2026"\nsubstituida_por: "[[B]]"').vigencia == VAZIA
    assert "não é data" in caplog.text


def test_parser_aceita_data_br_hora_e_links_variados(tmp_path):
    v = _nota(tmp_path, 'revogada_em: "15/03/2026"\nvigente_desde: "2024-12-01T08:00:00-03:00"\n'
                        'substituida_por: [[Normas/Portaria 2#Art. 3|nova]]').vigencia  # sem aspas: lista de lista
    assert v == {"vigente_desde": date(2024, 12, 1), "revogada_em": date(2026, 3, 15),
                 "substituida_por": "Portaria 2"}
    assert _nota(tmp_path, 'superseded_by: ["[[B]]", "[[C]]"]').vigencia["substituida_por"] == "B"  # 1ª da lista


def test_parser_usa_as_chaves_do_perfil(tmp_path, monkeypatch, chaves_padrao):
    monkeypatch.setitem(perfil.vigencia, "revogada_em", ["revogado"])
    v = _nota(tmp_path, "revogado: 2025-01-02\nrevogada_em: 2020-01-01").vigencia
    assert v["revogada_em"] == date(2025, 1, 2)


def test_perfil_troca_chaves_e_recusa_campo_desconhecido(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "perfil.yaml").write_text("vigencia:\n  revogada_em: revogado\n", encoding="utf-8")
    p = carregar_perfil("x", str(tmp_path))
    assert p.vigencia["revogada_em"] == ["revogado"]
    assert p.vigencia["substituida_por"] == list(VIGENCIA_PADRAO["substituida_por"])
    (tmp_path / "x" / "perfil.yaml").write_text("vigencia:\n  expira: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expira"):
        carregar_perfil("x", str(tmp_path))


@pytest.mark.parametrize("ini, fim, nova, esperado", [
    (None, None, None, None),
    (None, date(2026, 10, 18), None, "revogada"),   # revogada hoje já não vale
    (None, date(2027, 1, 1), "B", None),             # revogação marcada para depois: ainda vale
    (None, None, "B", "revogada"),                   # substituída sem data
    (date(2027, 1, 1), None, None, "futura"),
    (date(2026, 1, 1), None, None, None),
])
def test_estado_vigencia(ini, fim, nova, esperado):
    assert estado_vigencia(ini, fim, nova, HOJE) == esperado


def _row(chunk, nota, rrf):
    return (chunk, nota, f"{nota}.md", str(nota), "", "", {}, None, None, rrf, None)


def test_revogada_cai_e_libera_a_vaga_antes_do_corte_por_nota():
    rows = [_row("a1", 1, 0.030), _row("a2", 1, 0.029), _row("b1", 2, 0.020), _row("c1", 3, 0.016)]
    vig = {1: (None, date(2025, 1, 1), "Nova"), 2: (None, None, None)}
    saida = cap_per_note(aplicar_vigencia(rows, vig, 0.5, HOJE), 3)
    assert [r[0] for r in saida] == ["b1", "c1", "a1"]       # 0,030 × 0,5 = 0,015 < 0,016
    assert saida[2][9] == pytest.approx(0.015)


def test_peso_1_e_sem_vigencia_nao_mexem_na_ordem():
    rows = [_row("a1", 1, 0.030), _row("b1", 2, 0.020)]
    vig = {1: (None, date(2025, 1, 1), None)}
    assert aplicar_vigencia(rows, vig, 1.0, HOJE) == rows
    assert aplicar_vigencia(rows, {}, 0.5, HOJE) == rows
    assert aplicar_vigencia(rows, {1: (None, None, None)}, 0.5, HOJE) == rows


def test_aviso_do_mcp():
    assert _aviso_vigencia(None, None, None, None) == ""
    rev = _aviso_vigencia("revogada", None, date(2025, 1, 2), "Nova")
    assert "REVOGADA em 2025-01-02" in rev and "[[Nova]]" in rev and rev.endswith("\n")
    assert "a partir de 2027-01-01" in _aviso_vigencia("futura", date(2027, 1, 1), None, None)


def test_hoje_usa_o_fuso_configurado(monkeypatch):
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    import src.search as busca
    monkeypatch.setattr(busca.settings, "fuso", "America/Sao_Paulo")
    assert busca.hoje() == datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    monkeypatch.setattr(busca.settings, "fuso", "")
    assert busca.hoje() == date.today()


def test_peso_fora_de_0_a_1_e_recusado():
    from pydantic import ValidationError
    from src.config import Settings
    with pytest.raises(ValidationError):
        Settings(revogada_peso=1.5)
    with pytest.raises(ValidationError):
        Settings(fuso="America/Sao Paulo")  # erro na subida, não em cada busca


def test_limiar_do_perfil_tambem_e_validado():
    from pydantic import ValidationError
    from src.config import Settings
    from src.perfil import Perfil, aplicar_limiares
    s = Settings()
    with pytest.raises(ValidationError):
        aplicar_limiares(Perfil(nome="x", limiares={"revogada_peso": 2.0}), s)
    aplicar_limiares(Perfil(nome="x", limiares={"revogada_peso": 0.25}), s)
    assert s.revogada_peso == 0.25


DB = os.environ.get("DATABASE_URL")


@pytest.mark.skipif(not DB, reason="DATABASE_URL ausente (rode no container)")
def test_colunas_de_vigencia_no_postgres():
    with psycopg.connect(DB) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO notes (file_path, title, file_hash, revogada_em, substituida_por) "
                            "VALUES ('__pytest__/velha.md', 'velha', %s, %s, 'nova') RETURNING id",
                            ("0" * 32, date(2025, 1, 2)))
                nid = cur.fetchone()[0]
                cur.execute(SQL_VIGENCIA, ([nid],))
                linha = cur.fetchone()
        finally:
            conn.rollback()
    assert linha == (nid, None, date(2025, 1, 2), "nova")
