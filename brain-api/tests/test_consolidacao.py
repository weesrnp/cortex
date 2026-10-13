"""Relatório de consolidação (src/consolidacao.py) e brain_feedback contra o Postgres real, dentro de transação com
ROLLBACK (pula sem DATABASE_URL). Notas de teste em __pytest__/ com vetores num eixo que o índice real não usa."""
import asyncio
import os
from contextlib import asynccontextmanager

import psycopg
import pytest

from src import consolidacao, mcp_server
from src.config import settings

DB = os.environ.get("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL ausente (rode no container: docker compose exec brain-api python -m pytest tests)")
PREFIX = "__pytest__/"


def vec(i: int) -> str:
    v = [0.0] * settings.embedding_dim
    v[i] = 1.0
    return str(v)


class PoolDeUmaConexao:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self.conn


async def _nota(cur, nome: str, ficha=None, conteudo: str = "x", links=()):
    await cur.execute("INSERT INTO notes (file_path, title, file_hash, ficha_embedding) VALUES (%s, %s, %s, %s) "
                      "RETURNING id", (PREFIX + nome + ".md", nome, "0" * 32, ficha))
    nid = (await cur.fetchone())[0]
    await cur.execute("INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, 0, %s, %s)",
                      (nid, f"[{nome}]\n{conteudo}", vec(3)))
    for alvo in links:
        await cur.execute("INSERT INTO note_links (source_note_id, target_title) VALUES (%s, %s)", (nid, alvo))
    return nid


def test_relatorio_acha_duplicata_orfa_link_quebrado_lacuna_e_feedback(monkeypatch):
    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                async with conn.cursor() as cur:
                    eixo = settings.embedding_dim - 7
                    await _nota(cur, "copia_a", vec(eixo), "texto repetido")
                    await _nota(cur, "copia_b", vec(eixo), "texto repetido")
                    await _nota(cur, "sozinha")
                    await _nota(cur, "com_link_ruim", links=("zqx_nota_inexistente",))
                    await cur.execute("INSERT INTO search_log (origem, query, max_similarity, baixa_confianca) "
                                      "VALUES ('mcp', 'zqx pergunta sem resposta cpf 12345678901', 0.4, true) "
                                      "RETURNING id")
                    busca = (await cur.fetchone())[0]

                    monkeypatch.setattr(mcp_server, "db", type("DB", (), {"pool": PoolDeUmaConexao(conn)}))
                    ok = await mcp_server.brain_feedback(busca, True, nota=PREFIX + "copia_a.md")
                    ruim = await mcp_server.brain_feedback(busca, False, comentario="faltou cpf 12345678901")
                    inexistente = await mcp_server.brain_feedback(-1, True)

                    rel = await consolidacao.gerar(cur, limite=50, dias=1)
                    return rel, ok, ruim, inexistente
            finally:
                await conn.rollback()

    rel, ok, ruim, inexistente = asyncio.run(rodar())
    par = [d for d in rel.duplicatas if {d[0], d[1]} == {PREFIX + "copia_a.md", PREFIX + "copia_b.md"}]
    assert par and par[0][2] == pytest.approx(1.0) and par[0][3] == 1      # ficha igual + 1 trecho idêntico
    assert par[0][6] > 0.5                                                  # texto inteiro parecido (trigramas)
    assert PREFIX + "sozinha.md" in rel.orfas
    assert (PREFIX + "com_link_ruim.md", 1, ["zqx_nota_inexistente"]) in [(c, t, list(a)) for c, t, a in rel.quebrados]
    assert any(q.startswith("zqx pergunta sem resposta") for q, *_ in rel.lacunas)
    assert "resolveu" in ok and "não serviu" in ruim and "não existe" in inexistente
    assert (PREFIX + "copia_a.md", 1) in rel.uteis
    nao = [n for n in rel.nao_serviu if n[0].startswith("zqx")]
    assert nao and nao[0][1] is None and "123.***.***-01" in nao[0][2]       # comentário gravado mascarado

    texto = consolidacao.formatar(rel, limite=50, dias=1)
    assert "## Quase duplicatas" in texto and "## Notas órfãs" in texto and "[[zqx_nota_inexistente]]" in texto
    assert "12345678901" not in texto                                         # LGPD na saída
    assert set(consolidacao.como_dict(rel)) == {"duplicatas", "orfas", "quebrados", "lacunas", "nao_serviu",
                                                 "inbox", "uteis", "sinalizadas"}


def test_nota_sinalizada_vale_ate_o_texto_mudar_ou_ser_resolvida(tmp_path, monkeypatch):
    """problema="desatualizada" entra em sinalizadas. Reindexar (updated_at novo) e mexer só no frontmatter não tiram o
    aviso; mudar o texto tira; "resolvida" encerra o aviso de nota que não se edita. "Não serviu" não repete o aviso."""
    pasta = tmp_path / "__pytest__"
    pasta.mkdir()
    (pasta / "portaria_velha.md").write_text("---\nstatus: ativo\n---\nvalor de 2024", encoding="utf-8")
    (pasta / "oficial.md").write_text("texto oficial", encoding="utf-8")
    monkeypatch.setattr(settings, "vault_path", str(tmp_path))

    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                async with conn.cursor() as cur:
                    await _nota(cur, "portaria_velha")
                    await _nota(cur, "oficial")
                    await cur.execute("INSERT INTO search_log (origem, query, baixa_confianca) "
                                      "VALUES ('mcp', 'zqx valor', false) RETURNING id")
                    busca = (await cur.fetchone())[0]
                    monkeypatch.setattr(mcp_server, "db", type("DB", (), {"pool": PoolDeUmaConexao(conn)}))
                    fb = mcp_server.brain_feedback
                    recusas = [await fb(busca, False, problema="errada"),
                               await fb(busca, False, nota="portaria_velha", problema="velha"),
                               await fb(busca, True, nota="portaria_velha", problema="errada")]
                    ok = await fb(busca, False, nota="portaria_velha", problema="Desatualizada",
                                  comentario="revogada; cpf 12345678901")
                    await fb(busca, False, nota="oficial", problema="errada")

                    async def ver():
                        rel = await consolidacao.gerar(cur, ["sinalizadas", "lacunas"], limite=50, dias=1,
                                                       vault_path=str(tmp_path))
                        return ({c for c, *_ in rel.sinalizadas if c.startswith(PREFIX)}, rel)
                    etapas = {}
                    etapas["antes"], rel = await ver()
                    await cur.execute("UPDATE notes SET updated_at = now() + interval '1 hour'")   # reindexação
                    (pasta / "portaria_velha.md").write_text("---\nstatus: revisar\n---\nvalor de 2024",
                                                             encoding="utf-8")                   # só frontmatter
                    etapas["reindexada"], _ = await ver()
                    resolvida = await fb(busca, False, nota="oficial", problema="resolvida")
                    etapas["resolvida"], _ = await ver()
                    (pasta / "portaria_velha.md").write_text("valor de 2026", encoding="utf-8")
                    etapas["corrigida"], _ = await ver()
                    return recusas, ok, resolvida, etapas, rel
            finally:
                await conn.rollback()

    recusas, ok, resolvida, etapas, rel = asyncio.run(rodar())
    assert "exige a nota" in recusas[0] and "deve ser um de" in recusas[1] and "util=False" in recusas[2]
    assert "desatualizada" in ok and "encerrados" in resolvida
    velha, oficial = PREFIX + "portaria_velha.md", PREFIX + "oficial.md"
    assert etapas["antes"] == {velha, oficial}
    assert etapas["reindexada"] == {velha, oficial}
    assert etapas["resolvida"] == {velha}
    assert etapas["corrigida"] == set()
    linha = [s for s in rel.sinalizadas if s[0] == velha]
    assert linha[0][1:3] == ("desatualizada", 1) and "123.***.***-01" in linha[0][4]
    assert not [n for n in rel.nao_serviu if n[1] in (velha, oficial)]        # não repete em "não serviu"
    texto = consolidacao.formatar(consolidacao.Relatorio(sinalizadas=linha), ["sinalizadas"])
    assert "## Sinalizadas" in texto and "portaria_velha" in texto and "12345678901" not in texto


def test_formatar_so_as_secoes_pedidas():
    rel = consolidacao.Relatorio(orfas=[f"n{i}.md" for i in range(12)])
    texto = consolidacao.formatar(rel, ["orfas"], limite=10)
    assert "## Notas órfãs (12" in texto and "… e mais 2" in texto
    assert "Quase duplicatas" not in texto and "Inbox" not in texto


def test_link_para_anexo_existente_nao_conta_como_quebrado(tmp_path):
    (tmp_path / "_anexos").mkdir()
    (tmp_path / "_anexos" / "Portaria 3493.pdf").write_bytes(b"%PDF")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "lixo.pdf").write_bytes(b"x")              # pasta com ponto: ignorada
    anexos = consolidacao.anexos_do_vault(str(tmp_path))
    assert anexos == {"Portaria 3493.pdf"}
    linhas = [("a.md", "Portaria 3493.pdf"), ("a.md", "Falta.pdf"), ("b.md", "Nota_Sumida"), ("b.md", "Outra")]
    assert consolidacao.agrupar_quebrados(linhas, anexos) == [("b.md", 2, ["Nota_Sumida", "Outra"]),
                                                             ("a.md", 1, ["Falta.pdf"])]
