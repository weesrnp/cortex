"""Aviso de baixa confiança (maior cosseno abaixo do limite) e log das buscas em search_log.
Partes com Postgres real rodam dentro de transação com ROLLBACK (pulam sem DATABASE_URL)."""
import asyncio
import os
from contextlib import asynccontextmanager

import psycopg
import pytest

from src import mcp_server
from src.config import settings
from src.search import HybridSearchEngine, SearchResult, is_low_confidence, log_search, max_similarity

DB = os.environ.get("DATABASE_URL")
precisa_db = pytest.mark.skipif(not DB, reason="DATABASE_URL ausente (rode no container: docker compose exec brain-api python -m pytest tests)")
LIMITE = settings.low_confidence_threshold


def res(sim, path="a/Nota.md"):
    return SearchResult(chunk_id=1, note_id=1, file_path=path, title="Nota", heading_hierarchy="Nota",
                        content="texto", metadata={}, vector_rank=1, text_rank=None, rrf_score=1 / 61,
                        outlinks=[], backlinks=[], similarity=sim)


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


def test_baixa_confianca_usa_o_maior_cosseno():
    assert is_low_confidence([])                                  # nada devolvido = nada no vault
    assert is_low_confidence([res(None)])                         # sem cosseno não dá para confiar
    assert is_low_confidence([res(LIMITE - 0.01), res(0.1)])
    assert not is_low_confidence([res(0.1), res(LIMITE)])         # basta um trecho no limite
    assert max_similarity([res(0.3), res(None), res(0.5)]) == 0.5


def test_brain_search_avisa_so_quando_baixa(monkeypatch):
    async def nada(*a, **k):
        return None
    monkeypatch.setattr(mcp_server, "log_search", nada)

    for sim, avisa in ((LIMITE - 0.05, True), (LIMITE + 0.05, False)):
        async def busca(**k):
            return [res(sim)]
        monkeypatch.setattr(mcp_server.search_engine, "search", busca)
        txt = asyncio.run(mcp_server.brain_search("pergunta"))
        assert ("BAIXA CONFIANÇA" in txt) is avisa
        assert f"sim {sim:.2f}" in txt


def test_falha_no_log_nao_derruba_a_busca():
    class PoolQuebrado:
        def connection(self):
            raise RuntimeError("banco fora")
    asyncio.run(log_search(PoolQuebrado(), "mcp", "q", {}, [res(0.9)], 5))   # não levanta


@precisa_db
def test_log_grava_pergunta_mascarada_e_resultados():
    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                await log_search(PoolDeUmaConexao(conn), "mcp", "paciente cpf 12345678901 hipertenso",
                                 {"include": ["01_"], "exclude": None}, [res(0.81, "x/A.md"), res(0.4, "y/B.md")], 42)
                cur = await conn.execute(
                    "SELECT origem, query, filtros, resultados, max_similarity, baixa_confianca, duracao_ms "
                    "FROM search_log ORDER BY id DESC LIMIT 1")
                return await cur.fetchone()
            finally:
                await conn.rollback()

    origem, query, filtros, resultados, top, baixa, ms = asyncio.run(rodar())
    assert origem == "mcp" and ms == 42
    assert "12345678901" not in query and "123.***.***-01" in query      # LGPD: nunca grava CPF inteiro
    assert filtros == {"include": ["01_"]}                                # filtro vazio não ocupa espaço
    assert [r["file_path"] for r in resultados] == ["x/A.md", "y/B.md"]
    assert top == pytest.approx(0.81) and baixa is (0.81 < LIMITE)


@precisa_db
def test_search_devolve_cosseno_de_cada_trecho():
    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                for nome, emb in (("igual", vec(0)), ("ortogonal", vec(7))):
                    cur = await conn.execute(
                        "INSERT INTO notes (file_path, title, file_hash) VALUES (%s, %s, %s) RETURNING id",
                        (f"__pytest__/{nome}.md", nome, "0" * 32))
                    nid = (await cur.fetchone())[0]
                    await conn.execute(
                        "INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, 0, %s, %s)",
                        (nid, f"trecho {nome}", emb))
                engine = HybridSearchEngine(PoolDeUmaConexao(conn), embed_client=None)
                return await engine.search("zzz", limit=5, include_folders=["__pytest__/"], expand_graph=False,
                                           query_vector=[float(x) for x in vec(0)[1:-1].split(",")])
            finally:
                await conn.rollback()

    sims = {r.title: r.similarity for r in asyncio.run(rodar())}
    assert sims["igual"] == pytest.approx(1.0)
    assert sims["ortogonal"] == pytest.approx(0.0)
