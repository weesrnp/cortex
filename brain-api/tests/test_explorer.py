"""/explorer: justificativa do RRF, fragmentos HTML (escape + LGPD), grafo e página sem CDN.
Os testes de SQL rodam no Postgres real dentro de transação com ROLLBACK (pulam sem DATABASE_URL)."""
import asyncio
import os
import re

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import explorer
from src.config import settings
from src.search import SearchResult

DB = os.environ.get("DATABASE_URL")
com_banco = pytest.mark.skipif(not DB, reason="DATABASE_URL ausente (rode no container: docker compose exec brain-api python -m pytest tests)")
PREFIX = "__pytest__/"


def resultado(**campos) -> SearchResult:
    base = dict(chunk_id=1, note_id=10, file_path="01_A/Nota.md", title="Nota", heading_hierarchy="Seção",
                content="texto", metadata={}, vector_rank=1, text_rank=1, rrf_score=2 / 61, outlinks=[], backlinks=[])
    return SearchResult(**{**base, **campos})


def test_justificativa_explica_cada_lado():
    assert "1/(60+2) + 1/(60+1) = 0.0325" in explorer.justificativa(2, 1, True, True)
    so_vetor = explorer.justificativa(3, None, False, True, "and")
    assert so_vetor.startswith("Só pelo vetor: 1/(60+3) = 0.0159") and "não contém todos os termos" in so_vetor
    assert "nenhum dos termos" in explorer.justificativa(3, None, False, True, "idf")
    assert "fora dos 25 melhores" in explorer.justificativa(3, None, True, True)
    assert "palavras vazias" in explorer.justificativa(3, None, False, False)
    assert explorer.justificativa(None, 4, True, True).startswith("Só pelo FTS: 1/(60+4)")


def test_resultados_escapam_html_e_mascaram_cpf():
    r = resultado(title="<script>alert(1)</script> CPF 12345678901", content="<img src=x onerror=alert(1)>",
                  metadata={"tickets": ["WEB-12"], "datas": []}, text_rank=None, rrf_score=1 / 61)
    html = explorer.render_resultados("pergunta", [r, resultado(chunk_id=2, note_id=11)],
                                      {1: (0.81, None), 2: (0.7, 0.4)}, [("<b>pergunt", 3, 1.2)])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<img" not in html and "<b>pergunt" not in html
    assert "12345678901" not in html and "123.***.***-01" in html
    assert "cos 0.810" in html and "não casou" in html and "ts_rank_cd 0.400" in html
    idf = explorer.render_resultados("pergunta", [resultado()], {1: (0.9, 5.25)}, [("pergunt", 0, 3.0)], "idf")
    assert "idf 5.250" in idf and "FTS idf:" in idf and 'class="chip zero"' in idf and "pergunt · 0 · 3.0" in idf
    assert 'data-notas="10,11"' in html
    assert '<span class="chip" title="tickets">WEB-12</span>' in html and 'title="datas"' not in html


def test_resultados_vazio_e_ajuda_sem_pergunta():
    assert "Nada encontrado" in explorer.render_resultados("<b>x</b>", [], {}, [])
    assert "&lt;b&gt;" in explorer.render_resultados("<b>x</b>", [], {}, [])
    cliente = TestClient(_app())
    assert "300 ms" in cliente.get("/explorer/buscar", params={"q": "  "}).text


def test_grafo_marca_links_para_nota_inexistente_e_ignora_auto_link():
    g = explorer.build_graph(
        centros=[(1, "Centro", "a/Centro.md")],
        saidas=[(1, "Vizinha", 2, "Vizinha", "b/Vizinha.md"), (1, "Sumida", None, None, None),
                (1, "Centro", 1, "Centro", "a/Centro.md")],
        entradas=[(3, "Quem cita", "c/Quem cita.md", 1), (2, "Vizinha", "b/Vizinha.md", 1)],
    )
    nos = {n["id"]: n for n in g["nodes"]}
    assert nos["n1"]["group"] == "centro" and nos["n2"]["group"] == "nota"
    assert nos["x:Sumida"]["group"] == "ausente" and nos["x:Sumida"]["nota"] is None
    pares = {(e["from"], e["to"]) for e in g["edges"]}
    assert pares == {("n1", "n2"), ("n1", "x:Sumida"), ("n3", "n1"), ("n2", "n1")}  # sem laço n1→n1


def test_pagina_tem_csp_e_so_carrega_arquivos_locais():
    cliente = TestClient(_app())
    for url in ("/explorer", "/explorer/"):
        resp = cliente.get(url)
        assert resp.status_code == 200
        assert "script-src 'self'" in resp.headers["content-security-policy"]
    srcs = re.findall(r'(?:src|href)="([^"]+)"', resp.text)
    assert srcs and all(s.startswith("/explorer/static/") for s in srcs), srcs
    for s in srcs:
        assert (explorer.STATIC_DIR / s.rsplit("/", 1)[-1]).is_file(), s


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(explorer.router)
    return app


def vec(i: int) -> str:
    v = [0.0] * settings.embedding_dim
    v[i] = 1.0
    return str(v)


async def _com_notas(notas, consulta):
    """notas: [(nome, conteúdo, [links])]. Insere, roda consulta(cur, ids) e desfaz tudo."""
    async with await psycopg.AsyncConnection.connect(DB) as conn:
        try:
            async with conn.cursor() as cur:
                ids = {}
                for nome, conteudo, links in notas:
                    await cur.execute("INSERT INTO notes (file_path, title, file_hash) VALUES (%s, %s, %s) RETURNING id",
                                      (f"{PREFIX}{nome}.md", nome, "0" * 32))
                    ids[nome] = (await cur.fetchone())[0]
                    await cur.execute("INSERT INTO chunks (note_id, chunk_index, content, embedding) "
                                      "VALUES (%s, 0, %s, %s) RETURNING id", (ids[nome], conteudo, vec(0)))
                    ids[nome + "#0"] = (await cur.fetchone())[0]
                    for alvo in links:
                        await cur.execute("INSERT INTO note_links (source_note_id, target_title) VALUES (%s, %s)",
                                          (ids[nome], alvo))
                return ids, await consulta(cur, ids)
        finally:
            await conn.rollback()


@com_banco
def test_grafo_no_postgres_resolve_links_pelo_nome_do_arquivo():
    notas = [("pytest_A", "a", ["pytest_B", "pasta/pytest_B#Seção", "pytest_sumida"]),
             ("pytest_B", "b", []), ("pytest_C", "c", ["pytest_A"])]
    ids, g = asyncio.run(_com_notas(notas, lambda cur, ids: explorer.fetch_graph(cur, [ids["pytest_A"]])))
    a, b, c = (f"n{ids[n]}" for n in ("pytest_A", "pytest_B", "pytest_C"))
    pares = {(e["from"], e["to"]) for e in g["edges"]}
    assert pares == {(a, b), (a, "x:pytest_sumida"), (c, a)}  # #Seção e pasta/ caem no mesmo stem


@com_banco
def test_scores_e_nota_inteira_no_postgres():
    notas = [("pytest_H", "Protocolo de hipertensão arterial na gestante", ["pytest_G", "pytest_nada"]),
             ("pytest_G", "outra", ["pytest_H"])]

    async def consulta(cur, ids):
        vetor = [1.0] + [0.0] * (settings.embedding_dim - 1)
        trechos = [ids["pytest_H#0"], ids["pytest_G#0"]]
        return ({m: await explorer.fetch_scores(cur, "hipertensão gestante zqxinexistente", vetor, trechos, m)
                 for m in ("and", "idf")}, await explorer.fetch_note(cur, ids["pytest_H"]))

    ids, (por_modo, dados) = asyncio.run(_com_notas(notas, consulta))
    termos, scores = por_modo["idf"]
    assert termos[0][0].startswith("zqxinexist") and termos[0][1] == 0  # df 0 vem primeiro
    assert any(re.fullmatch(r"hipertens\w*", t[0]) for t in termos)  # radical exato depende do Snowball
    cos, score = scores[ids["pytest_H#0"]]
    assert cos == pytest.approx(1.0) and score > 0
    assert scores[ids["pytest_G#0"]][1] is None                        # FTS não aceitou o trecho
    assert por_modo["and"][1][ids["pytest_H#0"]][1] is None             # and: falta o termo inexistente
    assert dados["links"] == [("pytest_G", ids["pytest_G"]), ("pytest_nada", None)]
    assert dados["backlinks"] == [(ids["pytest_G"], "pytest_G")]
    html = explorer.render_nota(dados, trecho_destacado=ids["pytest_H#0"])
    assert "chunk destaque" in html and 'class="ausente"' in html and "Backlinks (1)" in html


def test_justificativa_com_a_ficha_da_nota():
    """Migração 007: a ficha soma peso/(k + posição da nota); trecho só da ficha tem frase própria."""
    assert "1/(60+3) + 1/(60+2) = 0.0320" in explorer.justificativa(3, None, False, True, "and", 2, 1.0)
    assert "ficha da nota (#2" in explorer.justificativa(3, None, False, True, "and", 2, 1.0)
    so_ficha = explorer.justificativa(None, None, False, True, "and", 1, 1.0)
    assert so_ficha.startswith("Só pela ficha da nota: 1/(60+1) = 0.0164")
    assert "ficha" not in explorer.justificativa(3, None, False, True, "and", 2, 0.0)  # via desligada
