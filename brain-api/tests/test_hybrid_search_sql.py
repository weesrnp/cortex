"""Integração: cortex_hybrid_search (RRF) contra o Postgres real, dentro de transação com ROLLBACK.
Roda no container: docker compose exec brain-api python -m pytest tests -q"""
import os

import psycopg
import pytest

from src.config import settings

DB = os.environ.get("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL ausente (rode no container: docker compose exec brain-api python -m pytest tests)")

PREFIX = "__pytest__/"


def vec(i: int) -> str:
    v = [0.0] * settings.embedding_dim
    v[i] = 1.0
    return str(v)


def test_rrf_prefers_chunk_matching_vector_and_text():
    with psycopg.connect(DB) as conn:
        try:
            with conn.cursor() as cur:
                rows = {
                    "ambos": (vec(0), "Protocolo de hipertensão arterial na gestante"),
                    "so_texto": (vec(5), "Hipertensão arterial: notas soltas"),
                    "so_vetor": (vec(0), "Calendário de férias da equipe"),
                }
                for name, (emb, content) in rows.items():
                    cur.execute(
                        "INSERT INTO notes (file_path, title, file_hash) VALUES (%s, %s, %s) RETURNING id",
                        (PREFIX + name + ".md", name, "0" * 32),
                    )
                    nid = cur.fetchone()[0]
                    cur.execute(
                        "INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, 0, %s, %s)",
                        (nid, content, emb),
                    )

                cur.execute(
                    "SELECT title, vector_rank, text_rank, rrf_score "
                    "FROM cortex_hybrid_search(%s, %s, 10, 60, %s)",
                    ("hipertensao gestante", vec(0), PREFIX),
                )
                res = cur.fetchall()
        finally:
            conn.rollback()

    titles = [r[0] for r in res]
    assert titles[0] == "ambos"                      # casa vetor E texto → maior RRF
    assert set(titles) == set(rows)                  # FULL OUTER JOIN mantém quem casa só um lado
    top = res[0]
    assert top[1] is not None and top[2] is not None
    assert top[3] == pytest.approx(1 / (60 + top[1]) + 1 / (60 + top[2]))


def test_filtros_include_exclude_de_pastas():
    with psycopg.connect(DB) as conn:
        try:
            with conn.cursor() as cur:
                for path in ("a/nota.md", "b/nota.md", "Excalidraw/desenho.md"):
                    cur.execute(
                        "INSERT INTO notes (file_path, title, file_hash) VALUES (%s, %s, %s) RETURNING id",
                        (PREFIX + path, path, "0" * 32),
                    )
                    cur.execute(
                        "INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, 0, %s, %s)",
                        (cur.fetchone()[0], "protocolo de hipertensão", vec(0)),
                    )

                def busca(include=None, exclude=None):
                    cur.execute(
                        "SELECT title FROM cortex_hybrid_search(%s, %s, 10, 60, %s, %s, %s)",
                        ("hipertensao", vec(0), PREFIX, include, exclude),
                    )
                    return {r[0] for r in cur.fetchall()}

                todos = busca()
                so_a = busca(include=[PREFIX + "a/"])
                sem_a = busca(exclude=[PREFIX + "a/"])
                a_e_b = busca(include=[PREFIX + "a/", PREFIX + "b/"], exclude=[PREFIX + "b/"])
        finally:
            conn.rollback()

    assert todos == {"a/nota.md", "b/nota.md", "Excalidraw/desenho.md"}
    assert so_a == {"a/nota.md"}
    assert sem_a == {"b/nota.md", "Excalidraw/desenho.md"}
    assert a_e_b == {"a/nota.md"}                     # exclude vence include


def test_lado_vetorial_e_exato():
    """Regressão (parser v5, 2026-09-24): o índice HNSW é aproximado e devolve no máximo hnsw.ef_search (40) candidatos,
    contando trechos já apagados até o VACUUM. Depois da reindexação ele deixou uma nota de referência fora do top-5
    (MRR 0,504) com a nota em 2º no cálculo exato (0,521). Aqui 60 trechos apagados mais próximos escondem o único vivo."""
    def perto(i: int) -> str:  # vetores distintos: o HNSW junta vetores idênticos numa entrada só
        v = [0.0] * settings.embedding_dim
        v[-1], v[i] = 1.0, 0.01
        return str(v)

    viva = [0.0] * settings.embedding_dim
    viva[-1], viva[-2] = 1.0, 0.1
    with psycopg.connect(DB) as conn:
        try:
            with conn.cursor() as cur:
                # Com 7 mil trechos o planejador lê o HNSW já em ordem; com meia dúzia, prefere ordenar. Força o plano grande.
                cur.execute("SET LOCAL enable_seqscan = off; SET LOCAL enable_sort = off")
                ids = {}
                for nome in ("velha", "viva"):
                    cur.execute(
                        "INSERT INTO notes (file_path, title, file_hash) VALUES (%s, %s, %s) RETURNING id",
                        (PREFIX + nome + ".md", nome, "0" * 32),
                    )
                    ids[nome] = cur.fetchone()[0]
                for i in range(60):  # versão velha, mais perto da pergunta que o trecho vivo
                    cur.execute(
                        "INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, %s, 'x', %s)",
                        (ids["velha"], i, perto(i)),
                    )
                cur.execute("DELETE FROM chunks WHERE note_id = %s", (ids["velha"],))
                cur.execute(
                    "INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, 0, 'y', %s)",
                    (ids["viva"], str(viva)),
                )
                cur.execute(
                    "SELECT title, vector_rank FROM cortex_hybrid_search(%s, %s, 10, 60, %s)",
                    ("zqxnadacasa", vec(settings.embedding_dim - 1), PREFIX),
                )
                res = cur.fetchall()
        finally:
            conn.rollback()

    assert res == [("viva", 1)]


def test_modos_do_fts_and_or_idf():
    """and exige todos os termos; or aceita qualquer um; idf põe o termo raro acima do comum."""
    textos = {"ambos": "zqxraro zqxcomum", "raro": "zqxraro"}
    textos.update({f"comum{i}": "zqxcomum" for i in range(5)})
    with psycopg.connect(DB) as conn:
        try:
            with conn.cursor() as cur:
                for nome, conteudo in textos.items():
                    cur.execute(
                        "INSERT INTO notes (file_path, title, file_hash) VALUES (%s, %s, %s) RETURNING id",
                        (PREFIX + nome + ".md", nome, "0" * 32),
                    )
                    cur.execute(
                        "INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, 0, %s, %s)",
                        (cur.fetchone()[0], conteudo, vec(1)),
                    )

                def fts(modo, pergunta="zqxraro zqxcomum"):
                    cur.execute(
                        "SELECT title, text_rank FROM cortex_hybrid_search(%s, %s, 20, 60, %s, NULL, NULL, %s)",
                        (pergunta, vec(0), PREFIX, modo),
                    )
                    return {t: r for t, r in cur.fetchall() if r is not None}

                por_modo = {m: fts(m) for m in ("and", "or", "idf")}
                vazias = [fts(m, "qual como quem") for m in ("or", "idf")]
                aspas = fts("idf", "d'água \\ it's \"x\"")
        finally:
            conn.rollback()

    assert por_modo["and"] == {"ambos": 1}
    assert set(por_modo["or"]) == set(textos) and por_modo["or"]["ambos"] == 1
    assert por_modo["idf"]["ambos"] == 1 and por_modo["idf"]["raro"] == 2
    assert vazias == [{}, {}] and aspas == {}  # só palavras vazias / aspas: sem erro e sem FTS


def test_via_de_fichas_traz_nota_fora_do_top25_do_vetor():
    """Migração 007: nota cujo trecho ficou fora dos 25 do vetor entra pela ficha, pelo trecho mais próximo, com
    note_weight/(k + posição da nota); note_weight = 0 desliga a via."""
    with psycopg.connect(DB) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO notes (file_path, title, file_hash) VALUES (%s, 'enche', %s) RETURNING id",
                            (PREFIX + "enche.md", "0" * 32))
                enche = cur.fetchone()[0]
                for i in range(26):  # 26 trechos colados na pergunta: ocupam os 25 do vetor
                    cur.execute("INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, %s, 'x', %s)",
                                (enche, i, vec(0)))
                cur.execute("INSERT INTO notes (file_path, title, file_hash, ficha_embedding) "
                            "VALUES (%s, 'alvo', %s, %s) RETURNING id", (PREFIX + "alvo.md", "0" * 32, vec(0)))
                alvo = cur.fetchone()[0]
                for i, eixo in enumerate((3, 4)):  # trecho do eixo 3 e do 4: os dois a distância 1
                    cur.execute("INSERT INTO chunks (note_id, chunk_index, content, embedding) VALUES (%s, %s, 'y', %s)",
                                (alvo, i, vec(eixo)))

                def buscar(peso):
                    cur.execute("SELECT title, vector_rank, note_rank, rrf_score "
                                "FROM cortex_hybrid_search(%s, %s, 50, 60, %s, NULL, NULL, 'and', %s)",
                                ("zqxnada", vec(0), PREFIX, peso))
                    return cur.fetchall()

                com, sem, dobro = buscar(1.0), buscar(0.0), buscar(2.0)
        finally:
            conn.rollback()

    alvos = [r for r in com if r[0] == "alvo"]
    assert len(alvos) == 1                                   # só o trecho mais próximo da nota entra
    assert alvos[0][1] is None and alvos[0][2] == 1          # fora do vetor, 1º na via de fichas
    assert alvos[0][3] == pytest.approx(1 / 61)
    assert all(r[2] is None for r in com if r[0] == "enche")  # nota sem ficha não ganha nada da via
    assert len([r for r in com if r[0] == "enche"]) == 25
    assert "alvo" not in [r[0] for r in sem]
    assert [r for r in dobro if r[0] == "alvo"][0][3] == pytest.approx(2 / 61)
