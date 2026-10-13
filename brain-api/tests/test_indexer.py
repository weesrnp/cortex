"""Indexador: trecho com texto já indexado reaproveita o vetor; só o que mudou vai para o Ollama.
Postgres real dentro de transação com ROLLBACK (pula sem DATABASE_URL)."""
import asyncio
import os
from contextlib import asynccontextmanager

import psycopg
import pytest

from src.config import settings
from src.indexer import VaultIndexer
from src.parser import NoteChunk, ParsedNote, hash_ficha

DB = os.environ.get("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL ausente (rode no container: docker compose exec brain-api python -m pytest tests)")


class OllamaFalso:
    def __init__(self):
        self.pedidos = []

    async def get_document_embeddings(self, textos):
        self.pedidos.append(list(textos))
        return [[len(t) / 1000] + [0.0] * (settings.embedding_dim - 1) for t in textos]


class PoolDeUmaConexao:
    """index_single_note abre transação própria; aqui ela vira savepoint da transação do teste (desfeita no fim)."""
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self.conn


def nota(*textos) -> ParsedNote:
    return ParsedNote(file_path="__pytest__/reuso.md", title="reuso", file_hash="0" * 32, tags=[], links=[],
                      chunks=[NoteChunk(i, "reuso", t, len(t) // 4) for i, t in enumerate(textos)])


def test_so_trecho_novo_vai_para_o_ollama():
    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                ollama = OllamaFalso()
                indexer = VaultIndexer("/nao/usado", PoolDeUmaConexao(conn), ollama)
                await indexer.index_single_note(nota("[reuso]\nA", "[reuso]\nB", "[reuso]\nA"))
                await indexer.index_single_note(nota("[reuso]\nA", "[reuso]\nB editado", "[reuso]\nC"))
                cur = await conn.execute("""
                    SELECT c.content, c.embedding::text FROM chunks c JOIN notes n ON n.id = c.note_id
                    WHERE n.file_path = '__pytest__/reuso.md' ORDER BY c.chunk_index;
                """)
                return ollama.pedidos, await cur.fetchall()
            finally:
                await conn.rollback()

    pedidos, trechos = asyncio.run(rodar())
    ficha = nota("[reuso]\nA").ficha()                               # 1º trecho igual nas duas versões: mesma ficha
    assert pedidos == [["[reuso]\nA", "[reuso]\nB", ficha],        # "A" repetido na nota: um vetor só; ficha junto
                       ["[reuso]\nB editado", "[reuso]\nC"]]        # "A" e a ficha já estavam no banco: reaproveitados
    assert [t for t, _ in trechos] == ["[reuso]\nA", "[reuso]\nB editado", "[reuso]\nC"]
    assert trechos[0][1].startswith("[0.009")                       # vetor de "A" veio do banco, não do zero


def test_varredura_ignora_pasta_com_ponto_e_tira_nota_sem_arquivo(tmp_path):
    (tmp_path / "Modulo").mkdir()
    (tmp_path / "Modulo" / "nota.md").write_text("# Nota\n\nTexto da nota.", encoding="utf-8")
    for pasta in (".vault-operator/data/skills", ".smart-env", ".trash"):   # dados de plugin e lixeira do Obsidian
        (tmp_path / pasta).mkdir(parents=True)
        (tmp_path / pasta / "lixo.md").write_text("# Lixo\n\nArquivo de plugin.", encoding="utf-8")

    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                indexer = VaultIndexer(str(tmp_path), PoolDeUmaConexao(conn), OllamaFalso())
                await indexer.index_single_note(nota("[reuso]\nA"))    # no banco, sem arquivo no vault
                await indexer.scan_full_vault()
                cur = await conn.execute("SELECT file_path FROM notes;")
                return [r[0] for r in await cur.fetchall()]
            finally:
                await conn.rollback()

    # o índice real também sai (este vault de teste não tem as notas dele); o ROLLBACK devolve tudo. Os PDFs reais
    # ficam: vault sem nenhum PDF = montagem falhou, sincronizar_pdfs não apaga nada
    assert [c for c in asyncio.run(rodar()) if not c.lower().endswith(".pdf")] == ["Modulo/nota.md"]


def test_varredura_refaz_so_a_ficha_de_nota_inalterada(tmp_path):
    """Nota sem ficha (índice anterior às fichas) ganha a ficha sem reindexar os trechos."""
    (tmp_path / "Modulo").mkdir()
    (tmp_path / "Modulo" / "nota.md").write_text("# Nota\n\nResumo da nota.\n\n## Seção A\n\nTexto.", encoding="utf-8")

    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                ollama = OllamaFalso()
                indexer = VaultIndexer(str(tmp_path), PoolDeUmaConexao(conn), ollama)
                await indexer.scan_full_vault()                         # indexa trechos + ficha
                await conn.execute("UPDATE notes SET ficha_embedding = NULL, ficha_hash = NULL "
                                   "WHERE file_path = 'Modulo/nota.md';")
                ollama.pedidos.clear()
                await indexer.scan_full_vault()                         # trechos iguais: só a ficha volta
                cur = await conn.execute("SELECT ficha_hash, ficha_embedding IS NOT NULL FROM notes "
                                         "WHERE file_path = 'Modulo/nota.md';")
                return ollama.pedidos, await cur.fetchone()
            finally:
                await conn.rollback()

    pedidos, (ficha_hash, tem_vetor) = asyncio.run(rodar())
    assert len(pedidos) == 1 and len(pedidos[0]) == 1
    texto = pedidos[0][0]
    assert texto.startswith("[Nota]\nPasta: Modulo\nSeções: Nota; Seção A\n") and "Resumo da nota." in texto
    assert ficha_hash == hash_ficha(texto) and tem_vetor



def test_pdfs_copia_ligada_fica_e_nome_nominal_sai(tmp_path, monkeypatch):
    """Cópias com o mesmo texto: fica a que é alvo de [[link]]; nome com "paciente" fica fora (LGPD)."""
    from tests.test_parser import _pdf
    texto = ["Nota de referencia da equipe " + "conteudo " * 40]
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    (tmp_path / "A" / "Nota_C1.pdf").write_bytes(_pdf(texto))
    (tmp_path / "B" / "Nota Metodologica C1 - Mais acesso.pdf").write_bytes(_pdf(texto))  # mesma coisa, outro nome
    (tmp_path / "B" / "Lista_Pacientes_ACS.pdf").write_bytes(_pdf(["Fulano de Tal " * 30]))
    monkeypatch.setattr(settings, "pdf_exclude_pattern", r"(?i)paciente")

    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                await conn.execute("INSERT INTO notes (file_path, title, file_hash) VALUES "
                                   "('__pytest__/indice.md', 'indice', %s)", ("0" * 32,))
                await conn.execute("INSERT INTO note_links (source_note_id, target_title) SELECT id, "
                                   "'Nota Metodologica C1 - Mais acesso.pdf' FROM notes WHERE file_path = '__pytest__/indice.md'")
                indexer = VaultIndexer(str(tmp_path), PoolDeUmaConexao(conn), OllamaFalso())
                return [p.file_path for p in (await indexer.pdfs_para_indexar()).notas]
            finally:
                await conn.rollback()

    assert asyncio.run(rodar()) == ["B/Nota Metodologica C1 - Mais acesso.pdf"]


def test_pdf_ilegivel_continua_no_indice(tmp_path):
    """Erro de leitura passageiro (OneDrive baixando o arquivo) não tira o PDF do índice nem obriga a revetorizar."""
    from tests.test_parser import _pdf
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / "Bom.pdf").write_bytes(_pdf(["texto do documento bom " * 20]))
    (tmp_path / "Docs" / "Baixando.pdf").write_bytes(b"%PDF-1.4 corrompido")

    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                await conn.execute("INSERT INTO notes (file_path, title, file_hash) VALUES "
                                   "('Docs/Baixando.pdf', 'Baixando', %s), ('Docs/Sumiu.pdf', 'Sumiu', %s)",
                                   ("1" * 32, "2" * 32))
                indexer = VaultIndexer(str(tmp_path), PoolDeUmaConexao(conn), OllamaFalso())
                n = await indexer.sincronizar_pdfs()
                cur = await conn.execute("SELECT file_path FROM notes WHERE file_path LIKE 'Docs/%%' ORDER BY 1")
                return n, [r[0] for r in await cur.fetchall()]
            finally:
                await conn.rollback()

    n, caminhos = asyncio.run(rodar())
    assert n == 1
    assert caminhos == ["Docs/Baixando.pdf", "Docs/Bom.pdf"]   # ilegível fica; apagado do vault sai; bom entra


class Reinicio(BaseException):
    """`docker compose restart` no meio da passada: nada depois deste ponto roda."""


def test_pdf_movido_para_lixeira_sai_antes_de_reler_os_outros(tmp_path, monkeypatch):
    """2026-10-12: PDF movido para .trash ficou no índice. O evento chegou, mas a passada relê os ~60 PDFs (30-55 s)
    antes de apagar o que sumiu, e o restart 20 s depois a matou. O que sumiu do vault sai antes da releitura."""
    from src.parser import VaultParser
    from tests.test_parser import _pdf
    (tmp_path / "Rede").mkdir()
    (tmp_path / "Rede" / "Fica.pdf").write_bytes(_pdf(["manual do equipamento " * 20]))
    (tmp_path / ".trash").mkdir()
    (tmp_path / ".trash" / "Switch.pdf").write_bytes(_pdf(["guia de instalacao " * 20]))  # já movido

    def reler_e_morrer(*_):
        raise Reinicio()
    monkeypatch.setattr(VaultParser, "parse_pdf", staticmethod(reler_e_morrer))

    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                await conn.execute("INSERT INTO notes (file_path, title, file_hash) VALUES "
                                   "('Rede/Fica.pdf', 'Fica', %s), ('Rede/Switch.pdf', 'Switch', %s)",
                                   ("1" * 32, "2" * 32))
                indexer = VaultIndexer(str(tmp_path), PoolDeUmaConexao(conn), OllamaFalso())
                with pytest.raises(Reinicio):
                    await indexer.sincronizar_pdfs()
                cur = await conn.execute("SELECT file_path FROM notes WHERE file_path LIKE 'Rede/%%' ORDER BY 1")
                return [r[0] for r in await cur.fetchall()]
            finally:
                await conn.rollback()

    assert asyncio.run(rodar()) == ["Rede/Fica.pdf"]


def test_pdf_inalterado_nao_e_relido(tmp_path, monkeypatch):
    """Passada disparada por evento só relê o PDF que mudou (mtime ou tamanho): as outras vêm da leitura anterior."""
    from src.parser import VaultParser
    from tests.test_parser import _pdf
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / "A.pdf").write_bytes(_pdf(["documento A " * 20]))
    (tmp_path / "Docs" / "B.pdf").write_bytes(_pdf(["documento B " * 20]))
    lidos = []
    original = VaultParser.parse_pdf

    def contar(caminho, *args):
        lidos.append(caminho.name)
        return original(caminho, *args)
    monkeypatch.setattr(VaultParser, "parse_pdf", staticmethod(contar))

    async def rodar():
        async with await psycopg.AsyncConnection.connect(DB) as conn:
            try:
                indexer = VaultIndexer(str(tmp_path), PoolDeUmaConexao(conn), OllamaFalso())
                await indexer.sincronizar_pdfs()
                (tmp_path / "Docs" / "B.pdf").write_bytes(_pdf(["documento B editado " * 20]))
                await indexer.sincronizar_pdfs()
                cur = await conn.execute("SELECT file_path FROM notes WHERE file_path LIKE 'Docs/%%' ORDER BY 1")
                return [r[0] for r in await cur.fetchall()]
            finally:
                await conn.rollback()

    assert asyncio.run(rodar()) == ["Docs/A.pdf", "Docs/B.pdf"]
    assert lidos == ["A.pdf", "B.pdf", "B.pdf"]
