import asyncio
from pathlib import Path

import pytest

from src import mcp_server

_REAL_AUTO_LINKS = mcp_server._with_auto_links  # antes do fixture autouse trocar o atributo


@pytest.fixture(autouse=True)
def sem_auto_linker(monkeypatch):
    """Testes de gravação não dependem de Ollama/banco; o auto-linker tem teste próprio abaixo."""
    async def identidade(title, content, rel_path):
        return content, 0
    monkeypatch.setattr(mcp_server, "_with_auto_links", identidade)


def test_brain_store_writes_inside_vault(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    msg = asyncio.run(mcp_server.brain_store("Nota/Teste", "conteúdo", folder="00_Inbox"))
    assert (tmp_path / "00_Inbox" / "Nota-Teste.md").read_text(encoding="utf-8") == "conteúdo"
    assert "sucesso" in msg


def test_brain_store_blocks_path_traversal(tmp_path: Path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(vault))
    for bad in ("../fora", "../../etc", str(tmp_path)):
        msg = asyncio.run(mcp_server.brain_store("x", "y", folder=bad))
        assert msg.startswith("Erro"), bad
    assert not list(tmp_path.glob("**/x.md"))


def test_brain_record_decision_grava_adr(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    msg = asyncio.run(mcp_server.brain_record_decision(
        topic="Embeddings: qwen3 vs bge-m3?",
        decision="qwen3-embedding:4b cortado em 1024d",
        rejected="bge-m3: abaixo no MTEB multilíngue",
        rationale="nomic fraco em português",
    ))
    assert "sucesso" in msg
    [adr] = (tmp_path / "00_Agent_Decisions").glob("*.md")
    assert ":" not in adr.name and "?" not in adr.name        # nome válido no Windows/OneDrive
    texto = adr.read_text(encoding="utf-8")
    assert texto.startswith("---\ntipo: adr-agente")
    for secao in ("## Contexto e Problema", "## Decisão Adotada", "## Abordagens Descartadas"):
        assert secao in texto
    assert "bge-m3: abaixo" in texto


def test_brain_store_recusa_titulo_vazio(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    assert asyncio.run(mcp_server.brain_store("  ..  ", "x")).startswith("Erro")


def test_brain_store_recusa_vazio_e_lixo_repetido(tmp_path: Path, monkeypatch):
    """Porta de qualidade (A4 do ai-memory), calibrada nas notas do vault: só barra lixo, nada é gravado."""
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    for lixo in ("", "  \n\t ", "---\nfonte: x\n---\n", "=" * 80, "erro erro erro " * 40, "ok " * 25):
        msg = asyncio.run(mcp_server.brain_store("Lixo", lixo))
        assert msg.startswith("Erro: nada gravado"), (lixo[:20], msg)
    assert not (tmp_path / "00_Inbox").exists()
    for bom in ("C2: 4 doses", "linha velha", "| a | b |\n|---|---|\n| sim | não |",
                "A equipe comercial (código 1234560) atende o interior; o contrato 0000123456 cobre 4 regiões.",
                # notas só de dados (revisão de 2026-10-07): tabela 0/1, calendário, matriz Sim/Não, CPF mascarado
                "| a | b | c |\n|---|---|---|\n" + "| 0 | 1 | 0 |\n" * 40,
                "| mês | dose |\n|---|---|\n" + "| jan | ✔ |\n| fev | — |\n" * 30,
                "| Loja | Sim | Não |\n|---|---|---|\n" + "| Central | Sim | Não |\n" * 7,
                "```sql\nSELECT 1;\n```\n" + "```\n" + "a" * 500 + "\n```",
                "\n".join(f"- 0{i}6.***.***-7{i}" for i in range(10))):
        assert mcp_server.texto_ruim(bom) is None, bom


class _FakeConn:
    def __init__(self, rows, calls):
        self.rows, self.calls = rows, calls
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def execute(self, sql, params):
        self.calls.append(params)
        rows = self.rows
        class Cur:
            async def fetchall(self_inner):
                return rows
        return Cur()


def test_auto_linker_filtra_limiar_links_existentes_e_refaz_secao(monkeypatch):
    real = _REAL_AUTO_LINKS
    calls = []
    rows = [("Nota_Ja_Linkada", 0.95), ("Nota_A", 0.91), ("Nota_B", 0.80), ("Nota_C", 0.74), ("Nota_Fraca", 0.69)]

    class FakeDB:
        class pool:
            @staticmethod
            def connection():
                return _FakeConn(rows, calls)

    class FakeEmbed:
        async def get_document_embeddings(self, texts):
            assert texts[0].startswith("Titulo\n")
            return [[0.0] * 1024]

    monkeypatch.setattr(mcp_server, "db", FakeDB)
    monkeypatch.setattr(mcp_server, "embed_client", FakeEmbed())
    monkeypatch.setattr(mcp_server.settings, "auto_link_threshold", 0.70)
    monkeypatch.setattr(mcp_server.settings, "auto_link_max", 3)

    antigo = ("Texto com [[pasta/Nota_Ja_Linkada#Seção|apelido]].\n\n---\n"
              "## Conexões Sugeridas (Auto-Linker)\n- [[Velha]] (Similaridade: 99%)\n")
    novo, n = asyncio.run(real("Titulo", antigo, "00_Inbox/Titulo.md"))

    assert n == 3
    assert "[[Velha]]" not in novo                                   # seção antiga refeita, não duplicada
    assert novo.count("## Conexões Sugeridas (Auto-Linker)") == 1
    assert "- [[Nota_A]] (Similaridade: 91%)" in novo
    assert "[[Nota_C]]" in novo and "Nota_Fraca" not in novo         # limiar 0,70
    assert novo.count("Nota_Ja_Linkada") == 1                        # não repete link que o texto já tem
    assert calls[0][1] == "00_Inbox/Titulo.md"                       # exclui a própria nota


def test_auto_linker_falhando_nao_impede_gravacao(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    async def quebra(*a):
        raise RuntimeError("ollama fora do ar")
    monkeypatch.setattr(mcp_server, "_with_auto_links", quebra)
    msg = asyncio.run(mcp_server.brain_store("Nota", "conteúdo"))
    assert "sucesso" in msg
    assert (tmp_path / "00_Inbox" / "Nota.md").read_text(encoding="utf-8") == "conteúdo"


def test_sobrescrever_guarda_versao_anterior_e_historico_mostra_o_diff(tmp_path: Path, monkeypatch):
    """Ideia do AgentMemoryRepo/OpenMemory: memória com histórico. A versão velha vai para .cortex_historico/ (pasta
    com ponto: o indexador ignora); gravar o mesmo conteúdo não cria versão."""
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    asyncio.run(mcp_server.brain_store("Nota", "linha velha\n", folder="05_Pasta"))
    assert "nada mudou" in asyncio.run(mcp_server.brain_store("Nota", "linha velha\n", folder="05_Pasta"))
    msg = asyncio.run(mcp_server.brain_store("Nota", "linha nova\n", folder="05_Pasta"))
    [versao] = (tmp_path / ".cortex_historico" / "05_Pasta" / "Nota.md").glob("*.md")
    assert versao.read_text(encoding="utf-8") == "linha velha\n"
    assert "Versão anterior guardada" in msg and ".cortex_historico/05_Pasta/Nota.md/" in msg

    monkeypatch.setattr(mcp_server, "db", _DBSoCursor)
    monkeypatch.setattr(mcp_server, "_find_notes", lambda cur, key: _lista([(1, "05_Pasta/Nota.md", "Nota")]))
    lista = asyncio.run(mcp_server.brain_historico("Nota"))
    assert versao.stem in lista and "1 versões anteriores" in lista
    diff = asyncio.run(mcp_server.brain_historico("Nota", versao=versao.stem))
    assert "-linha velha" in diff and "+linha nova" in diff


async def _lista(rows):
    return rows


class _DBSoCursor:
    """db.pool.connection() → conn.cursor(): só para chegar ao _find_notes (trocado no teste)."""
    class pool:
        @staticmethod
        def connection():
            class Conn:
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *a):
                    return False
                def cursor(self):
                    return self
            return Conn()


def test_fonte_vai_para_o_frontmatter_sem_repetir(tmp_path: Path, monkeypatch):
    import frontmatter
    monkeypatch.setattr(mcp_server.settings, "vault_path", str(tmp_path))
    fonte = 'Portaria "3.493/2024" em C:\\Users\\x\\portaria.pdf'               # aspas e barra invertida
    asyncio.run(mcp_server.brain_store("A", "texto", fonte=fonte))
    nota = frontmatter.loads((tmp_path / "00_Inbox" / "A.md").read_text(encoding="utf-8"))
    assert nota.metadata["fonte"] == fonte and nota.content == "texto"     # YAML válido, valor intacto
    com_fm = mcp_server._com_procedencia("---\ntags: [x]\nfonte: \"velha\"\n---\ncorpo", "nova")
    meta = frontmatter.loads(com_fm)
    assert meta.metadata["tags"] == ["x"] and meta.metadata["fonte"] == "nova" and meta.content == "corpo"
    assert com_fm.count("fonte:") == 1
    for fm in ("---\n---\ncorpo", "---\r\ntags: [y]\r\n---\r\ncorpo"):      # vazio e CRLF: um bloco só
        saida = mcp_server._com_procedencia(fm, "z")
        assert saida.count("---") == 2 and frontmatter.loads(saida).metadata["fonte"] == "z"
