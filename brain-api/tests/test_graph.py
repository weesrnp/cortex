"""Testes unitários e de integração do motor de grafo (GraphEngine) e brain_graph MCP."""
import asyncio
import pytest

from src import mcp_server
from src.graph import GraphEngine, GraphNode


class _CursorGrafoFalso:
    """Simula o cursor do PostgreSQL com respostas parametrizadas para testes isolados."""
    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.sqls = []

    async def execute(self, sql, params=None):
        self.sqls.append((sql, params))
        if self.respostas:
            self._atual = self.respostas.pop(0)
        else:
            self._atual = []

    async def fetchone(self):
        return self._atual if isinstance(self._atual, (tuple, list)) and self._atual and not isinstance(self._atual[0], tuple) else (self._atual[0] if self._atual else None)

    async def fetchall(self):
        return self._atual if isinstance(self._atual, list) else [self._atual]


class _ConnGrafoFalso:
    def __init__(self, respostas):
        self.cursor_obj = _CursorGrafoFalso(respostas)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def cursor(self):
        return self._FakeContext(self.cursor_obj)

    class _FakeContext:
        def __init__(self, c):
            self.c = c
        async def __aenter__(self):
            return self.c
        async def __aexit__(self, *args):
            return False


class _PoolGrafoFalso:
    def __init__(self, respostas):
        self.respostas = respostas

    def connection(self):
        return _ConnGrafoFalso(self.respostas)


def test_mcp_tool_brain_graph_registrada_com_anotacoes():
    """A nova ferramenta brain_graph deve estar registrada como somente leitura."""
    tools = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert "brain_graph" in tools
    bg = tools["brain_graph"]
    assert bg.annotations.readOnlyHint is True
    assert bg.annotations.openWorldHint is False


def test_graph_engine_resolve_note_por_titulo():
    pool = _PoolGrafoFalso([
        (10, "Mapa de Projetos", "01_Projetos/00_MAPA.md", "00_MAPA")
    ])
    engine = GraphEngine(pool)

    async def run():
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                return await engine.resolve_note(cur, "Mapa de Projetos")

    node = asyncio.run(run())
    assert node is not None
    assert node.id == 10
    assert node.title == "Mapa de Projetos"
    assert node.folder == "01_Projetos"
    assert node.stem == "00_MAPA"


def test_graph_engine_traverse_hop1():
    # 1. resolve_note
    # 2. outlinks
    # 3. backlinks
    respostas = [
        (1, "Origem", "01_Pasta/Origem.md", "Origem"),
        [
            (2, "Alvo 1", "02_Pasta/Alvo1.md", "Alvo1", "Alvo 1"),
            (None, None, None, "Nota_Quebrada", "Nota Quebrada"),
        ],
        [
            (3, "Referenciador", "03_Pasta/Ref.md", "Ref"),
        ]
    ]
    pool = _PoolGrafoFalso(respostas)
    engine = GraphEngine(pool)

    res = asyncio.run(engine.traverse("Origem", hops=1))
    assert res is not None
    assert res.root.title == "Origem"
    assert len(res.hop1_outlinks) == 1
    assert res.hop1_outlinks[0].title == "Alvo 1"
    assert len(res.hop1_backlinks) == 1
    assert res.hop1_backlinks[0].title == "Referenciador"
    assert "Nota Quebrada" in res.unresolved_links or "Nota_Quebrada" in res.unresolved_links
    assert res.total_nodes_visited == 3


def test_mcp_brain_graph_retorna_caminho_formatado(monkeypatch):
    """Testa a formatação da tool brain_graph quando target_title é informado."""
    async def fake_shortest_path(source_key, target_key, max_depth, direction):
        u = GraphNode(1, "Plano Anual", "01_Projetos/Plano.md", "01_Projetos", "Plano")
        p = GraphNode(2, "Rotina_de_Backup", "02_Referencias/Rotina_de_Backup.md", "02_Referencias", "Rotina_de_Backup")
        v = GraphNode(3, "Equipe Comercial", "03_Equipes/Comercial.md", "03_Equipes", "Comercial")
        from src.graph import PathResult
        return PathResult(
            source=u,
            target=v,
            length=2,
            steps=[(u, "->", p), (p, "<-", v)],
            path_nodes=[u, p, v],
        )

    monkeypatch.setattr(mcp_server.graph_engine, "find_shortest_path", fake_shortest_path)

    out = asyncio.run(mcp_server.brain_graph("Plano Anual", target_title="Equipe Comercial"))
    assert "Rota no Grafo de Conhecimento (2 saltos)" in out
    assert "Plano Anual" in out
    assert "Rotina_de_Backup" in out
    assert "Equipe Comercial" in out


def test_mcp_brain_graph_traversal_formatado(monkeypatch):
    """Testa a formatação da tool brain_graph com vizinhança e pontes."""
    async def fake_traverse(root_key, hops, direction, limit_per_hop):
        root = GraphNode(1, "Central", "01_Pasta/Central.md", "01_Pasta", "Central")
        h1_out = [GraphNode(2, "Vizinho 1", "02_Pasta/V1.md", "02_Pasta", "V1")]
        h1_in = [GraphNode(3, "Vizinho 2", "03_Pasta/V2.md", "03_Pasta", "V2")]
        bridge = GraphNode(4, "Ponte Importante", "04_Pasta/Ponte.md", "04_Pasta", "Ponte")
        from src.graph import TraversalResult
        return TraversalResult(
            root=root,
            hops=2,
            direction="both",
            hop1_outlinks=h1_out,
            hop1_backlinks=h1_in,
            hop2_nodes=[bridge],
            bridges=[(bridge, ["V1", "V2"])],
            unresolved_links=["Futura_Nota"],
            total_nodes_visited=4,
        )

    monkeypatch.setattr(mcp_server.graph_engine, "traverse", fake_traverse)

    out = asyncio.run(mcp_server.brain_graph("Central", hops=2))
    assert "Grafo de Conhecimento: Central" in out
    assert "Hop 1 — Conexões Diretas" in out
    assert "Vizinho 1" in out
    assert "Pontes Conceituais (Hop 2)" in out
    assert "Ponte Importante" in out
    assert "Conectado via: `[[V1]]`, `[[V2]]`" in out
    assert "Futura_Nota" in out
