"""Instruções do servidor MCP e localização tolerante de notas (brain_recall)."""
import asyncio

from src import mcp_server
from src.config import perfil


class CursorFalso:
    """Devolve uma resposta por execute(), na ordem; registra o SQL para conferir a cascata."""
    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.sqls = []

    async def execute(self, sql, params=None):
        self.sqls.append(sql)
        self._atual = self.respostas.pop(0)

    async def fetchall(self):
        return self._atual


def test_instrucoes_chegam_ao_servidor():
    txt = mcp_server.mcp.instructions
    for termo in ("brain_search", "brain_recall", "00_Agent_Decisions", perfil.dominio, "mascarados"):
        assert termo in txt, termo


def test_find_notes_titulo_exato_para_na_primeira_consulta():
    cur = CursorFalso([(1, "a/Nota.md", "Nota")])
    assert asyncio.run(mcp_server._find_notes(cur, "nota")) == [(1, "a/Nota.md", "Nota")]
    assert len(cur.sqls) == 1


def test_find_notes_cai_para_caminho_e_depois_parcial():
    # caminho sem ".md" e com barra invertida vira "pasta/Arquivo.md"
    cur = CursorFalso([], [(2, "02_Referencias/03_Rede.md", "Rede")])
    assert asyncio.run(mcp_server._find_notes(cur, r"02_Referencias\03_Rede"))[0][0] == 2
    assert len(cur.sqls) == 2

    varias = [(3, "x/Rede A.md", "Rede A"), (4, "y/Rede B.md", "Rede B")]
    cur = CursorFalso([], [], varias)
    assert asyncio.run(mcp_server._find_notes(cur, "Rede")) == varias
    assert "ORDER BY length(title)" in cur.sqls[2]


def test_anotacoes_das_ferramentas():
    """Cliente MCP usa as dicas para aprovar/paralelizar: leitura nunca pode vir marcada como escrita, e vice-versa."""
    ferramentas = {t.name: t.annotations for t in asyncio.run(mcp_server.mcp.list_tools())}
    leitura = ("brain_search", "brain_recall", "brain_graph", "brain_list", "brain_historico", "brain_consolidar")
    escrita = ("brain_store", "brain_record_decision")
    registro = ("brain_feedback",)
    assert set(ferramentas) == {*leitura, *escrita, *registro}
    for nome in leitura:
        assert ferramentas[nome].readOnlyHint is True and ferramentas[nome].openWorldHint is False, nome
    for nome in escrita:   # mesmo título sobrescreve a nota
        a = ferramentas[nome]
        assert a.readOnlyHint is False and a.destructiveHint is True and a.openWorldHint is False, nome
    for nome in registro:  # só acrescenta uma linha de telemetria: não destrói nada
        a = ferramentas[nome]
        assert a.readOnlyHint is False and a.destructiveHint is False and a.openWorldHint is False, nome


def test_recall_corta_nota_grande_em_linha_inteira():
    pequena = "a, b\n" * 10
    assert mcp_server._cortar_recall(pequena, "T") == pequena                     # nada muda (vírgulas inclusive)
    grande = ("linha, com vírgula\n" * 5000)                                       # ~95 mil caracteres
    cortada = mcp_server._cortar_recall(grande, "Portaria X")
    corpo, aviso = cortada.split("\n\n… (nota cortada", 1)
    assert len(corpo) <= mcp_server.MAX_RECALL_CHARS and corpo.endswith("linha, com vírgula")
    assert "de 95.000 caracteres" in aviso and "Portaria X (p. 40)" in aviso
