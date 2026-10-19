from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from typing import List, Optional
from pathlib import Path
from collections import Counter
from datetime import date
import difflib
import json
import logging
import math
import re
import time
import zlib

from .database import db
from .search import (HybridSearchEngine, SQL_VIGENCIA, estado_vigencia, hoje, mask_pii, is_low_confidence,
                     max_similarity, log_search)
from .embeddings import OllamaEmbeddingClient
from .config import perfil, settings
from .graph import GraphEngine
from . import __version__, consolidacao

def montar_instrucoes(p) -> str:
    """Instruções do servidor MCP: fluxo de uso (núcleo, igual para toda área) + o que vem do perfil (descrição,
    domínio, exemplo, PASTAS e fatos fixos em instrucoes.md, regras extras)."""
    exemplo = f" (ex.: {p.exemplo_reformular})" if p.exemplo_reformular else ""
    extras = f"\n{p.instrucoes_extras}\n" if p.instrucoes_extras else ""
    regras = "".join(f"- {r}\n" for r in p.regras)
    return f"""\
{p.descricao}
Tudo roda local; o vault é a fonte da verdade e este índice é só um cache dele.

COMO USAR (fluxo recomendado)
1. Antes de responder sobre {p.dominio},
   chame brain_search com a pergunta em linguagem natural (português). Não chute: o vault tem a resposta local.
2. Resultado relevante mas cortado? Chame brain_recall com o título da nota (ou parte dele, ou o caminho
   do arquivo) para ler a nota inteira.
3. Quer explorar conexões, dependências estruturais ou o caminho entre dois assuntos?
   Chame brain_graph(title="...") para ver a vizinhança de 1 e 2 saltos (pontes e ramos) ou
   brain_graph(title="A", target_title="B") para o menor caminho lógico entre dois tópicos.
4. Cada trecho traz "sim" (similaridade 0-1 com a pergunta). Veio "⚠ BAIXA CONFIANÇA"? O vault provavelmente não
   tem o assunto: não invente a partir dos trechos; reformule uma vez ou diga que o vault não cobre.
   Veio "⚠ REVOGADA"? A nota não vale mais (a busca já a rebaixa): responda pela nota de "Vale [[...]]".
   Para revogar uma nota, ponha no frontmatter dela revogada_em: AAAA-MM-DD e substituida_por: "[[Nota nova]]".
5. Resultado ruim? Reformule com termos técnicos{exemplo}
   ou restrinja com include_folders.
6. Antes de propor uma abordagem técnica, veja se já foi decidida:
   brain_search(query, include_folders=["00_Agent_Decisions"]).
7. Aprendeu algo durável e não óbvio? Grave com brain_store (fato/nota; passe fonte="documento/nota/URL") ou
   brain_record_decision (escolha técnica). Sobrescrever guarda a versão anterior (brain_historico).
   Nunca edite o banco; só o vault.
8. Depois de usar uma busca, diga se ela resolveu: brain_feedback(nº da busca, util=True, nota="título") ou
   util=False. Isso mede a busca com perguntas reais (não muda o ranking).
   Nota com informação velha ou incorreta? util=False + problema="desatualizada" ou "errada", antes de corrigir.
9. Manutenção: brain_consolidar lista quase duplicatas, notas órfãs, links quebrados, lacunas, inbox a triar e
   notas sinalizadas como desatualizadas/erradas.
{extras}
REGRAS
- Dados pessoais sensíveis já vêm mascarados; nunca tente reconstruí-los nem os repita por extenso.
- Cite a nota de origem (título ou caminho) ao usar uma informação.
{regras}"""


INSTRUCOES = montar_instrucoes(perfil)

logger = logging.getLogger("cortex.mcp")
mcp = FastMCP(perfil.nome_mcp, instructions=INSTRUCOES)
if hasattr(mcp, "_mcp_server"):  # o FastMCP não recebe versão; sem isto o cliente vê a versão do pacote mcp
    mcp._mcp_server.version = __version__
# Dicas para o cliente MCP (aprovação/paralelismo): tudo é local (openWorld=False). Gravar com título já existente
# sobrescreve a nota → destrutiva; o auto-linker pode mudar os links a cada chamada → não idempotente.
LEITURA = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
ESCRITA = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
# Só acrescenta uma linha de telemetria no banco (feedback): não lê nem muda o vault, não apaga nada.
REGISTRO = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
PROBLEMAS = consolidacao.PROBLEMAS
embed_client = OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)
search_engine = HybridSearchEngine(db.pool, embed_client)
graph_engine = GraphEngine(db.pool)

@mcp.tool(annotations=LEITURA)
async def brain_search(
    query: str,
    limit: int = 5,
    folder_prefix: str = None,
    include_folders: Optional[List[str]] = None,
    exclude_folders: Optional[List[str]] = None,
) -> str:
    """
    Busca no vault por significado + palavras-chave (vetor + FTS + RRF), com expansão por [[wiki-links]].
    Primeira ferramenta a usar para qualquer pergunta sobre o assunto do vault.
    query: pergunta em português, natural ou com termos técnicos (ex.: "prazo de entrega do relatório anual").
    limit: trechos a retornar (padrão 5, máximo 15). Cada trecho traz o título da nota para usar no brain_recall.
    include_folders: só estas pastas (prefixos, ex.: ["01_Projetos", "02_Referencias"]).
    exclude_folders: ignorar pastas (padrão: ["Excalidraw/"]; passe [] para não excluir nada).
    folder_prefix: legado, equivale a include_folders com uma pasta.
    Decisões anteriores de agentes: include_folders=["00_Agent_Decisions"].
    """
    inicio = time.monotonic()
    results = await search_engine.search(
        query=query, limit=max(1, min(limit, MAX_SEARCH_LIMIT)), folder_prefix=folder_prefix,
        include_folders=include_folders, exclude_folders=exclude_folders,
    )
    busca_id = await log_search(db.pool, "mcp", query,
                                {"folder_prefix": folder_prefix, "include": include_folders, "exclude": exclude_folders},
                                results, int((time.monotonic() - inicio) * 1000))
    rodape = (f"\n\n(Busca nº {busca_id}. Depois de usar: brain_feedback({busca_id}, util=True, nota=\"título\") "
              f"para a nota que respondeu, ou util=False se nenhuma serviu.)" if busca_id else "")
    if not results:
        return (f"Nenhum resultado relevante para: '{query}'. Tente sinônimos/termos técnicos, "
                f"tire filtros de pasta ou use brain_list(folder=...) para ver o que existe.") + rodape

    output = []
    for r in results:
        meta = {k: v for k, v in (r.metadata or {}).items() if v}  # listas vazias só gastam contexto do agente
        meta_str = f" | Metadados: {meta}" if meta else ""
        links_str = f"\n  - Conexões: {', '.join(r.outlinks[:4])}" if r.outlinks else ""
        sim = f" · sim {r.similarity:.2f}" if r.similarity is not None else ""
        output.append(
            f"### [{r.rrf_score:.3f}{sim}] {r.title}\n"
            f"{_aviso_vigencia(r.vigencia, r.vigente_desde, r.revogada_em, r.substituida_por)}"
            f"Seção: {r.heading_hierarchy} | Arquivo: {r.file_path}{meta_str}\n\n"
            f"{r.content}\n"
            f"{links_str}"
        )
    aviso = ""
    if is_low_confidence(results):
        aviso = (f"⚠ BAIXA CONFIANÇA: o trecho mais parecido tem similaridade {max_similarity(results):.2f} "
                 f"(< {settings.low_confidence_threshold:.2f}). O vault provavelmente NÃO tem a resposta: não "
                 f"responda com base nestes trechos; reformule com termos técnicos ou diga que o vault não cobre.\n\n")
    return (aviso + "\n\n---\n\n".join(output) + "\n\n(Para a nota inteira: brain_recall com o título acima.)"
            + rodape)


MAX_SEARCH_LIMIT = 15


def _aviso_vigencia(estado, vigente_desde, revogada_em, substituida_por) -> str:
    """Linha de aviso (com quebra) para nota revogada ou ainda não vigente; "" quando a nota vale."""
    nova = f" Vale [[{mask_pii(substituida_por)}]]: leia com brain_recall." if substituida_por else ""
    if estado == "revogada":
        quando = f" em {revogada_em.isoformat()}" if revogada_em else ""
        return f"⚠ REVOGADA{quando}: não use como regra atual.{nova}\n"
    if estado == "futura":
        return f"⚠ AINDA NÃO VIGENTE: vale a partir de {vigente_desde.isoformat()}.\n"
    return ""


@mcp.tool(annotations=REGISTRO)
async def brain_feedback(busca_id: int, util: bool, nota: Optional[str] = None,
                         comentario: Optional[str] = None, problema: Optional[str] = None) -> str:
    """
    Diz ao Cortex se uma busca resolveu. Não muda o vault nem o ranking: vira avaliação da busca com perguntas reais.
    busca_id: o "Busca nº" que o brain_search mostrou.
    util: True = a nota informada respondeu; False = a nota (ou, sem nota, a busca inteira) não serviu.
    nota: título, parte do título ou caminho da nota (pode ser uma que você achou por outro caminho, ex.: brain_recall).
    comentario: opcional, curto (o que faltou ou o que está errado). Nunca coloque dado pessoal (documento, telefone).
    problema: "desatualizada" (ex.: norma revogada, valor antigo) ou "errada" (fato incorreto), com util=False e
    nota; mande ANTES de corrigir a nota. Ela entra na seção "sinalizadas" do brain_consolidar até o texto dela mudar.
    "resolvida" encerra os avisos de nota que não se edita (PDF, nota oficial) depois de registrar a versão nova.
    """
    if problema is not None:
        problema = problema.strip().lower()
        if problema not in (*PROBLEMAS, consolidacao.RESOLVIDA):
            return f"problema deve ser um de: {', '.join((*PROBLEMAS, consolidacao.RESOLVIDA))}; nada gravado."
        if not nota:
            return "problema exige a nota (título ou caminho); nada gravado."
        if util and problema in PROBLEMAS:
            return f"nota {problema} não resolveu a busca: use util=False; nada gravado."
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM search_log WHERE id = %s;", (busca_id,))
            if not await cur.fetchone():
                return f"Busca nº {busca_id} não existe (use o número que o brain_search mostrou)."
            caminho = None
            if nota:
                rows = await _find_notes(cur, nota)
                if not rows:
                    return f"Nota '{nota}' não encontrada; nada gravado. Use o título que a busca mostrou."
                if len(rows) > 1:
                    cands = "\n".join(f"- {t} (`{p}`)" for _, p, t in rows)
                    return f"'{nota}' casa com {len(rows)} notas; nada gravado. Repita com um destes:\n{cands}"
                caminho = rows[0][1]
            conteudo = None
            if problema in PROBLEMAS:
                conteudo = consolidacao.hash_conteudo(settings.vault_path, caminho)
                if conteudo is None:
                    return f"Arquivo '{caminho}' não está no vault; nada gravado."
            await cur.execute(
                "INSERT INTO search_feedback (search_id, file_path, util, comentario, problema, conteudo_hash) "
                "VALUES (%s, %s, %s, %s, %s, %s);",
                (busca_id, caminho, util, mask_pii(comentario.strip())[:500] if comentario else None, problema,
                 conteudo))
    alvo = f"'{caminho}'" if caminho else "a busca inteira"
    sinal = ""
    if problema in PROBLEMAS:
        sinal = f" Marcada como {problema}: aparece no brain_consolidar até o texto da nota mudar."
    elif problema:
        sinal = " Avisos anteriores desta nota encerrados."
    return (f"Feedback gravado: {alvo} {'resolveu' if util else 'não serviu'} na busca nº {busca_id}.{sinal} "
            "Obrigado.")

@mcp.tool(annotations=LEITURA)
async def brain_recall(title: str) -> str:
    """
    Lê uma nota inteira (todos os trechos em ordem) e suas conexões no grafo.
    title: título exato, parte do título ("Site novo") ou caminho do arquivo
    ("01_Projetos/Site_Novo.md").
    Se o termo casar com várias notas, devolve a lista de candidatas para você escolher.
    """
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            rows = await _find_notes(cur, title)
            if not rows:
                return (f"Nota '{title}' não encontrada. Use brain_search com o assunto "
                        f"ou brain_list(folder=...) para ver os títulos existentes.")
            if len(rows) > 1:
                cands = "\n".join(f"- {t} (`{p}`)" for _, p, t in rows)
                return f"'{title}' casa com {len(rows)} notas; chame brain_recall com um destes títulos:\n{cands}"

            note_id, file_path, clean_title = rows[0]
            await cur.execute("SELECT content FROM chunks WHERE note_id = %s ORDER BY chunk_index ASC;", (note_id,))
            chunks = [r[0] for r in await cur.fetchall()]
            await cur.execute(SQL_VIGENCIA, ([note_id],))
            _, ini, fim, nova = await cur.fetchone() or (None, None, None, None)  # nota saiu no meio da leitura
            aviso = _aviso_vigencia(estado_vigencia(ini, fim, nova, hoje()), ini, fim, nova)

            # Outlinks
            await cur.execute("SELECT target_title FROM note_links WHERE source_note_id = %s;", (note_id,))
            outlinks = [r[0] for r in await cur.fetchall()]

            # Backlinks
            await cur.execute("""
                SELECT DISTINCT s.title
                FROM note_links nl
                JOIN notes s ON s.id = nl.source_note_id
                WHERE nl.target_stem = (SELECT stem FROM notes WHERE id = %s) AND s.id <> %s
                ORDER BY s.title LIMIT 10;
            """, (note_id, note_id))
            backlinks = [r[0] for r in await cur.fetchall()]

    conexoes = []
    if outlinks:
        conexoes.append(f"- **Saída (Outlinks - {len(outlinks)}):** {', '.join(outlinks[:8])}{'...' if len(outlinks) > 8 else ''}")
    if backlinks:
        conexoes.append(f"- **Entrada (Backlinks - {len(backlinks)}):** {', '.join(backlinks[:8])}{'...' if len(backlinks) > 8 else ''}")
    conexoes_str = "\n".join(conexoes) if conexoes else "- Nenhuma conexão registrada"

    return (
        f"# {clean_title} (Arquivo: {file_path})\n\n"
        f"{aviso + chr(10) if aviso else ''}"
        f"**Conexões Diretas no Grafo:**\n{conexoes_str}\n"
        f"*(Para explorar a vizinhança de 2º grau e nós pontes: `brain_graph('{clean_title}')`)*\n\n"
        f"{_cortar_recall(mask_pii(chr(10).join(chunks)), clean_title)}"
    )


MAX_RECALL_CHARS = 40_000  # ~10 mil tokens: um PDF longo inteiro (centenas de milhares de caracteres) afoga o agente


def _cortar_recall(texto: str, titulo: str) -> str:
    """Nota grande demais para o contexto do agente: devolve o começo e diz como achar o resto."""
    if len(texto) <= MAX_RECALL_CHARS:
        return texto
    corte = texto.rfind("\n", 0, MAX_RECALL_CHARS)
    corte = corte if corte > MAX_RECALL_CHARS // 2 else MAX_RECALL_CHARS
    mil = lambda n: f"{n:,}".replace(",", ".")
    return (texto[:corte] + f"\n\n… (nota cortada: mostrados {mil(corte)} de {mil(len(texto))} caracteres. Para o "
            f"resto, brain_search com os termos do que procura e include_folders na pasta desta nota; os trechos "
            f"trazem a seção ou a página, ex.: \"{titulo} (p. 40)\".)")


@mcp.tool(annotations=LEITURA)
async def brain_graph(
    title: str,
    target_title: Optional[str] = None,
    hops: int = 2,
    direction: str = "both",
    limit_per_hop: int = 15,
) -> str:
    """
    Navega no grafo de conhecimento de [[wiki-links]] do vault (1 ou 2 saltos) ou acha o menor caminho entre notas.
    title: nota inicial (título, stem ou caminho, ex.: "00_MAPA", "Site novo").
    target_title: opcional; se informado, calcula a menor rota de raciocínio (shortest path) entre as duas notas.
    hops: profundidade da vizinhança (1 ou 2 saltos, padrão 2).
    direction: sentido das conexões ("both" bidirecional, "out" outlinks que saem, "in" backlinks que apontam para a nota).
    limit_per_hop: limite de nós listados por categoria (padrão 15).
    """
    if target_title:
        # Modo 1: Path Finding / Shortest Path
        path_res = await graph_engine.find_shortest_path(
            source_key=title, target_key=target_title, max_depth=4, direction=direction
        )
        if not path_res:
            return (
                f"Nenhum caminho de até 4 saltos encontrado entre '{title}' e '{target_title}' "
                f"no sentido '{direction}'. Verifique se ambas as notas existem ou use brain_graph sem target_title."
            )
        if path_res.length == 0:
            return f"As chaves '{title}' e '{target_title}' referem-se à mesma nota: {path_res.source.title}."

        linhas = [
            f"## 🧭 Rota no Grafo de Conhecimento ({path_res.length} saltos)\n",
            f"**Origem:** [{path_res.source.title}] (`{path_res.source.file_path}`)",
            f"**Destino:** [{path_res.target.title}] (`{path_res.target.file_path}`)\n",
            "### Trilha de Conexões:",
        ]
        for idx, (u, dir_sym, v) in enumerate(path_res.steps, 1):
            linhas.append(f"{idx}. **{u.title}** {dir_sym} **{v.title}** (`{v.file_path}`)")
        return "\n".join(linhas)

    # Modo 2: Traversal 1 ou 2 saltos
    res = await graph_engine.traverse(
        root_key=title, hops=hops, direction=direction, limit_per_hop=limit_per_hop
    )
    if not res:
        return (
            f"Nota '{title}' não encontrada no vault para expansão do grafo. "
            f"Tente parte do título ou consulte brain_list()."
        )

    out = [
        f"# 🕸️ Grafo de Conhecimento: {res.root.title}\n",
        f"**Arquivo:** `{res.root.file_path}` | **Pasta:** `{res.root.folder}` | "
        f"**Profundidade:** {res.hops} salto(s) | **Nós no subgrafo:** {res.total_nodes_visited}\n",
    ]

    out.append("## 📍 Hop 1 — Conexões Diretas")
    if res.hop1_outlinks:
        out.append(f"### Links de Saída (Outlinks - {len(res.hop1_outlinks)}):")
        for n in res.hop1_outlinks:
            out.append(f"- **{n.title}** (`{n.file_path}`) [_{n.folder}_]")
    else:
        out.append("### Links de Saída: Nenhum")

    if res.hop1_backlinks:
        out.append(f"\n### Backlinks (Notas que referenciam esta - {len(res.hop1_backlinks)}):")
        for n in res.hop1_backlinks:
            out.append(f"- **{n.title}** (`{n.file_path}`) [_{n.folder}_]")
    else:
        out.append("\n### Backlinks: Nenhum")

    if res.bridges:
        out.append("\n## 🌉 Pontes Conceituais (Hop 2)")
        out.append("*(Nós de 2º grau que convergem conexões de múltiplos vizinhos diretos)*")
        for b_node, stems in res.bridges:
            stems_fmt = ", ".join(f"`[[{s}]]`" for s in stems[:4])
            out.append(f"- **{b_node.title}** (`{b_node.file_path}`) — Conectado via: {stems_fmt}")

    if res.hop2_nodes:
        out.append(f"\n## 🔗 Subtópicos e Vizinhança Expandida (Hop 2 - {len(res.hop2_nodes)}):")
        for n in res.hop2_nodes:
            out.append(f"- **{n.title}** (`{n.file_path}`) [_{n.folder}_]")

    if res.unresolved_links:
        out.append(f"\n## ⚠️ Links Pendentes (Sem arquivo no vault):")
        out.append(", ".join(f"`[[{u}]]`" for u in res.unresolved_links[:8]))

    out.append(f"\n\n*(Para ler o conteúdo de qualquer nota: use brain_recall(title=...))*\n"
               f"*(Para testar a rota lógica com outro tópico: use brain_graph('{res.root.title}', target_title=...))*")

    return "\n".join(out)

async def _find_notes(cur, key: str) -> list:
    """Título exato → caminho do arquivo → parte do título. Retorna 1 nota ou até 10 candidatas."""
    key = key.strip()
    path = key.replace("\\", "/")
    path = path if path.lower().endswith(".md") else path + ".md"
    await cur.execute("SELECT id, file_path, title FROM notes WHERE title ILIKE %s "
                      "ORDER BY (file_path ILIKE '%%.md') DESC, length(file_path), id LIMIT 1;", (key,))
    rows = await cur.fetchall()
    if not rows:
        await cur.execute("SELECT id, file_path, title FROM notes WHERE file_path ILIKE %s OR file_path ILIKE %s LIMIT 1;",
                          (path, "%/" + path))
        rows = await cur.fetchall()
    if not rows:
        like = "%" + key.replace("%", r"\%").replace("_", r"\_") + "%"
        await cur.execute("SELECT id, file_path, title FROM notes WHERE title ILIKE %s OR file_path ILIKE %s "
                          "ORDER BY length(title) LIMIT 10;", (like, like))
        rows = await cur.fetchall()
    return rows


@mcp.tool(annotations=ESCRITA)
async def brain_store(title: str, content: str, folder: str = "00_Inbox", auto_link: bool = True,
                      fonte: Optional[str] = None) -> str:
    """
    Grava uma nota Markdown no vault (fica pesquisável em ~10 s). Use para conhecimento durável e não óbvio.
    title: título da nota (vira o nome do arquivo; mesmo título sobrescreve, e a versão anterior vai para o
    histórico: brain_historico).
    content: Markdown; use [[Título de outra nota]] para ligar notas. Nunca grave dado pessoal (documento, telefone) por extenso.
    folder: pasta relativa ao vault (padrão "00_Inbox"; ex.: "02_Referencias").
    auto_link: anexa "## Conexões Sugeridas" com [[links]] para as notas mais parecidas.
    fonte: de onde veio a informação (documento, nota, URL, sessão); vai para o frontmatter com a data de hoje.
    Para escolhas técnicas use brain_record_decision.
    """
    motivo = texto_ruim(content)
    if motivo:
        return f"Erro: nada gravado, o conteúdo {motivo}. Grave o texto da nota, não uma mensagem de erro ou repetição."
    if fonte:
        content = _com_procedencia(content, fonte)
    return await _write_note(title, content, folder, auto_link)


# Porta de qualidade antes de gravar (A4 do ai-memory: chars mínimos, entropia, repetição). Calibrada nas 577 notas do
# vault em 2026-10-07: entropia mínima 4,61 bits/caractere, compressão zlib máxima 6,9 (páginas raspadas do gov.br),
# repetição máxima 0,52 nas notas de até 300 palavras. Os limites ficam bem longe disso: só barra lixo. Texto curto
# tem entropia baixa por natureza ("linha velha" = 2,9): entropia só a partir de 64 caracteres, repetição de 20 palavras.
# Linhas de tabela e blocos de código ficam fora das contas: nota só de dados (tabela 0/1, calendário ✔/—, matriz
# Sim/Não, escala) é legítima e tem entropia 1,8-2,7 (revisão de 2026-10-07). Lista de CPF mascarado = 2,9: passa.
MIN_ENTROPIA = 2.0
_RE_CODIGO = re.compile(r"^ {0,3}(```|~~~).*?^ {0,3}\1[^\n]*$", re.S | re.M)
_RE_TABELA = re.compile(r"^\s*\|.*$", re.M)
MAX_COMPRESSAO = 10.0
MAX_REPETICAO = 0.7


def texto_ruim(content: str) -> Optional[str]:
    """Motivo para recusar o conteúdo (quase vazio, entropia baixa, repetitivo) ou None. Frontmatter não conta."""
    corpo = _RE_FM_BLOCO.sub("", content, count=1)
    if not "".join(corpo.split()):
        return "está vazio"
    corpo = _RE_TABELA.sub("", _RE_CODIGO.sub("", corpo))   # dados e código não são prosa
    letras = "".join(corpo.split())
    if not letras:
        return None
    total = len(letras)
    entropia = -sum(n / total * math.log2(n / total) for n in Counter(letras).values())
    if total >= 64 and entropia < MIN_ENTROPIA:
        return f"tem entropia baixa ({entropia:.1f} bits/caractere; nota de verdade fica acima de 4,5)"
    bruto = corpo.encode("utf-8")
    if len(bruto) >= 400 and len(bruto) / len(zlib.compress(bruto, 9)) > MAX_COMPRESSAO:
        return "é o mesmo trecho repetido muitas vezes"
    palavras = re.findall(r"\w+", corpo.lower())
    if 20 <= len(palavras) <= 300 and 1 - len(set(palavras)) / len(palavras) > MAX_REPETICAO:
        return "repete as mesmas palavras (menos de 30% distintas)"
    return None


def _com_procedencia(content: str, fonte: str) -> str:
    """Frontmatter ganha fonte e gravado_em (metadados [source; added] do AgentMemoryRepo). O parser não indexa o
    frontmatter: a procedência não mexe na busca."""
    fonte = json.dumps(" ".join(fonte.split()), ensure_ascii=False)  # string JSON = escalar YAML válido (\\ e ")
    campos = f"fonte: {fonte}\ngravado_em: {date.today().isoformat()}\n"
    m = _RE_FM_BLOCO.match(content)
    if not m:
        return f"---\n{campos}---\n{content}"
    corpo = re.sub(r"(?m)^(fonte|gravado_em):.*\r?\n", "", m.group(1))  # regravação substitui, não repete
    return f"---\n{corpo}{campos}---\n{content[m.end():]}"


_RE_FM_BLOCO = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*\r?\n", re.S | re.M)  # aceita vazio e CRLF


@mcp.tool(annotations=LEITURA)
async def brain_historico(title: str, versao: Optional[str] = None) -> str:
    """
    Versões anteriores de uma nota que brain_store/brain_record_decision sobrescreveram (mais recente primeiro).
    title: título, parte do título ou caminho (como no brain_recall).
    versao: nome de uma versão da lista (ex.: "2026-10-06_141500") para ver o que mudou dela para a nota atual.
    """
    vault_base = Path(settings.vault_path).resolve()
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            rows = await _find_notes(cur, title)
    if len(rows) > 1:
        cands = "\n".join(f"- {t} (`{p}`)" for _, p, t in rows)
        return f"'{title}' casa com {len(rows)} notas; chame brain_historico com um destes títulos:\n{cands}"
    rel = rows[0][1] if rows else title.replace("\\", "/").removesuffix(".md") + ".md"
    alvo = (vault_base / rel).resolve()
    if not alvo.is_relative_to(vault_base):
        return "Erro: caminho fora do vault."
    pasta = _pasta_historico(vault_base, alvo)
    versoes = sorted(pasta.glob("*.md"), reverse=True) if pasta.is_dir() else []
    if not versoes:
        return f"'{rel}' não tem versões anteriores (nunca foi sobrescrita pelo Cortex)."
    if not versao:
        linhas = [f"- {v.stem} ({v.stat().st_size} bytes)" for v in versoes[:30]]
        return (f"# Histórico de '{rel}' ({len(versoes)} versões anteriores)\n" + "\n".join(linhas)
                + "\n\n(Para ver o que mudou: brain_historico(title, versao=\"...\").)")
    escolhida = next((v for v in versoes if v.stem == versao.strip()), None)
    if escolhida is None:
        return f"Versão '{versao}' não existe. Versões: {', '.join(v.stem for v in versoes[:10])}"
    antes = escolhida.read_text(encoding="utf-8", errors="replace").splitlines()
    agora = alvo.read_text(encoding="utf-8", errors="replace").splitlines() if alvo.exists() else []
    diff = list(difflib.unified_diff(antes, agora, f"versão {escolhida.stem}", "atual", lineterm="", n=1))
    corpo = "\n".join(diff[:120]) + ("\n… (diff cortado em 120 linhas)" if len(diff) > 120 else "")
    return mask_pii(f"# '{rel}': versão {escolhida.stem} → atual\n```diff\n{corpo or '(sem diferença)'}\n```")


@mcp.tool(annotations=ESCRITA)
async def brain_record_decision(topic: str, decision: str, rejected: str, rationale: str) -> str:
    """
    Registra uma decisão técnica (ADR) em 00_Agent_Decisions/ para que os agentes
    não repitam abordagens já descartadas. Consulte antes com
    brain_search(query, include_folders=["00_Agent_Decisions"]).
    topic: assunto curto. decision: o que foi adotado. rejected: o que foi descartado e por quê.
    rationale: contexto/problema que motivou a decisão.
    """
    today = date.today().isoformat()
    content = (
        f"---\ntipo: adr-agente\ndata: {today}\nstatus: ativo\ntags: [decisao]\n---\n"
        f"# ADR: {topic}\n\n"
        f"## Contexto e Problema\n{rationale}\n\n"
        f"## Decisão Adotada\n{decision}\n\n"
        f"## Abordagens Descartadas e Por Quê\n{rejected}\n"
    )
    return await _write_note(f"ADR {today} {topic}", content, DECISIONS_FOLDER)


DECISIONS_FOLDER = "00_Agent_Decisions"
_RE_BAD_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')  # proibidos no Windows/OneDrive


async def _write_note(title: str, content: str, folder: str, auto_link: bool = True) -> str:
    """Grava nota .md dentro do vault; recusa pasta que escape dele; anexa links sugeridos."""
    vault_base = Path(settings.vault_path).resolve()
    target_dir = (vault_base / folder).resolve()
    if not target_dir.is_relative_to(vault_base):
        return f"Erro: pasta '{folder}' fora do vault. Use um caminho relativo dentro do vault."
    clean_title = _RE_BAD_FILENAME.sub("-", title).strip(" .")
    if not clean_title:
        return "Erro: título vazio."
    target = target_dir / f"{clean_title}.md"
    links_msg = ""
    if auto_link:
        rel_path = target.relative_to(vault_base).as_posix()
        try:
            content, n = await _with_auto_links(title, content, rel_path)
            links_msg = f" {n} conexão(ões) sugerida(s) pelo auto-linker."
        except Exception as e:  # Ollama/banco fora do ar não pode impedir a gravação
            logger.warning(f"Auto-linker falhou para {rel_path}: {e}")
    target_dir.mkdir(parents=True, exist_ok=True)
    versao_msg = ""
    if target.exists():
        anterior = target.read_text(encoding="utf-8", errors="replace")
        if anterior == content:
            return f"Memória '{clean_title}' já estava gravada com este conteúdo (nada mudou)."
        guardada = _guardar_versao(vault_base, target, anterior)
        versao_msg = f" Versão anterior guardada em '{guardada}' (brain_historico('{clean_title}'))."
    target.write_text(content, encoding="utf-8")
    return (f"Memória '{clean_title}' gravada com sucesso em '{folder}/{clean_title}.md'."
            f"{links_msg}{versao_msg} Sincronização automática em andamento.")


HISTORICO_DIR = ".cortex_historico"  # pasta com ponto: o indexador e o Obsidian ignoram (não vira nota nem resultado)


def _pasta_historico(vault_base: Path, target: Path) -> Path:
    """.cortex_historico/<caminho da nota, com extensão>/ — uma pasta por arquivo, um arquivo por versão
    ("X.md" e "X.pdf" de mesmo nome não dividem histórico)."""
    return vault_base / HISTORICO_DIR / target.relative_to(vault_base)


def _guardar_versao(vault_base: Path, target: Path, anterior: str) -> str:
    """Antes de sobrescrever, copia a versão atual para o histórico (o vault não é um repositório git).
    Ideia do AgentMemoryRepo (memória com histórico) e do OpenMemory (MemoryStatusHistory)."""
    pasta = _pasta_historico(vault_base, target)
    pasta.mkdir(parents=True, exist_ok=True)
    carimbo = time.strftime("%Y-%m-%d_%H%M%S")
    destino = pasta / f"{carimbo}.md"
    n = 1
    while destino.exists():  # duas gravações no mesmo segundo
        n += 1
        destino = pasta / f"{carimbo}_{n}.md"
    destino.write_text(anterior, encoding="utf-8")
    return destino.relative_to(vault_base).as_posix()


_AUTO_LINK_HEADER = "## Conexões Sugeridas (Auto-Linker)"
_RE_AUTO_LINK_SECTION = re.compile(r"\n*(?:---\n+)?" + re.escape(_AUTO_LINK_HEADER) + r"[\s\S]*$")
_RE_FRONTMATTER = re.compile(r"\A---\n[\s\S]*?\n---\n")
_RE_WIKILINK_TARGET = re.compile(r"\[\[([^\]|#]+)")


async def _with_auto_links(title: str, content: str, rel_path: str) -> tuple[str, int]:
    """Anexa até N [[links]] para as notas mais parecidas (título + começo do texto).
    Refaz a seção se ela já existir (nota regravada) e não repete links que o texto já tem."""
    content = _RE_AUTO_LINK_SECTION.sub("", content).rstrip() + "\n"
    body = _RE_FRONTMATTER.sub("", content)
    already = {t.strip().rsplit("/", 1)[-1].lower() for t in _RE_WIKILINK_TARGET.findall(body)}
    [vec] = await embed_client.get_document_embeddings([f"{title}\n{body[:600]}"])
    async with db.pool.connection() as conn:
        cur = await conn.execute("""
            SELECT n.stem, max(1 - (c.embedding <=> %s::vector)) AS sim
            FROM chunks c JOIN notes n ON n.id = c.note_id
            WHERE n.file_path <> %s AND n.file_path NOT LIKE 'Excalidraw/%%'
            GROUP BY n.stem
            ORDER BY sim DESC
            LIMIT %s;
        """, (str(vec), rel_path, settings.auto_link_max + len(already)))
        rows = await cur.fetchall()
    picks = [(stem, sim) for stem, sim in rows
             if sim >= settings.auto_link_threshold and stem.lower() not in already][:settings.auto_link_max]
    if not picks:
        return content, 0
    lines = "\n".join(f"- [[{stem}]] (Similaridade: {sim:.0%})" for stem, sim in picks)
    return f"{content}\n---\n{_AUTO_LINK_HEADER}\n{lines}\n", len(picks)

@mcp.tool(annotations=LEITURA)
async def brain_list(tag: Optional[str] = None, folder: Optional[str] = None, limit: int = 20) -> str:
    """
    Lista notas (mais recentes primeiro) para descobrir o que existe e pegar títulos para o brain_recall.
    folder: prefixo da pasta (ex.: "02_Referencias", "00_Agent_Decisions"). tag: parte de uma tag.
    limit: padrão 20, máximo 100.
    """
    limit = max(1, min(limit, 100))
    where, params = [], []
    if tag:
        where.append("EXISTS (SELECT 1 FROM note_tags nt WHERE nt.note_id = n.id AND nt.tag ILIKE %s)")
        params.append(f"%{tag}%")
    if folder:
        where.append("n.file_path ILIKE %s")
        params.append(folder.replace("\\", "/").rstrip("/") + "%")
    sql = ("SELECT n.title, n.file_path, n.updated_at FROM notes n "
           + ("WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY n.updated_at DESC LIMIT %s;")
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (*params, limit))
            rows = await cur.fetchall()

    if not rows:
        return "Nenhuma nota encontrada com esses filtros."

    lines = [f"- **{r[0]}** (`{r[1]}`) — Atualizado em: {r[2]}" for r in rows]
    return "\n".join(lines)


@mcp.tool(annotations=LEITURA)
async def brain_consolidar(secoes: Optional[List[str]] = None, limite: int = 10, dias: int = 30) -> str:
    """
    Relatório de manutenção da memória (só leitura): quase duplicatas, notas órfãs, links quebrados, lacunas (buscas
    sem resposta e feedback "não serviu"), inbox a triar, notas mais úteis e notas sinalizadas (desatualizada/errada).
    Use de tempos em tempos ou antes de reorganizar uma pasta; quem junta, liga ou apaga notas é você, no vault
    (brain_store), nunca o banco.
    secoes: subconjunto de ["duplicatas", "orfas", "quebrados", "lacunas", "inbox", "uteis", "sinalizadas"]
    (padrão: todas).
    limite: itens por seção (padrão 10, máximo 50). dias: janela das lacunas (padrão 30).
    """
    validas = [s for s in (secoes or consolidacao.SECOES) if s in consolidacao.SECOES]
    if not validas:
        return f"Seções válidas: {', '.join(consolidacao.SECOES)}."
    limite, dias = max(1, min(limite, 50)), max(1, min(dias, 365))
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            rel = await consolidacao.gerar(cur, validas, limite, dias, vault_path=settings.vault_path)
    return consolidacao.formatar(rel, validas, limite, dias)
