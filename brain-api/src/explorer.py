"""/explorer — página para auditar a busca (posição e score do vetor, do FTS e do RRF) e navegar
no grafo de [[links]]. htmx e vis-network ficam em static/ (sem CDN) e a CSP só permite a própria origem:
a página mostra o conteúdo do cofre, então nem o navegador pode mandar nada para fora da máquina."""
import logging
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, HTMLResponse

from .api import embed_client, search_engine
from .config import settings
from .database import db
from .search import RRF_K, SearchResult, mask_pii
from . import consolidacao

logger = logging.getLogger("cortex.explorer")

STATIC_DIR = Path(__file__).parent / "static"
TOP_POR_LADO = 25  # cortex_hybrid_search considera só os 25 melhores trechos do vetor e os 25 do FTS
TOP_FICHAS = 25  # e as 25 notas mais parecidas pela ficha
MODOS_FTS = {  # modo → (score que ordena o FTS, o que o trecho precisa ter); ver cortex_fts_terms em db/init.sql
    "and": ("ts_rank_cd", "o trecho precisa ter todos os termos"),
    "or": ("ts_rank", "basta um termo; ordena pela fração dos termos presentes"),
    "idf": ("idf", "basta um termo; soma o peso de raridade (BM25) dos termos presentes"),
}
HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
}

router = APIRouter(prefix="/explorer", include_in_schema=False)


def _html(body: str) -> HTMLResponse:
    return HTMLResponse(body, headers=HEADERS)


def _txt(s) -> str:
    """Todo texto do vault que vai para a página: máscara LGPD e depois escape HTML."""
    return escape(mask_pii(str(s or "")))


def _lista(campo: str) -> Optional[List[str]]:
    """'01_Projetos, 03_Reunioes' → ['01_Projetos', '03_Reunioes']; vazio → None (padrão da busca)."""
    itens = [p.strip().replace("\\", "/") for p in campo.split(",")]
    return [p for p in itens if p] or None


@router.get("")
@router.get("/")
async def pagina():
    return FileResponse(STATIC_DIR / "explorer.html", headers=HEADERS)


@router.get("/status")
async def status():
    async with db.pool.connection() as conn:
        cur = await conn.execute("""
            SELECT (SELECT count(*) FROM notes), (SELECT count(*) FROM notes WHERE file_path ILIKE '%%.pdf'),
                   (SELECT count(*) FROM chunks), (SELECT count(*) FROM note_links);
        """)
        notas, pdfs, trechos, links = await cur.fetchone()
        async with conn.cursor() as c:  # mesma regra do relatório de consolidação: anexo existente não é link quebrado
            rel = await consolidacao.gerar(c, ["quebrados"], vault_path=settings.vault_path)
    quebrados = sum(t for _, t, _ in rel.quebrados)
    return _html(f"{notas} notas ({pdfs} PDFs) · {trechos} trechos · {links} links ({quebrados} quebrados)"
                 f" · FTS padrão: {settings.fts_mode}")


@router.get("/buscar")
async def buscar(
    q: str = "",
    limite: int = Query(10, ge=1, le=20),
    pastas: str = "",
    excluir: str = "",
    fts: str = "",
):
    q = q.strip()
    if not q:
        return _html(AJUDA)
    modo = fts if fts in MODOS_FTS else settings.fts_mode
    try:
        vetor = await embed_client.get_query_embedding(q)
        resultados = await search_engine.search(
            q, limit=limite, include_folders=_lista(pastas), exclude_folders=_lista(excluir), query_vector=vetor,
            fts_mode=modo,
        )
        async with db.pool.connection() as conn:
            async with conn.cursor() as cur:
                termos, scores = await fetch_scores(cur, q, vetor, [r.chunk_id for r in resultados], modo)
    except Exception as e:  # htmx não troca o conteúdo em resposta 5xx: o erro tem que chegar como fragmento
        logger.exception("Explorer: busca falhou")
        return _html(f'<p class="erro">A busca falhou: {escape(str(e))}. Ollama e Postgres estão no ar? '
                     "(<code>docker ps</code>)</p>")
    return _html(render_resultados(q, resultados, scores, termos, modo))


@router.get("/grafo")
async def grafo(ids: List[int] = Query(default=[])):
    if not ids:
        return {"nodes": [], "edges": []}
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            return await fetch_graph(cur, ids[:50])


@router.get("/nota/{note_id}")
async def nota(note_id: int, trecho: Optional[int] = None):
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            dados = await fetch_note(cur, note_id)
    if dados is None:
        return _html('<p class="vazio">Nota não encontrada (apagada ou renomeada no vault?).</p>')
    return _html(render_nota(dados, trecho))


# ---------- consultas ----------

async def fetch_scores(cur, query: str, vetor: List[float], chunk_ids: List[int],
                       modo: str) -> Tuple[List[tuple], Dict[int, tuple]]:
    """cortex_hybrid_search só devolve as posições; aqui saem o cosseno e o score do FTS de cada trecho (None = o FTS
    não aceitou o trecho), pelas mesmas funções SQL que ordenam a busca, e os termos da pergunta com df e peso."""
    await cur.execute("""
        SELECT c.id, 1 - (c.embedding <=> %s::vector), f.score
        FROM chunks c LEFT JOIN cortex_fts_candidates(%s, %s) f ON f.chunk_id = c.id
        WHERE c.id = ANY(%s);
    """, (str(vetor), query, modo, chunk_ids))
    scores = {r[0]: (float(r[1]), r[2]) for r in await cur.fetchall()}
    await cur.execute("SELECT lexeme, df, idf FROM cortex_fts_terms(%s) ORDER BY df, lexeme;", (query,))
    return await cur.fetchall(), scores


async def fetch_graph(cur, note_ids: List[int]) -> dict:
    """Notas pedidas (centro) + vizinhas a 1 salto: links que saem e backlinks que chegam.
    [[link]] resolve pelo nome do arquivo (stem), como no Obsidian; stem repetido → caminho mais curto."""
    await cur.execute("SELECT id, title, file_path FROM notes WHERE id = ANY(%s);", (note_ids,))
    centros = await cur.fetchall()
    await cur.execute("""
        SELECT DISTINCT ON (nl.source_note_id, nl.target_stem)
               nl.source_note_id, nl.target_stem, t.id, t.title, t.file_path
        FROM note_links nl
        LEFT JOIN notes t ON t.stem = nl.target_stem
        WHERE nl.source_note_id = ANY(%s) AND nl.target_stem <> ''
        ORDER BY nl.source_note_id, nl.target_stem, length(t.file_path);
    """, (note_ids,))
    saidas = await cur.fetchall()
    await cur.execute("""
        SELECT DISTINCT s.id, s.title, s.file_path, c.id
        FROM notes c
        JOIN note_links nl ON nl.target_stem = c.stem
        JOIN notes s ON s.id = nl.source_note_id
        WHERE c.id = ANY(%s);
    """, (note_ids,))
    return build_graph(centros, saidas, await cur.fetchall())


async def fetch_note(cur, note_id: int) -> Optional[dict]:
    await cur.execute("SELECT id, title, file_path, updated_at FROM notes WHERE id = %s;", (note_id,))
    nota = await cur.fetchone()
    if not nota:
        return None
    await cur.execute("""
        SELECT id, chunk_index, heading_hierarchy, content, metadata
        FROM chunks WHERE note_id = %s ORDER BY chunk_index;
    """, (note_id,))
    trechos = await cur.fetchall()
    await cur.execute("SELECT tag FROM note_tags WHERE note_id = %s ORDER BY tag;", (note_id,))
    tags = [r[0] for r in await cur.fetchall()]
    await cur.execute("""
        SELECT DISTINCT ON (nl.target_stem) nl.target_title, t.id
        FROM note_links nl LEFT JOIN notes t ON t.stem = nl.target_stem
        WHERE nl.source_note_id = %s AND nl.target_stem <> ''
        ORDER BY nl.target_stem, length(t.file_path);
    """, (note_id,))
    links = await cur.fetchall()
    await cur.execute("""
        SELECT DISTINCT s.id, s.title
        FROM note_links nl JOIN notes s ON s.id = nl.source_note_id
        WHERE nl.target_stem = (SELECT stem FROM notes WHERE id = %s) AND s.id <> %s
        ORDER BY s.title;
    """, (note_id, note_id))
    backlinks = await cur.fetchall()
    return {"nota": nota, "trechos": trechos, "tags": tags, "links": links, "backlinks": backlinks}


# ---------- montagem (funções puras, testadas sem banco) ----------

def build_graph(centros, saidas, entradas) -> dict:
    """Formato do vis-network. Nó 'n<id>' = nota do vault; 'x:<stem>' = [[link]] para nota que não existe."""
    nos: Dict[str, dict] = {}
    arestas = set()

    def no_nota(nid, titulo, caminho, grupo="nota") -> str:
        chave = f"n{nid}"
        if chave not in nos:
            titulo = mask_pii(titulo)
            nos[chave] = {"id": chave, "nota": nid, "group": grupo,
                          "label": titulo if len(titulo) <= 40 else titulo[:38] + "…",
                          "title": f"{titulo}\n{mask_pii(caminho)}"}
        return chave

    for nid, titulo, caminho in centros:
        no_nota(nid, titulo, caminho, "centro")
    for origem, stem, alvo_id, alvo_titulo, alvo_caminho in saidas:
        if alvo_id is None:
            destino = f"x:{stem}"
            nos.setdefault(destino, {"id": destino, "nota": None, "group": "ausente", "label": mask_pii(stem),
                                     "title": f"[[{mask_pii(stem)}]]\nnota inexistente no vault"})
        else:
            destino = no_nota(alvo_id, alvo_titulo, alvo_caminho)
        arestas.add((f"n{origem}", destino))
    for sid, s_titulo, s_caminho, centro_id in entradas:
        arestas.add((no_nota(sid, s_titulo, s_caminho), f"n{centro_id}"))
    return {"nodes": list(nos.values()),
            "edges": [{"id": f"{a}>{b}", "from": a, "to": b} for a, b in sorted(arestas) if a != b]}


def justificativa(v_rank: Optional[int], t_rank: Optional[int], fts_casou: bool, tem_termos: bool,
                  modo: str = "and", n_rank: Optional[int] = None, peso: Optional[float] = None) -> str:
    """Por que o trecho tem esse RRF, em uma frase (+ a ficha da nota, 3ª via do RRF)."""
    peso = settings.note_weight if peso is None else peso
    v = f"1/({RRF_K}+{v_rank})" if v_rank else ""
    t = f"1/({RRF_K}+{t_rank})" if t_rank else ""
    f = f"{peso:g}/({RRF_K}+{n_rank})" if n_rank and peso else ""
    total = ((1 / (RRF_K + v_rank) if v_rank else 0) + (1 / (RRF_K + t_rank) if t_rank else 0)
             + (peso / (RRF_K + n_rank) if f else 0))
    conta = " + ".join(x for x in (v, t, f) if x) + f" = {total:.4f}"
    ficha = f" A ficha da nota (#{n_rank} entre as notas) também somou." if f else ""
    if v and t:
        return (f"Casou nos dois lados: {conta}. "
                "Concordância entre significado e palavras-chave é o que mais pesa no RRF." + ficha)
    if v:
        if not tem_termos:
            motivo = "a pergunta não tem termos indexáveis (só palavras vazias)"
        elif fts_casou:
            motivo = f"o FTS aceitou o trecho, mas ele ficou fora dos {TOP_POR_LADO} melhores"
        elif modo == "and":
            motivo = "o trecho não contém todos os termos da pergunta"
        else:
            motivo = "o trecho não contém nenhum dos termos da pergunta"
        return f"Só pelo vetor: {conta}. O FTS não somou: {motivo}.{ficha}"
    if t:
        return (f"Só pelo FTS: {conta}. O vetor não somou: o trecho ficou fora dos "
                f"{TOP_POR_LADO} mais parecidos semanticamente.{ficha}")
    return (f"Só pela ficha da nota: {conta}. A nota inteira (título, seções e começo) parece com a pergunta "
            f"(#{n_rank} entre as notas); este é o trecho dela mais próximo, fora dos {TOP_POR_LADO} do vetor e do FTS.")


def _metadados(meta: dict) -> str:
    chips = [f'<span class="chip" title="{escape(k)}">{_txt(v)}</span>'
             for k, vals in (meta or {}).items() if vals
             for v in (vals if isinstance(vals, list) else [vals])]
    return f'<div class="chips">{"".join(chips)}</div>' if chips else ""


def _barra(v_rank: Optional[int], t_rank: Optional[int], n_rank: Optional[int] = None) -> str:
    """Largura de cada via = contribuição dela sobre o RRF máximo possível (1º lugar nas três)."""
    peso = settings.note_weight
    maximo = (2 + peso) / (RRF_K + 1)
    bv = 100 * (1 / (RRF_K + v_rank)) / maximo if v_rank else 0
    bt = 100 * (1 / (RRF_K + t_rank)) / maximo if t_rank else 0
    bf = 100 * (peso / (RRF_K + n_rank)) / maximo if n_rank and peso else 0
    return (f'<div class="barra" aria-hidden="true"><i class="bv" style="width:{bv:.1f}%"></i>'
            f'<i class="bt" style="width:{bt:.1f}%"></i><i class="bf" style="width:{bf:.1f}%"></i></div>')


def _termos(termos: List[tuple], modo: str) -> str:
    """Termos da pergunta como o índice os vê, com quantos trechos têm cada um (termo comum = pouco útil)."""
    if not termos:
        return "a pergunta só tem palavras vazias; o FTS não participa."
    chips = "".join(
        f'<span class="chip{" zero" if df == 0 else ""}" title="em {df} trechos · peso BM25 {idf:.2f}">'
        f'{_txt(lex)} · {df}{f" · {idf:.1f}" if modo == "idf" else ""}</span>'
        for lex, df, idf in termos)
    return f'<span class="chips termos">{chips}</span>'


def render_resultados(query: str, resultados: List[SearchResult], scores: Dict[int, tuple],
                      termos: List[tuple], modo: str = "and") -> str:
    if not resultados:
        return (f'<p class="vazio">Nada encontrado para “{escape(query)}”. '
                "Tente termos técnicos ou tire os filtros de pasta.</p>")
    n_fts = sum(1 for r in resultados if r.text_rank)
    n_vet = sum(1 for r in resultados if r.vector_rank)
    n_fic = sum(1 for r in resultados if r.note_rank)
    rotulo, regra = MODOS_FTS[modo]
    topo = (
        '<div class="resumo">'
        f"<p><b>FTS {modo}:</b> {regra}. <b>Termos</b> (trechos que têm cada um): {_termos(termos, modo)}</p>"
        f"<p>{len(resultados)} trechos: {n_vet} com posição no vetor, {n_fts} no FTS, {n_fic} de notas entre as "
        f"{TOP_FICHAS} mais parecidas pela ficha (título, seções e começo da nota). "
        f"Posições contam todos os trechos do índice; no máximo 2 por nota aparecem aqui, como para os agentes.</p>"
        "</div>"
    )
    cards = []
    for i, r in enumerate(resultados, 1):
        cos, ts = scores.get(r.chunk_id, (None, None))  # ausente = nota reindexada no meio da busca
        casou = ts is not None
        cos_txt = f"cos {cos:.3f}" if cos is not None else "cos ?"
        vet = (f'<b>#{r.vector_rank}</b><small>{cos_txt}</small>' if r.vector_rank
               else f'<b>—</b><small>fora do top {TOP_POR_LADO} · {cos_txt}</small>')
        lex = (f'<b>#{r.text_rank}</b><small>{rotulo} {ts or 0:.3f}</small>' if r.text_rank
               else f'<b>—</b><small>{"fora do top " + str(TOP_POR_LADO) if casou else "não casou"}</small>')
        fic = (f'<b>#{r.note_rank}</b><small>nota inteira</small>' if r.note_rank
               else f'<b>—</b><small>fora das {TOP_FICHAS}</small>')
        secao = f" › {_txt(r.heading_hierarchy)}" if r.heading_hierarchy else ""
        cards.append(f"""
<article class="res" data-nota="{r.note_id}">
  <header>
    <span class="pos">{i}</span>
    <button type="button" class="titulo" hx-get="/explorer/nota/{r.note_id}?trecho={r.chunk_id}" hx-target="#nota">{_txt(r.title)}</button>
    <button type="button" class="focar" data-focar="{r.note_id}" title="Centralizar o grafo nesta nota">grafo</button>
  </header>
  <p class="caminho">{_txt(r.file_path)}{secao}</p>
  <div class="diag">
    <div class="lado vet"><span>Vetor</span>{vet}</div>
    <div class="lado fts"><span>FTS</span>{lex}</div>
    <div class="lado fic"><span>Ficha</span>{fic}</div>
    <div class="lado rrf"><span>RRF</span><b>{r.rrf_score:.4f}</b>{_barra(r.vector_rank, r.text_rank, r.note_rank)}</div>
  </div>
  <p class="just">{escape(justificativa(r.vector_rank, r.text_rank, casou, bool(termos), modo, r.note_rank))}</p>
  <div class="trecho" title="Clique para expandir">{escape(r.content)}</div>
  {_metadados(r.metadata)}
</article>""")
    ids = ",".join(dict.fromkeys(str(r.note_id) for r in resultados))
    return f'<div class="lista" data-notas="{ids}">{topo}{"".join(cards)}</div>'


def render_nota(dados: dict, trecho_destacado: Optional[int] = None) -> str:
    nid, titulo, caminho, atualizado = dados["nota"]
    tags = "".join(f'<span class="chip">#{_txt(t)}</span>' for t in dados["tags"])

    def ligacao(alvo_id, texto) -> str:
        if alvo_id is None:
            return f'<li class="ausente" title="nota inexistente no vault">{_txt(texto)}</li>'
        return f'<li><button type="button" class="link" hx-get="/explorer/nota/{alvo_id}" hx-target="#nota">{_txt(texto)}</button></li>'

    links = "".join(ligacao(alvo_id, alvo) for alvo, alvo_id in dados["links"])
    backlinks = "".join(ligacao(sid, s_titulo) for sid, s_titulo in dados["backlinks"])
    trechos = "".join(
        f'<section class="chunk{" destaque" if cid == trecho_destacado else ""}" id="t{cid}">'
        f'<h4>Trecho {idx}{" · " + _txt(sec) if sec else ""}</h4>'
        f"<pre>{_txt(conteudo)}</pre>{_metadados(meta)}</section>"
        for cid, idx, sec, conteudo, meta in dados["trechos"]
    )
    data = atualizado.strftime("%d/%m/%Y %H:%M") if atualizado else "?"
    return f"""
<div class="nota-cab">
  <h2>{_txt(titulo)}</h2>
  <button type="button" class="focar" data-focar="{nid}">grafo</button>
</div>
<p class="caminho">{_txt(caminho)} · atualizada em {data} · {len(dados["trechos"])} trechos</p>
{f'<div class="chips">{tags}</div>' if tags else ""}
<div class="ligacoes">
  <div><h3>Links ({len(dados["links"])})</h3><ul>{links or '<li class="vazio">nenhum</li>'}</ul></div>
  <div><h3>Backlinks ({len(dados["backlinks"])})</h3><ul>{backlinks or '<li class="vazio">nenhum</li>'}</ul></div>
</div>
{trechos}"""


AJUDA = """
<div class="ajuda">
  <p>Digite uma pergunta: a busca roda sozinha 300 ms depois que você para de digitar.</p>
  <p>Cada resultado mostra a posição e o score no <b class="c-vet">vetor</b> (cosseno) e no <b class="c-fts">FTS</b>,
  e o <b>RRF</b> final = 1/(60 + posição no vetor) + 1/(60 + posição no FTS).</p>
  <p>Modos do FTS: <b>and</b> exige todos os termos no trecho; <b>or</b> aceita qualquer termo e ordena pela
  cobertura; <b>idf</b> aceita qualquer termo e soma o peso de raridade de cada um. Troque em “FTS” para comparar.</p>
  <p>No grafo: clique numa nota para lê-la inteira; duplo clique para expandir as vizinhas.</p>
</div>"""
