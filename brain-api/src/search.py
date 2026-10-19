import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo
from psycopg.types.json import Jsonb
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from .config import settings
from .embeddings import OllamaEmbeddingClient

logger = logging.getLogger("cortex.search")

# LGPD: o vault guarda o dado bruto; toda saída (REST e MCP) passa por mask_pii.
_RE_CNS = re.compile(r'(?<!\d)([12789]\d{2}) ?\d{4} ?\d{4} ?(\d{4})(?!\d)')     # 15 dígitos
_RE_CPF_FMT = re.compile(r'(?<!\d)(\d{3})\.\d{3}\.\d{3}-(\d{2})(?!\d)')        # 000.000.000-00
_RE_CPF_RAW = re.compile(r'(?<!\d)(\d{3})\d{6}(\d{2})(?!\d)')                  # 11 dígitos (CPF ou celular)

# Desenhos do Excalidraw viram .md com JSON comprimido: só ruído na busca. Passe [] para incluí-los.
DEFAULT_EXCLUDE_FOLDERS = ["Excalidraw/"]
MAX_CHUNKS_PER_NOTE = 2
RRF_K = 60  # RRF = 1/(k + rank_vetor) + 1/(k + rank_fts) + peso/(k + rank_ficha); o /explorer usa o mesmo k

def cap_per_note(rows, limit: int, per_note_max: int = MAX_CHUNKS_PER_NOTE) -> list:
    """Diversidade: no máximo N trechos por nota (r[1] = note_id), mantendo a ordem do RRF;
    senão uma nota longa ocupa todos os resultados."""
    seen: Dict[int, int] = {}
    out = []
    for r in rows:
        seen[r[1]] = seen.get(r[1], 0) + 1
        if seen[r[1]] <= per_note_max:
            out.append(r)
    return out[:limit]

SQL_VIGENCIA = "SELECT id, vigente_desde, revogada_em, substituida_por FROM notes WHERE id = ANY(%s);"


def hoje() -> date:
    """Data de hoje no fuso de FUSO (o container roda em UTC: às 21h de Brasília já seria amanhã)."""
    return datetime.now(ZoneInfo(settings.fuso)).date() if settings.fuso else date.today()


def estado_vigencia(vigente_desde: Optional[date], revogada_em: Optional[date], substituida_por: Optional[str],
                    hoje: date) -> Optional[str]:
    """"revogada" (revogada_em já passou, ou substituida_por sem data), "futura" (vigente_desde depois de hoje) ou
    None (vale). Revogação marcada para o futuro ainda vale."""
    if (revogada_em is not None and revogada_em <= hoje) or (revogada_em is None and substituida_por):
        return "revogada"
    if vigente_desde is not None and vigente_desde > hoje:
        return "futura"
    return None


def aplicar_vigencia(rows, vigencias: Dict[int, tuple], peso: float, hoje: date) -> list:
    """Multiplica o RRF (r[9]) dos trechos de nota revogada por `peso` e reordena (sort estável: empate mantém a
    ordem do SQL). vigencias = {note_id: (vigente_desde, revogada_em, substituida_por)}. Roda antes do cap_per_note,
    para a vaga da revogada ir para outra nota. peso 1 = só avisa, não mexe na ordem."""
    if peso == 1 or not vigencias:
        return list(rows)
    def score(r):
        v = vigencias.get(r[1])
        return float(r[9]) * (peso if v and estado_vigencia(*v, hoje) == "revogada" else 1)
    return [(*r[:9], s, *r[10:]) for s, r in sorted(((score(r), r) for r in rows), key=lambda x: -x[0])]


def note_stem(file_path: str) -> str:
    """Nome do arquivo sem pasta e sem .md: é o alvo de [[link]] no Obsidian (igual a notes.stem no banco)."""
    name = file_path.rsplit("/", 1)[-1]
    return name[:-3] if name.endswith(".md") else name

def mask_pii(text: str) -> str:
    """CPF → 123.***.***-09 ; CNS → 700 **** **** 1234."""
    if not text:
        return text
    text = _RE_CNS.sub(r'\1 **** **** \2', text)
    text = _RE_CPF_FMT.sub(r'\1.***.***-\2', text)
    return _RE_CPF_RAW.sub(r'\1.***.***-\2', text)

@dataclass
class SearchResult:
    chunk_id: int
    note_id: int
    file_path: str
    title: str
    heading_hierarchy: str
    content: str
    metadata: Dict[str, Any]
    vector_rank: Optional[int]
    text_rank: Optional[int]
    rrf_score: float
    outlinks: List[str]
    backlinks: List[str]
    similarity: Optional[float] = None  # cosseno pergunta x trecho (1 - distância); o rrf_score só ordena
    note_rank: Optional[int] = None     # posição da nota na via de fichas; None = fora das 25
    vigencia: Optional[str] = None      # "revogada" | "futura" | None (vale): estado_vigencia
    revogada_em: Optional[date] = None
    vigente_desde: Optional[date] = None
    substituida_por: Optional[str] = None  # stem da nota que vale no lugar ([[link]])

def max_similarity(results: List[SearchResult]) -> Optional[float]:
    sims = [r.similarity for r in results if r.similarity is not None]
    return max(sims) if sims else None

def is_low_confidence(results: List[SearchResult]) -> bool:
    """Nenhum trecho parecido o bastante com a pergunta: o vault provavelmente não tem a resposta.
    Limite calibrado em tests/calibrar_confianca.py (perguntas do vault x perguntas de fora)."""
    top = max_similarity(results)
    return top is None or top < settings.low_confidence_threshold

async def log_search(db_pool, origem: str, query: str, filtros: Dict[str, Any],
                     results: List[SearchResult], duracao_ms: int) -> Optional[int]:
    """Grava a busca em search_log e devolve o id (para o brain_feedback).
    Falha aqui nunca derruba a busca (devolve None)."""
    if not settings.search_log_enabled:
        return None
    try:
        async with db_pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO search_log (origem, query, filtros, resultados, max_similarity, baixa_confianca, duracao_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (origem, mask_pii(query), Jsonb({k: v for k, v in filtros.items() if v}),
                 Jsonb([{"file_path": r.file_path, "similarity": r.similarity, "rrf_score": r.rrf_score}
                        for r in results]),
                 max_similarity(results), is_low_confidence(results), duracao_ms))
            return (await cur.fetchone())[0]
    except Exception:
        logger.exception("search_log: gravação falhou (busca segue)")
        return None

class HybridSearchEngine:
    def __init__(self, db_pool, embed_client: OllamaEmbeddingClient):
        self.db_pool = db_pool
        self.embed_client = embed_client

    async def search(
        self,
        query: str,
        limit: int = 5,
        folder_prefix: Optional[str] = None,
        expand_graph: bool = True,
        include_folders: Optional[List[str]] = None,
        exclude_folders: Optional[List[str]] = None,
        query_vector: Optional[List[float]] = None,
        fts_mode: Optional[str] = None,
        note_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        fts_mode = fts_mode or settings.fts_mode  # and | or | idf: ver cortex_fts_terms em db/init.sql
        note_weight = settings.note_weight if note_weight is None else note_weight  # via de fichas (007)
        if exclude_folders is None and not include_folders:
            exclude_folders = DEFAULT_EXCLUDE_FOLDERS  # include explícito já restringe; não esconder o que foi pedido
        # 1. Gera embedding da query com prefixo contrastivo (o /explorer passa o vetor pronto para reusá-lo)
        if query_vector is None:
            query_vector = await self.embed_client.get_query_embedding(query)

        # 2. Executa a função SQL stored híbrida RRF
        async with self.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT
                        chunk_id, note_id, file_path, title,
                        heading_hierarchy, content, metadata,
                        vector_rank, text_rank, rrf_score, note_rank
                    FROM cortex_hybrid_search(%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (query, str(query_vector), limit * 3, RRF_K, folder_prefix,
                      include_folders or None, exclude_folders or None, fts_mode, note_weight))
                rows = await cur.fetchall()
                if not rows:
                    return []

                # Vigência (frontmatter): nota revogada cai no placar e deixa a vaga para outra antes do corte por nota
                await cur.execute(SQL_VIGENCIA, (list({r[1] for r in rows}),))
                vigencias = {nid: (ini, fim, nova) for nid, ini, fim, nova in await cur.fetchall()}
                dia = hoje()
                rows = cap_per_note(aplicar_vigencia(rows, vigencias, settings.revogada_peso, dia), limit)

                results = []
                note_ids = [r[1] for r in rows]

                # Cosseno de cada trecho devolvido: o RRF só diz a posição, não se o trecho parece com a pergunta
                await cur.execute(
                    "SELECT id, 1 - (embedding <=> %s::vector) FROM chunks WHERE id = ANY(%s);",
                    (str(query_vector), [r[0] for r in rows]))
                sims = {cid: float(sim) for cid, sim in await cur.fetchall()}

                # 3. Graph Expansion: busca outlinks e backlinks das notas encontradas
                outlinks_map = {nid: [] for nid in note_ids}
                backlinks_map = {note_stem(r[2]): [] for r in rows}

                if expand_graph and note_ids:
                    # Outlinks
                    await cur.execute("""
                        SELECT source_note_id, target_title
                        FROM note_links
                        WHERE source_note_id = ANY(%s);
                    """, (note_ids,))
                    for src_id, target in await cur.fetchall():
                        outlinks_map[src_id].append(target)

                    # Backlinks: [[links]] do Obsidian apontam para o NOME DO ARQUIVO (stem), não para o título
                    await cur.execute("""
                        SELECT DISTINCT nl.target_stem, n.title
                        FROM note_links nl
                        JOIN notes n ON n.id = nl.source_note_id
                        WHERE nl.target_stem = ANY(%s);
                    """, (list(backlinks_map),))
                    for target, src_title in await cur.fetchall():
                        backlinks_map[target].append(src_title)

                for r in rows:
                    ini, fim, nova = vigencias.get(r[1], (None, None, None))
                    results.append(SearchResult(
                        chunk_id=r[0],
                        note_id=r[1],
                        file_path=r[2],
                        title=r[3],
                        heading_hierarchy=mask_pii(r[4]),
                        content=mask_pii(r[5]),
                        metadata=r[6] or {},
                        vector_rank=r[7],
                        text_rank=r[8],
                        rrf_score=float(r[9]),
                        outlinks=outlinks_map.get(r[1], []),
                        backlinks=backlinks_map.get(note_stem(r[2]), []),
                        similarity=sims.get(r[0]),
                        note_rank=r[10],
                        vigencia=estado_vigencia(ini, fim, nova, dia),
                        revogada_em=fim, vigente_desde=ini, substituida_por=mask_pii(nova),
                    ))

                return results
