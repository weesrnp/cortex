from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from .database import db
import time

from .search import HybridSearchEngine, mask_pii, log_search
from .embeddings import OllamaEmbeddingClient
from .config import settings
from .graph import GraphEngine
from . import consolidacao

router = APIRouter()
embed_client = OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)
search_engine = HybridSearchEngine(db.pool, embed_client)
graph_engine = GraphEngine(db.pool)

class SearchResponseItem(BaseModel):
    chunk_id: int
    note_id: int
    file_path: str
    title: str
    heading_hierarchy: str
    content: str
    metadata: Dict[str, Any]
    rrf_score: float
    vector_rank: Optional[int]
    text_rank: Optional[int]
    note_rank: Optional[int]  # posição da nota na via de fichas
    similarity: Optional[float]  # cosseno com a pergunta; abaixo de low_confidence_threshold = vault não cobre
    outlinks: List[str]
    backlinks: List[str]

@router.get("/health")
async def health():
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT count(*) FROM notes;")
            note_count = (await cur.fetchone())[0]
            await cur.execute("SELECT count(*) FROM chunks;")
            chunk_count = (await cur.fetchone())[0]
            await cur.execute("SELECT count(*) FROM notes WHERE ficha_embedding IS NOT NULL;")
            ficha_count = (await cur.fetchone())[0]
    return {
        "status": "healthy",
        "notes_indexed": note_count,
        "chunks_indexed": chunk_count,
        "notes_with_ficha": ficha_count,  # via de fichas: igual a notes_indexed depois do scan
        "embedding_model": settings.embedding_model
    }

@router.get("/search", response_model=List[SearchResponseItem])
async def search_endpoint(
    q: str = Query(..., description="Termo de pesquisa"),
    limit: int = Query(5, ge=1, le=20),
    folder: Optional[str] = Query(None, description="Filtrar por subpasta (ex: '01_Projetos')"),
    expand_graph: bool = Query(True, description="Expandir links e backlinks"),
    include: Optional[List[str]] = Query(None, description="Só estas pastas (repetível: ?include=01_&include=02_)"),
    exclude: Optional[List[str]] = Query(None, description="Ignorar estas pastas (padrão: Excalidraw/)")
):
    try:
        inicio = time.monotonic()
        results = await search_engine.search(
            query=q,
            limit=limit,
            folder_prefix=folder,
            expand_graph=expand_graph,
            include_folders=include,
            exclude_folders=exclude,
        )
        await log_search(db.pool, "rest", q, {"folder_prefix": folder, "include": include, "exclude": exclude},
                         results, int((time.monotonic() - inicio) * 1000))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recall/{title}")
async def recall_endpoint(title: str):
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT id, file_path, title, updated_at
                FROM notes
                WHERE title ILIKE %s OR file_path ILIKE %s;
            """, (title, f"%{title}%"))
            note = await cur.fetchone()
            if not note:
                raise HTTPException(status_code=404, detail="Nota não encontrada.")

            note_id, file_path, clean_title, updated_at = note

            await cur.execute("""
                SELECT chunk_index, heading_hierarchy, content, metadata
                FROM chunks
                WHERE note_id = %s
                ORDER BY chunk_index ASC;
            """, (note_id,))
            chunks = await cur.fetchall()

            await cur.execute("SELECT tag FROM note_tags WHERE note_id = %s;", (note_id,))
            tags = [r[0] for r in await cur.fetchall()]

            await cur.execute("SELECT target_title FROM note_links WHERE source_note_id = %s;", (note_id,))
            outlinks = [r[0] for r in await cur.fetchall()]

            await cur.execute("""
                SELECT DISTINCT n.title
                FROM note_links nl
                JOIN notes n ON n.id = nl.source_note_id
                WHERE nl.target_stem = (SELECT stem FROM notes WHERE id = %s);
            """, (note_id,))
            backlinks = [r[0] for r in await cur.fetchall()]

    return {
        "title": clean_title,
        "file_path": file_path,
        "updated_at": updated_at,
        "tags": tags,
        "outlinks": outlinks,
        "backlinks": backlinks,
        "chunks": [{"heading": mask_pii(c[1]), "content": mask_pii(c[2]), "metadata": c[3]} for c in chunks]
    }


@router.get("/graph/traverse")
async def graph_traverse_endpoint(
    title: str = Query(..., description="Título, stem ou caminho da nota raiz"),
    hops: int = Query(2, ge=1, le=2, description="Profundidade do salto (1 ou 2)"),
    direction: str = Query("both", description="Direção dos links: both | out | in"),
    limit: int = Query(15, ge=1, le=50, description="Limite por categoria"),
):
    try:
        res = await graph_engine.traverse(root_key=title, hops=hops, direction=direction, limit_per_hop=limit)
        if not res:
            raise HTTPException(status_code=404, detail=f"Nota '{title}' não encontrada no vault.")
        return {
            "root": {"id": res.root.id, "title": res.root.title, "file_path": res.root.file_path, "folder": res.root.folder, "stem": res.root.stem},
            "hops": res.hops,
            "direction": res.direction,
            "total_nodes_visited": res.total_nodes_visited,
            "hop1_outlinks": [{"id": n.id, "title": n.title, "file_path": n.file_path, "folder": n.folder, "stem": n.stem} for n in res.hop1_outlinks],
            "hop1_backlinks": [{"id": n.id, "title": n.title, "file_path": n.file_path, "folder": n.folder, "stem": n.stem} for n in res.hop1_backlinks],
            "bridges": [{"node": {"id": b[0].id, "title": b[0].title, "file_path": b[0].file_path, "folder": b[0].folder, "stem": b[0].stem}, "via_stems": b[1]} for b in res.bridges],
            "hop2_nodes": [{"id": n.id, "title": n.title, "file_path": n.file_path, "folder": n.folder, "stem": n.stem} for n in res.hop2_nodes],
            "unresolved_links": res.unresolved_links,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/path")
async def graph_path_endpoint(
    source: str = Query(..., description="Nota de origem"),
    target: str = Query(..., description="Nota de destino"),
    max_depth: int = Query(4, ge=1, le=5, description="Profundidade máxima de busca"),
    direction: str = Query("both", description="Direção: both | out | in"),
):
    try:
        res = await graph_engine.find_shortest_path(source_key=source, target_key=target, max_depth=max_depth, direction=direction)
        if not res:
            raise HTTPException(status_code=404, detail=f"Nenhum caminho encontrado entre '{source}' e '{target}'.")
        return {
            "source": {"id": res.source.id, "title": res.source.title, "file_path": res.source.file_path, "stem": res.source.stem},
            "target": {"id": res.target.id, "title": res.target.title, "file_path": res.target.file_path, "stem": res.target.stem},
            "length": res.length,
            "steps": [{"from_title": s[0].title, "dir": s[1], "to_title": s[2].title, "to_path": s[2].file_path} for s in res.steps],
            "path_nodes": [{"id": n.id, "title": n.title, "file_path": n.file_path, "stem": n.stem} for n in res.path_nodes],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consolidar")
async def consolidar_endpoint(
    secoes: Optional[List[str]] = Query(None, description="duplicatas, orfas, quebrados, lacunas, inbox, uteis, sinalizadas"),
    limite: int = Query(10, ge=1, le=50),
    dias: int = Query(30, ge=1, le=365),
):
    """Relatório de consolidação (src/consolidacao.py) em JSON, só leitura."""
    validas = [s for s in (secoes or consolidacao.SECOES) if s in consolidacao.SECOES]
    if not validas:
        raise HTTPException(status_code=422, detail=f"Seções válidas: {', '.join(consolidacao.SECOES)}")
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            rel = await consolidacao.gerar(cur, validas, limite, dias, vault_path=settings.vault_path)
    return consolidacao.como_dict(rel)
