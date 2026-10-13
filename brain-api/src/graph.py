"""Motor de Grafo de Conhecimento e Traversal de [[Wiki-links]] do Obsidian.

Implementa navegação multi-hop (1 e 2 saltos) e menor caminho (path finding)
direto sobre a tabela note_links do PostgreSQL, sem dependências externas.
LGPD: títulos e caminhos de notas são mascarados quando necessário.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import logging

from .search import mask_pii

logger = logging.getLogger("cortex.graph")


@dataclass
class GraphNode:
    id: int
    title: str
    file_path: str
    folder: str
    stem: str


@dataclass
class EdgeInfo:
    source_id: int
    source_title: str
    target_id: Optional[int]
    target_stem: str
    target_title: str
    direction: str  # 'out' (source -> target) ou 'in' (target -> source)
    resolved: bool


@dataclass
class TraversalResult:
    root: GraphNode
    hops: int
    direction: str
    hop1_outlinks: List[GraphNode]
    hop1_backlinks: List[GraphNode]
    hop2_nodes: List[GraphNode]
    bridges: List[Tuple[GraphNode, List[str]]]  # (nó_hop2, lista_de_stems_intermediários)
    unresolved_links: List[str]  # links para notas que ainda não existem
    total_nodes_visited: int


@dataclass
class PathResult:
    source: GraphNode
    target: GraphNode
    length: int
    steps: List[Tuple[GraphNode, str, GraphNode]]  # (u, '->' ou '<-', v)
    path_nodes: List[GraphNode]


def _folder_from_path(file_path: str) -> str:
    """Extrai a pasta raiz ou caminho de pasta da nota."""
    parts = file_path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else ""


class GraphEngine:
    def __init__(self, db_pool):
        self.db_pool = db_pool

    async def get_node_by_id(self, cur, note_id: int) -> Optional[GraphNode]:
        await cur.execute(
            "SELECT id, title, file_path, stem FROM notes WHERE id = %s;",
            (note_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return GraphNode(
            id=row[0],
            title=mask_pii(row[1]),
            file_path=row[2],
            folder=_folder_from_path(row[2]),
            stem=row[3],
        )

    async def resolve_note(self, cur, query: str) -> Optional[GraphNode]:
        """Localização tolerante: id numérico -> título exato -> caminho -> substring."""
        query = query.strip()
        if query.isdigit():
            node = await self.get_node_by_id(cur, int(query))
            if node:
                return node

        path = query.replace("\\", "/")
        path_md = path if path.lower().endswith(".md") else path + ".md"

        # 1. Título exato
        await cur.execute(
            "SELECT id, title, file_path, stem FROM notes WHERE title ILIKE %s LIMIT 1;",
            (query,),
        )
        row = await cur.fetchone()

        # 2. Caminho do arquivo ou stem
        if not row:
            await cur.execute(
                """SELECT id, title, file_path, stem FROM notes
                   WHERE file_path ILIKE %s OR file_path ILIKE %s OR stem ILIKE %s
                   ORDER BY length(file_path) LIMIT 1;""",
                (path_md, "%/" + path_md, query),
            )
            row = await cur.fetchone()

        # 3. Substring no título
        if not row:
            like = "%" + query.replace("%", r"\%").replace("_", r"\_") + "%"
            await cur.execute(
                """SELECT id, title, file_path, stem FROM notes
                   WHERE title ILIKE %s OR file_path ILIKE %s
                   ORDER BY length(title) LIMIT 1;""",
                (like, like),
            )
            row = await cur.fetchone()

        if not row:
            return None

        return GraphNode(
            id=row[0],
            title=mask_pii(row[1]),
            file_path=row[2],
            folder=_folder_from_path(row[2]),
            stem=row[3],
        )

    async def traverse(
        self,
        root_key: str,
        hops: int = 2,
        direction: str = "both",
        limit_per_hop: int = 15,
        exclude_folders: Optional[List[str]] = None,
    ) -> Optional[TraversalResult]:
        """Executa traversal de 1 ou 2 saltos a partir de uma nota raiz."""
        hops = max(1, min(hops, 2))
        direction = direction.lower() if direction in ("out", "in", "both") else "both"
        exclude_folders = exclude_folders or ["Excalidraw/"]

        async with self.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                root = await self.resolve_note(cur, root_key)
                if not root:
                    return None

                # 1. Hop 1: conexões diretas
                hop1_out_dict: Dict[int, GraphNode] = {}
                hop1_in_dict: Dict[int, GraphNode] = {}
                unresolved: Set[str] = set()

                if direction in ("out", "both"):
                    # Outlinks: links que saem de root
                    await cur.execute(
                        """
                        SELECT DISTINCT ON (nl.target_stem)
                            t.id, t.title, t.file_path, t.stem, nl.target_title
                        FROM note_links nl
                        LEFT JOIN notes t ON t.stem = nl.target_stem
                        WHERE nl.source_note_id = %s AND nl.target_stem <> ''
                        ORDER BY nl.target_stem, length(coalesce(t.file_path, ''));
                        """,
                        (root.id,),
                    )
                    for r in await cur.fetchall():
                        target_id, target_title, target_file, target_stem, raw_title = r
                        if target_id is not None:
                            folder = _folder_from_path(target_file)
                            if not any(target_file.startswith(ex) for ex in exclude_folders):
                                hop1_out_dict[target_id] = GraphNode(
                                    id=target_id,
                                    title=mask_pii(target_title),
                                    file_path=target_file,
                                    folder=folder,
                                    stem=target_stem,
                                )
                        else:
                            unresolved.add(raw_title or target_stem)

                if direction in ("in", "both"):
                    # Backlinks: notas que apontam para root
                    await cur.execute(
                        """
                        SELECT DISTINCT s.id, s.title, s.file_path, s.stem
                        FROM note_links nl
                        JOIN notes s ON s.id = nl.source_note_id
                        WHERE nl.target_stem = %s AND s.id <> %s;
                        """,
                        (root.stem, root.id),
                    )
                    for r in await cur.fetchall():
                        s_id, s_title, s_file, s_stem = r
                        if not any(s_file.startswith(ex) for ex in exclude_folders):
                            hop1_in_dict[s_id] = GraphNode(
                                id=s_id,
                                title=mask_pii(s_title),
                                file_path=s_file,
                                folder=_folder_from_path(s_file),
                                stem=s_stem,
                            )

                hop1_all_ids = set(hop1_out_dict.keys()) | set(hop1_in_dict.keys())

                # Se hops == 1, finaliza aqui
                if hops == 1 or not hop1_all_ids:
                    return TraversalResult(
                        root=root,
                        hops=1,
                        direction=direction,
                        hop1_outlinks=list(hop1_out_dict.values())[:limit_per_hop],
                        hop1_backlinks=list(hop1_in_dict.values())[:limit_per_hop],
                        hop2_nodes=[],
                        bridges=[],
                        unresolved_links=sorted(list(unresolved))[:limit_per_hop],
                        total_nodes_visited=1 + len(hop1_all_ids),
                    )

                # 2. Hop 2: conexões dos vizinhos de Hop 1
                hop1_id_list = list(hop1_all_ids)
                hop2_dict: Dict[int, GraphNode] = {}
                bridges_map: Dict[int, List[str]] = {}  # hop2_id -> [stems dos vizinhos hop1 que conectam a ele]

                # Coleta conexões de saída de Hop 1 para novos nós
                if direction in ("out", "both"):
                    await cur.execute(
                        """
                        SELECT DISTINCT ON (nl.source_note_id, nl.target_stem)
                            nl.source_note_id, t.id, t.title, t.file_path, t.stem
                        FROM note_links nl
                        JOIN notes t ON t.stem = nl.target_stem
                        WHERE nl.source_note_id = ANY(%s)
                          AND t.id <> %s
                          AND t.id <> ALL(%s);
                        """,
                        (hop1_id_list, root.id, hop1_id_list),
                    )
                    for r in await cur.fetchall():
                        src_id, t_id, t_title, t_file, t_stem = r
                        if any(t_file.startswith(ex) for ex in exclude_folders):
                            continue
                        if t_id not in hop2_dict:
                            hop2_dict[t_id] = GraphNode(
                                id=t_id,
                                title=mask_pii(t_title),
                                file_path=t_file,
                                folder=_folder_from_path(t_file),
                                stem=t_stem,
                            )
                        src_node = hop1_out_dict.get(src_id) or hop1_in_dict.get(src_id)
                        if src_node:
                            bridges_map.setdefault(t_id, []).append(src_node.stem)

                # Coleta conexões de entrada de Hop 1 (quem aponta para nós de Hop 1)
                if direction in ("in", "both"):
                    hop1_stems = [
                        (hop1_out_dict.get(nid) or hop1_in_dict.get(nid)).stem
                        for nid in hop1_id_list
                        if (hop1_out_dict.get(nid) or hop1_in_dict.get(nid))
                    ]
                    await cur.execute(
                        """
                        SELECT DISTINCT s.id, s.title, s.file_path, s.stem, nl.target_stem
                        FROM note_links nl
                        JOIN notes s ON s.id = nl.source_note_id
                        WHERE nl.target_stem = ANY(%s)
                          AND s.id <> %s
                          AND s.id <> ALL(%s);
                        """,
                        (hop1_stems, root.id, hop1_id_list),
                    )
                    for r in await cur.fetchall():
                        s_id, s_title, s_file, s_stem, tgt_stem = r
                        if any(s_file.startswith(ex) for ex in exclude_folders):
                            continue
                        if s_id not in hop2_dict:
                            hop2_dict[s_id] = GraphNode(
                                id=s_id,
                                title=mask_pii(s_title),
                                file_path=s_file,
                                folder=_folder_from_path(s_file),
                                stem=s_stem,
                            )
                        bridges_map.setdefault(s_id, []).append(tgt_stem)

                # Identifica nós ponte: nós de Hop 2 conectados a 2+ nós de Hop 1
                bridges_list: List[Tuple[GraphNode, List[str]]] = []
                for h2_id, connecting_stems in bridges_map.items():
                    unique_stems = sorted(list(set(connecting_stems)))
                    if len(unique_stems) >= 2 and h2_id in hop2_dict:
                        bridges_list.append((hop2_dict[h2_id], unique_stems))

                # Ordena pontes pelas mais conectadas primeiro
                bridges_list.sort(key=lambda item: len(item[1]), reverse=True)

                return TraversalResult(
                    root=root,
                    hops=2,
                    direction=direction,
                    hop1_outlinks=list(hop1_out_dict.values())[:limit_per_hop],
                    hop1_backlinks=list(hop1_in_dict.values())[:limit_per_hop],
                    hop2_nodes=list(hop2_dict.values())[:limit_per_hop],
                    bridges=bridges_list[:limit_per_hop],
                    unresolved_links=sorted(list(unresolved))[:limit_per_hop],
                    total_nodes_visited=1 + len(hop1_all_ids) + len(hop2_dict),
                )

    async def find_shortest_path(
        self,
        source_key: str,
        target_key: str,
        max_depth: int = 4,
        direction: str = "both",
    ) -> Optional[PathResult]:
        """Encontra o menor caminho entre duas notas usando busca recursiva em largura (BFS)."""
        max_depth = max(1, min(max_depth, 5))
        direction = direction.lower() if direction in ("out", "in", "both") else "both"

        async with self.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                src_node = await self.resolve_note(cur, source_key)
                tgt_node = await self.resolve_note(cur, target_key)

                if not src_node or not tgt_node:
                    return None

                if src_node.id == tgt_node.id:
                    return PathResult(
                        source=src_node,
                        target=tgt_node,
                        length=0,
                        steps=[],
                        path_nodes=[src_node],
                    )

                # Construção da query de arestas com base na direção
                if direction == "out":
                    edge_filter = "SELECT nl.source_note_id AS u, tn.id AS v, '->' AS dir FROM note_links nl JOIN notes tn ON tn.stem = nl.target_stem WHERE tn.id <> nl.source_note_id"
                elif direction == "in":
                    edge_filter = "SELECT tn.id AS u, nl.source_note_id AS v, '<-' AS dir FROM note_links nl JOIN notes tn ON tn.stem = nl.target_stem WHERE tn.id <> nl.source_note_id"
                else:
                    edge_filter = """
                    SELECT nl.source_note_id AS u, tn.id AS v, '->' AS dir
                    FROM note_links nl JOIN notes tn ON tn.stem = nl.target_stem WHERE tn.id <> nl.source_note_id
                    UNION
                    SELECT tn.id AS u, nl.source_note_id AS v, '<-' AS dir
                    FROM note_links nl JOIN notes tn ON tn.stem = nl.target_stem WHERE tn.id <> nl.source_note_id
                    """

                sql = f"""
                WITH RECURSIVE edges AS (
                    {edge_filter}
                ),
                bfs AS (
                    SELECT u, v, ARRAY[u, v] AS path, ARRAY[dir] AS dirs, 1 AS depth
                    FROM edges
                    WHERE u = %s
                    UNION ALL
                    SELECT b.u, e.v, b.path || e.v, b.dirs || e.dir, b.depth + 1
                    FROM bfs b
                    JOIN edges e ON e.u = b.v
                    WHERE NOT (e.v = ANY(b.path)) AND b.depth < %s
                )
                SELECT b.depth, b.path, b.dirs
                FROM bfs b
                WHERE b.v = %s
                ORDER BY b.depth ASC
                LIMIT 1;
                """
                await cur.execute(sql, (src_node.id, max_depth, tgt_node.id))
                row = await cur.fetchone()
                if not row:
                    return None

                depth, path_ids, dirs = row

                # Busca metadados de todos os nós do caminho
                await cur.execute(
                    "SELECT id, title, file_path, stem FROM notes WHERE id = ANY(%s);",
                    (path_ids,),
                )
                nodes_by_id = {
                    r[0]: GraphNode(
                        id=r[0],
                        title=mask_pii(r[1]),
                        file_path=r[2],
                        folder=_folder_from_path(r[2]),
                        stem=r[3],
                    )
                    for r in await cur.fetchall()
                }

                path_nodes = [nodes_by_id[nid] for nid in path_ids if nid in nodes_by_id]
                steps = []
                for i in range(len(path_nodes) - 1):
                    dir_sym = dirs[i] if i < len(dirs) else "->"
                    steps.append((path_nodes[i], dir_sym, path_nodes[i + 1]))

                return PathResult(
                    source=src_node,
                    target=tgt_node,
                    length=depth,
                    steps=steps,
                    path_nodes=path_nodes,
                )
