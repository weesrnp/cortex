import asyncio
import hashlib
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set
from psycopg.types.json import Jsonb
from watchdog.events import FileSystemEventHandler

from .parser import VaultParser, ParsedNote, hash_ficha
from .embeddings import OllamaEmbeddingClient
from .config import settings

logger = logging.getLogger("cortex.indexer")
logging.getLogger("pypdf").setLevel(logging.ERROR)  # avisos de fonte (fontTools) a cada página não ajudam
PDFS_PENDENTES = "<pdfs>"  # chave única de debounce para eventos de PDF (schedule_sync)


@dataclass
class ListaPdfs:
    notas: List[ParsedNote] = field(default_factory=list)  # entram no índice
    com_erro: Set[str] = field(default_factory=set)         # leitura falhou (OneDrive baixando, gravação): não apagar
    achados: int = 0                                         # PDFs vistos no vault (0 = montagem falhou: não apagar)

class VaultIndexer:
    def __init__(self, vault_path: str, db_pool, embed_client: OllamaEmbeddingClient):
        self.vault_path = Path(vault_path)
        self.db_pool = db_pool
        self.embed_client = embed_client
        self.pending_tasks: Dict[str, asyncio.TimerHandle] = {}
        self.debounce_seconds = 1.5
        self.trava_pdfs = asyncio.Lock()  # scan e evento de PDF não leem/vetorizam os PDFs ao mesmo tempo
        self.pdfs_lidos: Dict[str, tuple] = {}  # caminho -> ((mtime_ns, tamanho), ParsedNote | None): não reler

    async def scan_full_vault(self) -> None:
        """Scan inicial completo do vault: insere novas e atualiza alteradas."""
        logger.info(f"Iniciando varredura completa do vault em: {self.vault_path}")
        md_files = [f for f in self.vault_path.rglob("*.md") if not self._is_ignored(f)]
        logger.info(f"Total de {len(md_files)} arquivos .md localizados.")

        # Busca hashes existentes no banco
        async with self.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT file_path, file_hash, ficha_hash FROM notes;")
                linhas = await cur.fetchall()
                existing_hashes = {row[0]: row[1] for row in linhas}
                fichas_no_banco = {row[0]: row[2] for row in linhas}

        # Nota no banco sem arquivo (apagada com a API parada, ou em pasta que passou a ser ignorada) sai do índice.
        # Vault vazio = montagem falhou: não apaga nada.
        # PDFs ficam para sincronizar_pdfs, depois dos .md (a escolha da cópia ligada usa os links já indexados).
        no_vault = {str(f.relative_to(self.vault_path)).replace("\\", "/") for f in md_files}
        orfas = [p for p in existing_hashes if p not in no_vault and not p.lower().endswith(".pdf")]
        if md_files and orfas:
            async with self.db_pool.connection() as conn:
                await conn.execute("DELETE FROM notes WHERE file_path = ANY(%s);", (orfas,))
            logger.info(f"Removidas do índice {len(orfas)} notas sem arquivo no vault.")

        synced = 0
        skipped = 0
        fichas_pendentes = []  # nota inalterada sem ficha (ou com o texto da ficha mudado): só a ficha vai para a GPU
        for file in md_files:
            try:
                rel_path = str(file.relative_to(self.vault_path)).replace("\\", "/")
                parsed = VaultParser.parse_file(file, self.vault_path)

                if rel_path in existing_hashes and existing_hashes[rel_path] == parsed.file_hash:
                    skipped += 1
                    if fichas_no_banco.get(rel_path) != hash_ficha(parsed.ficha()):
                        fichas_pendentes.append(parsed)
                    continue

                await self.index_single_note(parsed)
                synced += 1
                if synced % 25 == 0:
                    logger.info(f"Progresso: {synced} notas indexadas...")
            except Exception as e:
                logger.error(f"Erro ao processar arquivo {file}: {e}")

        if fichas_pendentes:
            await self.atualizar_fichas(fichas_pendentes)
        n_pdfs = await self.sincronizar_pdfs() if settings.index_pdf else 0
        logger.info(f"Varredura concluída. Indexadas: {synced} | Inalteradas: {skipped} | "
                    f"Fichas refeitas: {len(fichas_pendentes)} | PDFs no índice: {n_pdfs}")

    def _pdfs_no_vault(self) -> List[Path]:
        return sorted(f for f in self.vault_path.rglob("*")
                      if f.suffix.lower() == ".pdf" and f.is_file() and not self._is_ignored(f))

    async def pdfs_para_indexar(self, arquivos: List[Path] = None) -> ListaPdfs:
        """PDFs do vault que entram no índice (settings.index_pdf): fora os de caminho que casa com
        (pdf_exclude_pattern, LGPD), os longos demais (pdf_max_pages: manuais de equipamento) e os sem texto. Cópias
        com o mesmo texto e nomes diferentes viram uma nota só: fica a que é alvo de [[link]] em alguma nota (nome
        descritivo, ligado pelos índices), depois o caminho mais curto. PDF com mtime e tamanho iguais aos da passada
        anterior não é relido (ler os ~60 leva 30-55 s; a 1ª passada depois de subir lê todos)."""
        excluir = re.compile(settings.pdf_exclude_pattern) if settings.pdf_exclude_pattern else None
        if arquivos is None:
            arquivos = self._pdfs_no_vault()
        lista = ListaPdfs(achados=len(arquivos))
        lidos, lidos_agora = [], {}
        for f in arquivos:
            rel = str(f.relative_to(self.vault_path)).replace("\\", "/")
            if excluir and excluir.search(rel):  # caminho inteiro: pasta "Pacientes/" com nome neutro também sai
                logger.info(f"PDF fora do índice (caminho com dado nominal): {rel}")
                continue
            try:
                st = f.stat()
                assinatura = (st.st_mtime_ns, st.st_size)
                anterior = self.pdfs_lidos.get(rel)
                if anterior and anterior[0] == assinatura:
                    p = anterior[1]
                else:
                    p = await asyncio.to_thread(VaultParser.parse_pdf, f, self.vault_path, settings.pdf_max_pages)
            except Exception as e:  # passageiro (OneDrive baixando, arquivo em gravação): o que já está no índice fica
                logger.error(f"PDF ilegível {rel}: {e}")
                lista.com_erro.add(rel)
                continue
            lidos_agora[rel] = (assinatura, p)
            if p is None:
                logger.info(f"PDF fora do índice (mais de {settings.pdf_max_pages} páginas ou sem texto): {f.name}")
                continue
            lidos.append(p)
        self.pdfs_lidos = lidos_agora  # PDF que saiu do vault sai da memória também
        async with self.db_pool.connection() as conn:
            cur = await conn.execute("SELECT DISTINCT target_stem FROM note_links;")
            ligados = {r[0] for r in await cur.fetchall()}
        grupos: Dict[str, List[ParsedNote]] = {}
        for p in lidos:
            texto = "\n".join(c.content.split("\n", 1)[-1] for c in p.chunks)  # sem o "[Título (p. N)]"
            grupos.setdefault(hashlib.md5(texto.encode("utf-8")).hexdigest(), []).append(p)
        escolhidos = []
        for copias in grupos.values():
            fica = min(copias, key=lambda p: (p.file_path.rsplit("/", 1)[-1] not in ligados,
                                               len(p.file_path), p.file_path))
            escolhidos.append(fica)
            for c in copias:
                if c is not fica:
                    logger.info(f"PDF fora do índice (cópia de {fica.file_path}): {c.file_path}")
        lista.notas = sorted(escolhidos, key=lambda p: p.file_path)
        return lista

    async def sincronizar_pdfs(self) -> int:
        """Lista de PDFs refeita inteira (cópias e exclusões dependem dos outros PDFs): no fim do scan e a cada
        rajada de eventos de PDF. Sai do índice o PDF que deixou a lista, menos o que deu erro de leitura e nada se o
        vault não mostrou nenhum PDF (montagem falhou). Devolve quantos PDFs ficaram no índice."""
        async with self.trava_pdfs:
            # PDF apagado ou movido para .trash sai já, com commit próprio: a leitura abaixo pode levar 30-55 s e um
            # restart no meio dela deixava o PDF no índice até a próxima subida (2026-10-12).
            arquivos = self._pdfs_no_vault()
            if arquivos:
                no_vault = [str(f.relative_to(self.vault_path)).replace("\\", "/") for f in arquivos]
                async with self.db_pool.connection() as conn:
                    cur = await conn.execute("DELETE FROM notes WHERE file_path ILIKE '%%.pdf' "
                                             "AND NOT (file_path = ANY(%s)) RETURNING file_path;", (no_vault,))
                    if sumiram := await cur.fetchall():
                        logger.info(f"PDFs apagados do vault saíram do índice: {len(sumiram)}")
            lista = await self.pdfs_para_indexar(arquivos)
            caminhos = {p.file_path for p in lista.notas} | lista.com_erro
            async with self.db_pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT file_path, file_hash, ficha_hash FROM notes WHERE file_path ILIKE '%%.pdf';")
                no_banco = {c: (h, f) for c, h, f in await cur.fetchall()}
                sairam = [c for c in no_banco if c not in caminhos]
                if sairam and lista.achados:
                    await conn.execute("DELETE FROM notes WHERE file_path = ANY(%s);", (sairam,))
                    logger.info(f"PDFs removidos do índice: {len(sairam)}")
            fichas = []
            for p in lista.notas:
                hash_banco, ficha_banco = no_banco.get(p.file_path, (None, None))
                if hash_banco != p.file_hash:
                    await self.index_single_note(p)
                    logger.info(f"PDF sincronizado: {p.file_path}")
                elif ficha_banco != hash_ficha(p.ficha()):
                    fichas.append(p)
            if fichas:
                await self.atualizar_fichas(fichas)
            return len(lista.notas)

    async def atualizar_fichas(self, notas: list, lote: int = 32) -> None:
        """Gera só o vetor da ficha de notas cujos trechos não mudaram, em lotes."""
        for i in range(0, len(notas), lote):
            parte = notas[i:i + lote]
            textos = [n.ficha() for n in parte]
            vetores = await self.embed_client.get_document_embeddings(textos)
            async with self.db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "UPDATE notes SET ficha_embedding = %s, ficha_hash = %s WHERE file_path = %s;",
                        [(str(v), hash_ficha(t), n.file_path) for n, t, v in zip(parte, textos, vetores)])
            logger.info(f"Fichas: {min(i + lote, len(notas))}/{len(notas)}")

    async def index_single_note(self, parsed: ParsedNote) -> None:
        """Gera vetores para os chunks e persiste no banco em transação atômica."""
        # 1. Embeddings: trecho com o mesmo texto que a nota já tem no banco reaproveita o vetor (mesmo texto, mesmo
        #    vetor). Editar uma seção ou trocar o parser só manda para a GPU o que mudou.
        conteudos = [ch.content for ch in parsed.chunks]
        ficha = parsed.ficha()
        async with self.db_pool.connection() as conn:
            cur = await conn.execute("""
                SELECT c.content, c.embedding::text
                FROM chunks c JOIN notes n ON n.id = c.note_id
                WHERE n.file_path = %s AND c.content = ANY(%s);
            """, (parsed.file_path, conteudos))
            vetores = dict(await cur.fetchall())
            cur = await conn.execute(
                "SELECT ficha_embedding::text FROM notes WHERE file_path = %s AND ficha_hash = %s;",
                (parsed.file_path, hash_ficha(ficha)))
            ficha_antiga = await cur.fetchone()
        novos = [t for t in dict.fromkeys(conteudos) if t not in vetores]
        if ficha_antiga is None:
            novos_vetores = await self.embed_client.get_document_embeddings(novos + [ficha])
            ficha_vetor = novos_vetores.pop()
        else:  # ficha com o mesmo texto: reaproveita o vetor (como os trechos)
            novos_vetores = await self.embed_client.get_document_embeddings(novos)
            ficha_vetor = ficha_antiga[0]
        vetores.update(zip(novos, novos_vetores))
        chunk_embeddings = [vetores[t] for t in conteudos]

        # 2. Persistência atômica no PostgreSQL
        async with self.db_pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    # Upsert da nota mestre
                    await cur.execute("""
                        INSERT INTO notes (file_path, title, file_hash, ficha_embedding, ficha_hash, updated_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (file_path) DO UPDATE SET
                            title = EXCLUDED.title,
                            file_hash = EXCLUDED.file_hash,
                            ficha_embedding = EXCLUDED.ficha_embedding,
                            ficha_hash = EXCLUDED.ficha_hash,
                            updated_at = NOW()
                        RETURNING id;
                    """, (parsed.file_path, parsed.title, parsed.file_hash, str(ficha_vetor), hash_ficha(ficha)))
                    note_id = (await cur.fetchone())[0]

                    # Remove chunks e links anteriores para recriação limpa
                    await cur.execute("DELETE FROM chunks WHERE note_id = %s;", (note_id,))
                    await cur.execute("DELETE FROM note_tags WHERE note_id = %s;", (note_id,))
                    await cur.execute("DELETE FROM note_links WHERE source_note_id = %s;", (note_id,))

                    # Insere chunks e vetores
                    for ch, emb in zip(parsed.chunks, chunk_embeddings):
                        await cur.execute("""
                            INSERT INTO chunks (note_id, chunk_index, heading_hierarchy, content, token_count, embedding, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """, (note_id, ch.chunk_index, ch.heading_hierarchy, ch.content, ch.token_count, str(emb), Jsonb(ch.metadata)))

                    # Insere tags
                    for tag in parsed.tags:
                        await cur.execute("""
                            INSERT INTO note_tags (note_id, tag) VALUES (%s, %s)
                            ON CONFLICT DO NOTHING;
                        """, (note_id, tag))

                    # Insere wiki-links
                    for link in parsed.links:
                        await cur.execute("""
                            INSERT INTO note_links (source_note_id, target_title) VALUES (%s, %s)
                            ON CONFLICT DO NOTHING;
                        """, (note_id, link))

    async def delete_note(self, rel_path: str) -> None:
        """Deleta a nota e em cascata todos os chunks associados."""
        async with self.db_pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute("DELETE FROM notes WHERE file_path = %s;", (rel_path,))
        logger.info(f"Nota removida da base: {rel_path}")

    def _is_ignored(self, path: Path) -> bool:
        # Igual ao Obsidian: pasta ou arquivo começando com "." não é nota (.obsidian, .trash, e dados de plugin
        # como .smart-env e outras pastas de plugin, que podem virar boa parte do índice).
        return any(p.startswith(".") for p in path.relative_to(self.vault_path).parts)

    def schedule_sync(self, path: str, deleted: bool) -> None:
        """Roda no loop asyncio: agrupa rajadas de eventos do mesmo arquivo (debounce). PDFs dividem uma chave só:
        qualquer PDF refaz a lista inteira, então 10 PDFs copiados juntos = 1 passada, não 10."""
        chave = PDFS_PENDENTES if path.lower().endswith(".pdf") else path
        if handle := self.pending_tasks.pop(chave, None):
            handle.cancel()
        loop = asyncio.get_running_loop()
        espera = self.debounce_seconds * (4 if chave == PDFS_PENDENTES else 1)
        self.pending_tasks[chave] = loop.call_later(
            espera, lambda: asyncio.create_task(self._sync_file(path, deleted, chave))
        )

    async def _sync_file(self, path_str: str, deleted: bool, chave: str = None) -> None:
        self.pending_tasks.pop(chave or path_str, None)
        path = Path(path_str)
        rel_path = str(path.relative_to(self.vault_path)).replace("\\", "/")
        try:
            if path.suffix.lower() == ".pdf":
                if settings.index_pdf:
                    await self.sincronizar_pdfs()
                return
            if deleted or not path.exists():
                await self.delete_note(rel_path)
                return
            parsed = VaultParser.parse_file(path, self.vault_path)
            async with self.db_pool.connection() as conn:
                cur = await conn.execute("SELECT file_hash FROM notes WHERE file_path = %s;", (rel_path,))
                row = await cur.fetchone()
            if row and row[0] == parsed.file_hash:
                return  # só mtime mudou (ex.: OneDrive), conteúdo igual
            await self.index_single_note(parsed)
            logger.info(f"Nota sincronizada: {rel_path}")
        except Exception as e:
            logger.error(f"Erro ao sincronizar {rel_path}: {e}")


class VaultEventHandler(FileSystemEventHandler):
    """Recebe eventos na thread do watchdog e repassa ao loop asyncio do indexador."""

    def __init__(self, indexer: VaultIndexer, loop: asyncio.AbstractEventLoop):
        self.indexer = indexer
        self.loop = loop

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return
        if event.event_type == "moved":
            self._push(event.src_path, True)
            self._push(event.dest_path, False)
        elif event.event_type in ("created", "modified", "deleted"):
            self._push(event.src_path, event.event_type == "deleted")

    def _push(self, path: str, deleted: bool) -> None:
        if path.lower().endswith((".md", ".pdf")) and not self.indexer._is_ignored(Path(path)):
            self.loop.call_soon_threadsafe(self.indexer.schedule_sync, path, deleted)
