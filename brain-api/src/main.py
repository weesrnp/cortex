import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from watchdog.observers.polling import PollingObserver

from . import __version__
from .config import perfil, settings
from .database import db
from .embeddings import OllamaEmbeddingClient
from .indexer import VaultIndexer, VaultEventHandler
from .api import router as api_router
from .explorer import router as explorer_router, STATIC_DIR
from .mcp_server import mcp

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cortex.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Boot Database Pool
    await db.open()

    # 2. Check Ollama & Embedding Model
    embed_client = OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)
    await embed_client.ensure_model_available()

    # 3. Vault Indexer Startup & Full Scan
    indexer = VaultIndexer(settings.vault_path, db.pool, embed_client)
    scan_task = asyncio.create_task(indexer.scan_full_vault())

    # 4. Sync em tempo real. Polling porque inotify não atravessa o bind mount Windows→container.
    # varre o cofre a cada 5 s (um stat por arquivo); com cofre muito grande, aumente o timeout.
    observer = PollingObserver(timeout=5)
    observer.schedule(VaultEventHandler(indexer, asyncio.get_running_loop()), settings.vault_path, recursive=True)
    observer.start()

    logger.info("%s inicializado (perfil %r) e pronto para requisições.", perfil.titulo, perfil.nome)
    async with mcp.session_manager.run():
        yield

    # Graceful Shutdown
    observer.stop()
    observer.join()
    scan_task.cancel()
    await db.close()
    await embed_client.close()
    logger.info("%s finalizado.", perfil.titulo)

app = FastAPI(title=f"{perfil.titulo} API", version=__version__, lifespan=lifespan)
app.include_router(api_router)
app.include_router(explorer_router)
app.mount("/explorer/static", StaticFiles(directory=STATIC_DIR), name="explorer-static")
app.mount("/", mcp.streamable_http_app())  # MCP em /mcp (streamable HTTP); rotas REST têm precedência

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=settings.api_host, port=settings.api_port, reload=False)
