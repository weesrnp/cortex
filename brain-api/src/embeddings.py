import asyncio
import logging
import httpx
from typing import List

from .config import perfil, settings
from .siglas import expandir_siglas

logger = logging.getLogger("cortex.embeddings")

# Instrução do Qwen3 na pergunta (em inglês: recomendação para multilíngue); o texto vem do perfil da área.
QUERY_INSTRUCTION = f"Instruct: {perfil.instrucao_embedding}\nQuery:"
BATCH_SIZE = 8

class OllamaEmbeddingClient:
    def __init__(self, base_url: str = "http://ollama:11434", model: str = "qwen3-embedding:4b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = settings.embedding_dim
        self.semaphore = asyncio.Semaphore(4)  # Limita concorrência na GPU
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(180.0, connect=10.0),  # lote de 8 chunks no 4b + carga fria do modelo
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )

    async def ensure_model_available(self) -> None:
        """Verifica se o modelo está carregado. Se não estiver, realiza o pull."""
        try:
            resp = await self.client.get("/api/tags")
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            # Aceita variações como nomic-embed-text:latest
            if not any(self.model in m for m in models):
                logger.info(f"Modelo {self.model} não encontrado. Iniciando pull no Ollama...")
                pull_resp = await self.client.post("/api/pull", json={"model": self.model, "stream": False}, timeout=600.0)
                pull_resp.raise_for_status()
                logger.info(f"Modelo {self.model} baixado com sucesso.")
            else:
                logger.info(f"Modelo {self.model} verificado e pronto.")
        except Exception as e:
            logger.error(f"Erro ao verificar/baixar modelo no Ollama: {e}")
            raise

    async def get_document_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embeddings de chunks em lotes (documentos no Qwen3 não levam prefixo)."""
        out: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            out += await self._embed(texts[i:i + BATCH_SIZE])
        return out

    async def get_query_embedding(self, text: str) -> List[float]:
        """Embedding da pergunta, com a instrução de tarefa que o Qwen3 espera (e as siglas expandidas: siglas.py)."""
        if settings.expand_siglas:
            text = expandir_siglas(text)
        return (await self._embed([f"{QUERY_INSTRUCTION}{text}"]))[0]

    async def _embed(self, inputs: List[str]) -> List[List[float]]:
        async with self.semaphore:
            for attempt in range(3):
                try:
                    resp = await self.client.post("/api/embed", json={
                        "model": self.model,
                        "input": inputs,
                        "truncate": True,  # chunk maior que o contexto é cortado, não gera 500
                        "options": {"num_ctx": settings.embedding_num_ctx},
                    })
                    resp.raise_for_status()
                    embs = resp.json()["embeddings"]
                    if len(embs[0]) < self.dim:
                        raise ValueError(f"Modelo {self.model} gera {len(embs[0])} dims; banco espera {self.dim}.")
                    # Qwen3 é Matryoshka: o prefixo de N dims preserva a semântica; cosseno dispensa renormalizar.
                    return [e[:self.dim] for e in embs]
                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1.0 * (attempt + 1))

    async def close(self):
        await self.client.aclose()
