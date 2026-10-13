import asyncio
import json

import httpx

from src.config import settings
from src.embeddings import BATCH_SIZE, QUERY_INSTRUCTION, OllamaEmbeddingClient

DIM = settings.embedding_dim


def _client(calls: list, dims: int) -> OllamaEmbeddingClient:
    """Cliente com Ollama falso: registra cada POST /api/embed e devolve vetores de `dims`."""
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1] * dims for _ in body["input"]]})

    c = OllamaEmbeddingClient(base_url="http://ollama-falso")
    c.client = httpx.AsyncClient(base_url=c.base_url, transport=httpx.MockTransport(handler))
    return c


def test_documents_batched_truncated_and_cut_to_dim():
    calls = []
    c = _client(calls, dims=2560)  # dimensão nativa do qwen3-embedding:4b
    out = asyncio.run(c.get_document_embeddings(["x"] * (BATCH_SIZE + 3)))

    assert [len(b["input"]) for b in calls] == [BATCH_SIZE, 3]
    assert all(b["truncate"] is True for b in calls)  # sem isso chunk longo volta 500
    assert len(out) == BATCH_SIZE + 3 and all(len(v) == DIM for v in out)
    assert calls[0]["input"][0] == "x"  # documento sem prefixo


def test_query_gets_instruction_and_small_model_is_rejected():
    calls = []
    asyncio.run(_client(calls, dims=DIM).get_query_embedding("prazo do relatório"))
    assert calls[0]["input"] == [f"{QUERY_INSTRUCTION}prazo do relatório"]

    try:
        asyncio.run(_client([], dims=DIM - 1).get_query_embedding("x"))
        raise AssertionError("modelo com menos dimensões que o banco deveria falhar")
    except ValueError:
        pass
