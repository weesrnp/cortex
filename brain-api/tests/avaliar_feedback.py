"""Avaliação com perguntas reais: cada feedback "útil" (brain_feedback) vira um par pergunta → nota certa.
Roda a busca de hoje em cada pergunta do search_log que tem nota marcada útil e mostra a posição dessa nota no top-5
(hit@1, hit@5, MRR) e as que caíram fora. Complementa o avaliar_busca.py (perguntas escritas à mão) com o que os
agentes perguntam de fato. Só lê (nada é gravado; esta busca não entra no search_log).
    docker compose exec brain-api python tests/avaliar_feedback.py [dias, padrão 90]
Não é teste do pytest."""
import asyncio
import sys

from avaliar_busca import erro_padrao, rr
from src.config import settings
from src.database import db
from src.embeddings import OllamaEmbeddingClient
from src.search import HybridSearchEngine, mask_pii

SQL = """
SELECT DISTINCT ON (l.query, f.file_path) l.query, f.file_path, l.filtros
FROM search_feedback f JOIN search_log l ON l.id = f.search_id
WHERE f.util AND f.file_path IS NOT NULL AND f.created_at > now() - make_interval(days => %s)
ORDER BY l.query, f.file_path, f.created_at DESC;
"""


async def main(dias: int) -> None:
    await db.open()
    async with db.pool.connection() as conn:
        cur = await conn.execute(SQL, (dias,))
        pares = await cur.fetchall()
    if not pares:
        print(f"Nenhum feedback útil nos últimos {dias} dias (os agentes ainda não usaram o brain_feedback).")
        await db.close()
        return
    client = OllamaEmbeddingClient(settings.ollama_base_url, settings.embedding_model)
    engine = HybridSearchEngine(db.pool, client)
    posicoes = []
    for query, alvo, filtros in pares:
        res = await engine.search(query, limit=5, expand_graph=False,
                                  include_folders=(filtros or {}).get("include"),
                                  exclude_folders=(filtros or {}).get("exclude"))
        caminhos = [r.file_path for r in res]
        posicoes.append(caminhos.index(alvo) + 1 if alvo in caminhos else None)
    n = len(pares)
    print(f"{n} pares pergunta real → nota útil (últimos {dias} dias)")
    print(f"hit@1 {sum(p == 1 for p in posicoes)}/{n}  hit@5 {sum(p is not None for p in posicoes)}/{n}  "
          f"MRR {sum(map(rr, posicoes)) / n:.3f} ± {erro_padrao(posicoes):.3f}")
    fora = [(q, a) for (q, a, _), p in zip(pares, posicoes) if p is None]
    if fora:
        print("\nnota útil fora do top-5 (candidatas a pergunta nova no avaliar_busca.py):")
        for q, a in fora:
            print(f"  \"{mask_pii(q)}\" → {a}")
    await db.close()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if sys.argv[1:] else 90))
