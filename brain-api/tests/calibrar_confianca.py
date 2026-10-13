"""Calibra o limite de "baixa confiança" da busca: o sinal é o MAIOR cosseno entre o vetor da pergunta e todos os
trechos do índice (fora Excalidraw/). Compara as perguntas que o vault responde (`principais` do perguntas.yaml) com
perguntas que o vault NÃO responde (`fora` do perguntas.yaml; sem elas, uma lista genérica) e varre o limite.
Não é teste do pytest; só leitura:
    docker compose exec brain-api python tests/calibrar_confianca.py
Critério: falso alarme (pergunta respondível marcada como "o vault não tem") é o pior erro, então o limite recomendado é
o maior sem nenhum falso alarme, com folga de 0,02 abaixo do menor score DENTRO. O resultado vai em `limiares` do
perfil.yaml. Confira antes que nenhuma pergunta FORA tenha resposta no vault (é fácil errar)."""
import asyncio
import math
import statistics

from avaliar_busca import PERGUNTAS  # sys.path[0] = /app/tests quando roda "python tests/calibrar_confianca.py"
from src.config import perfil, settings
from src.database import db
from src.embeddings import OllamaEmbeddingClient

FOLGA = 0.02

GENERICAS = [
    "receita de feijoada completa para 10 pessoas",
    "como fazer lasanha à bolonhesa no forno",
    "quem ganhou a Copa do Mundo de futebol de 1970 e quem era o técnico",
    "escalação do time na final do campeonato continental",
    "como programar a física de pulo de um personagem numa engine de jogos",
    "o que é um buraco negro e como se forma",
    "quando será o próximo eclipse solar total visível no Brasil",
    "como criar um cluster Kubernetes com Helm charts",
    "diferença entre useEffect e useMemo no React",
    "como formatar pendrive em exFAT no macOS",
]
FORA = perfil.lista("fora") or GENERICAS

SQL = ("SELECT 1 - min(c.embedding <=> %s::vector) FROM chunks c JOIN notes n ON n.id = c.note_id "
       "WHERE n.file_path NOT LIKE 'Excalidraw/%%'")


def resumo(nome: str, scores: list) -> str:
    q = statistics.quantiles(scores, n=10, method="inclusive")
    return (f"{nome:<7} n={len(scores):>2}  mín {min(scores):.3f}  p10 {q[0]:.3f}  mediana {statistics.median(scores):.3f}"
            f"  p90 {q[-1]:.3f}  máx {max(scores):.3f}")


async def main():
    await db.open()
    client = OllamaEmbeddingClient(settings.ollama_base_url, settings.embedding_model)
    await client.get_query_embedding("aquecer")

    async def score(pergunta: str) -> float:
        vetor = await client.get_query_embedding(pergunta)
        async with db.pool.connection() as conn:
            cur = await conn.execute(SQL, (str(vetor),))
            return float((await cur.fetchone())[0])

    dentro = [(await score(p), p) for p, _ in PERGUNTAS]
    fora = [(await score(p), p) for p in FORA]
    sd, sf = [s for s, _ in dentro], [s for s, _ in fora]

    print(resumo("DENTRO", sd))
    print(resumo("FORA", sf))
    print("\nDENTRO, do menor para o maior:")
    for s, p in sorted(dentro):
        print(f"  {s:.3f}  {p}")
    print("\nFORA, do maior para o menor:")
    for s, p in sorted(fora, reverse=True):
        print(f"  {s:.3f}  {p}")

    print(f"\n{'limite':>6}  {'falso alarme (DENTRO < limite)':>30}  {'FORA pegas':>10}")
    for i in range(40, 81):
        lim = i / 100
        print(f"{lim:>6.2f}  {sum(s < lim for s in sd):>27}/{len(sd)}  {sum(s < lim for s in sf):>7}/{len(sf)}")

    lim = math.floor((min(sd) - FOLGA) * 100) / 100  # arredonda para baixo: a folga nunca encolhe
    pegas = sum(s < lim for s in sf)
    print(f"\nrecomendado: {lim:.2f} (menor DENTRO {min(sd):.3f} - folga {FOLGA}); pega {pegas}/{len(sf)} FORA, "
          f"0/{len(sd)} falso alarme")
    print("FORA que escapam (score >= limite):")
    for s, p in sorted(fora, reverse=True):
        if s >= lim:
            print(f"  {s:.3f}  {p}")
    await db.close()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
