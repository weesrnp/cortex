"""Mede a qualidade da busca no vault real: acerto no 1º lugar, no top-5 e MRR, para cada modo do FTS.
Rodar antes/depois de mexer em embedding, RRF, chunking ou parser (não é teste do pytest):
    docker compose exec brain-api python tests/avaliar_busca.py            # compara and, or e idf
    docker compose exec brain-api python tests/avaliar_busca.py and        # só um modo
    docker compose exec brain-api python tests/avaliar_busca.py and --conjunto conferencia   # outro conjunto
Cada pergunta gera o embedding uma vez só e roda em todos os modos (comparação justa, sem a GPU no tempo).
As perguntas ficam em perfis/<PERFIL>/perguntas.yaml (`principais`); este script só mede.
Perguntas parafraseadas de propósito (sem repetir o título); nota renomeada no vault = atualizar o alvo lá (o script
para antes de medir se um alvo sumiu do índice ou tem o nome repetido em duas pastas). Perguntas novas: sortear_notas.py.
Pergunta ruim ≠ busca ruim: se uma nota na frente do alvo TAMBÉM responde a pergunta, corrija a pergunta (dê o contexto
que falta, sem copiar o título) ou aceite as duas notas (alvo = lista; a 1ª é a que o diagnosticar_busca.py examina).
"±" = erro padrão do MRR: diferença menor que isso entre duas medições pode ser sorte; olhe também o placar por pergunta."""
import asyncio
import statistics
import sys
import time
from typing import get_args

from src.config import FtsMode, perfil, settings
from src.database import db
from src.embeddings import OllamaEmbeddingClient
from src.search import HybridSearchEngine, note_stem

# Perguntas do perfil da área (perfis/<perfil>/perguntas.yaml): [pergunta, nome do arquivo da nota que responde]
ORIGINAIS = perfil.numero("originais")  # as primeiras N: comparáveis com medições antigas
PERGUNTAS = perfil.perguntas("principais")


def aceitas(alvo) -> tuple:
    """Notas que valem como acerto: o nome do arquivo, ou a tupla quando duas notas respondem igualmente."""
    return alvo if isinstance(alvo, tuple) else (alvo,)


def posicao(stems: list, alvo):
    """Posição (1-5) da primeira nota aceita no top-5, ou None."""
    ok = aceitas(alvo)
    return next((i for i, s in enumerate(stems, 1) if s in ok), None)


def rotulo(alvo) -> str:
    principal, *outras = aceitas(alvo)
    return principal + (f" (ou {', '.join(outras)})" if outras else "")


def rr(pos) -> float:
    """Posição recíproca: 1º = 1, 2º = 0,5 ... fora do top-5 = 0. A média é o MRR."""
    return 1 / pos if pos else 0.0


def erro_padrao(posicoes: list) -> float:
    return statistics.stdev(map(rr, posicoes)) / len(posicoes) ** 0.5 if len(posicoes) > 1 else 0.0


def placar(novo: list, base: list) -> str:
    melhor = sum(rr(a) > rr(b) for a, b in zip(novo, base))
    pior = sum(rr(a) < rr(b) for a, b in zip(novo, base))
    return f"melhor em {melhor}, pior em {pior} de {len(base)} perguntas"


async def validar_alvos(conn, perguntas: list = PERGUNTAS) -> None:
    """Alvo renomeado/apagado conta como erro da busca; nome repetido em duas pastas conta acerto na nota errada."""
    cur = await conn.execute(
        "SELECT stem, count(*) FROM notes WHERE stem = ANY(%s) AND file_path NOT LIKE 'Excalidraw/%%' GROUP BY stem;",
        ([a for _, alvo in perguntas for a in aceitas(alvo)],))
    achados = dict(await cur.fetchall())
    faltam = [a for _, alvo in perguntas for a in aceitas(alvo) if a not in achados]
    repetidos = [alvo for alvo, n in achados.items() if n > 1]
    if faltam or repetidos:
        sys.exit(f"alvos inválidos no perguntas.yaml do perfil (nada foi medido): fora do índice {faltam or '-'}; "
                 f"nome repetido em mais de uma pasta {repetidos or '-'}")


async def main(modos, conjunto: str = "principais"):
    global PERGUNTAS
    PERGUNTAS = perfil.perguntas(conjunto)
    if not PERGUNTAS:
        sys.exit(f"perguntas.yaml do perfil sem o conjunto '{conjunto}'")
    await db.open()
    async with db.pool.connection() as conn:
        await validar_alvos(conn, PERGUNTAS)
    client = OllamaEmbeddingClient(settings.ollama_base_url, settings.embedding_model)
    engine = HybridSearchEngine(db.pool, client)
    await client.get_query_embedding("aquecer")  # carga fria do modelo fora da medição
    vetores = [await client.get_query_embedding(p) for p, _ in PERGUNTAS]
    n = len(PERGUNTAS)
    posicoes = {}
    # trio de medidas do projeto ai-memory: acerto, tempo (média e p95) e contexto que o agente lê,
    # em tokens estimados por caracteres/4 sobre título + conteúdo dos 5 trechos devolvidos
    print(f"{'modo':<5} {'hit@1':>6} {'hit@5':>6} {'MRR':>6} {'±':>5}  {'FTS no top-5':>12}  {'alvo via FTS':>12}  "
          f"{'ms/consulta':>11} {'p95':>5} {'tokens':>6}")
    for modo in modos:
        hit1 = hit5 = com_fts = alvo_fts = 0
        mrr = 0.0
        posicoes[modo] = []
        tempos, tokens = [], []
        for (pergunta, alvo), vetor in zip(PERGUNTAS, vetores):
            t0 = time.perf_counter()
            res = await engine.search(pergunta, limit=5, expand_graph=False, query_vector=vetor, fts_mode=modo)
            tempos.append((time.perf_counter() - t0) * 1000)
            tokens.append(sum(len(r.title) + len(r.content) for r in res) / 4)
            stems = [note_stem(r.file_path) for r in res]
            com_fts += any(r.text_rank for r in res)  # o lado léxico chegou a pôr algum trecho no top-5?
            pos = posicao(stems, alvo)
            posicoes[modo].append(pos)
            if pos:
                hit5 += 1
                hit1 += pos == 1
                mrr += 1 / pos
                alvo_fts += any(r.text_rank for r in res if note_stem(r.file_path) in aceitas(alvo))
        p95 = statistics.quantiles(tempos, n=20)[-1] if len(tempos) > 1 else tempos[0]
        print(f"{modo:<5} {hit1:>4}/{n} {hit5:>4}/{n} {mrr / n:>6.3f} {erro_padrao(posicoes[modo]):>5.3f}  "
              f"{com_fts:>9}/{n}  {alvo_fts:>9}/{n}  {statistics.mean(tempos):>11.0f} {p95:>5.0f} "
              f"{statistics.mean(tokens):>6.0f}")
    for modo in modos[1:]:
        print(f"{modo} x {modos[0]}: {placar(posicoes[modo], posicoes[modos[0]])}")

    print("\nposição do alvo no top-5 (— = fora):")
    print("  ".join(f"{m:>4}" for m in modos) + "  alvo")
    for i, (_, alvo) in enumerate(PERGUNTAS):
        mudou = "   <-- mudou" if len({posicoes[m][i] for m in modos}) > 1 else ""
        print("  ".join(f"{posicoes[m][i] or '—':>4}" for m in modos) + f"  {rotulo(alvo)}{mudou}")
    await db.close()
    await client.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    conjunto = "principais"
    if "--conjunto" in args:  # --conjunto conferencia | pdfs (perguntas.yaml do perfil)
        i = args.index("--conjunto")
        conjunto = args[i + 1]
        del args[i:i + 2]
    modos = args or list(get_args(FtsMode))
    invalidos = [m for m in modos if m not in get_args(FtsMode)]
    if invalidos:
        sys.exit(f"modo desconhecido: {', '.join(invalidos)} (use {', '.join(get_args(FtsMode))})")
    asyncio.run(main(modos, conjunto))
