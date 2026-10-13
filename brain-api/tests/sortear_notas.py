"""Sorteia notas do vault, de todas as pastas, para escrever perguntas novas no perguntas.yaml do perfil.
Por nota: pasta, nome, quantos trechos, as seções e as 3 notas mais parecidas (centro dos vetores), para a pergunta
apontar para ESTA nota e não para uma irmã. Como o diagnosticar_busca.py, não imprime texto das notas: só nomes e
seções (com mask_pii). Sorteio fixo (md5 do nome): rodar de novo dá a mesma lista enquanto o vault não mudar.
Pastas grandes não engolem as pequenas: cada pasta recebe vagas proporcionais à raiz quadrada do tamanho.
Fica de fora: Excalidraw/, 00_Inbox/, 00_Agent_Decisions/, notas-índice (MAPA, INDICE, Início, README),
nota com menos de 300 caracteres e o que já é alvo no avaliar_busca.py. Só lê (transação READ ONLY).
    docker compose exec brain-api python tests/sortear_notas.py [quantas, padrão 40]
Não é teste do pytest."""
import asyncio
import hashlib
import math
import re
import sys

from avaliar_busca import PERGUNTAS, aceitas
from src.database import db
from src.search import mask_pii, note_stem

FORA = ("Excalidraw/", "00_Inbox/", "00_Agent_Decisions/")
RE_INDICE = re.compile(r"MAPA|^[IÍ]ndice|INDICE|^In[ií]cio$|^README$", re.IGNORECASE)
MAX_SECOES = 12


def pasta(caminho: str) -> str:
    return caminho.split("/", 1)[0] if "/" in caminho else "(raiz)"


def vagas(tamanhos: dict, total: int) -> dict:
    """Vagas por pasta ∝ raiz do tamanho, pelo menos 1 e nunca mais que a pasta tem (maiores restos primeiro)."""
    peso = {p: math.sqrt(n) for p, n in tamanhos.items()}
    soma = sum(peso.values())
    cota = {p: total * w / soma for p, w in peso.items()}
    saida = {p: min(tamanhos[p], max(1, int(c))) for p, c in cota.items()}
    for p in sorted(cota, key=lambda p: cota[p] - int(cota[p]), reverse=True):
        if sum(saida.values()) >= total:
            break
        if saida[p] < tamanhos[p]:
            saida[p] += 1
    return saida


def secoes(hierarquias: list) -> list:
    """Seções da nota sem repetir e sem o nome dela ("Nota > A > B" → "A > B")."""
    vistas = []
    for h in hierarquias:
        partes = [p.strip() for p in (h or "").split(">")]
        resto = " > ".join(partes[1:]) if len(partes) > 1 else ""
        if resto and resto not in vistas:
            vistas.append(resto)
    return vistas


async def main(quantas: int) -> None:
    alvos = {a for _, alvo in PERGUNTAS for a in aceitas(alvo)}
    await db.open()
    async with db.pool.connection() as conn:
        async with conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            cur = await conn.execute("""
                SELECT n.id, n.file_path, count(*), sum(length(c.content))
                FROM notes n JOIN chunks c ON c.note_id = n.id
                GROUP BY n.id, n.file_path;
            """)
            notas = [(i, f, t) for i, f, t, chars in await cur.fetchall()
                     if chars >= 300 and not f.startswith(FORA)
                     and not RE_INDICE.search(note_stem(f)) and note_stem(f) not in alvos]
            por_pasta = {}
            for nota in sorted(notas, key=lambda n: hashlib.md5(note_stem(n[1]).encode()).hexdigest()):
                por_pasta.setdefault(pasta(nota[1]), []).append(nota)
            cotas = vagas({p: len(ns) for p, ns in por_pasta.items()}, quantas)
            sorteadas = [n for p in sorted(por_pasta) for n in por_pasta[p][:cotas[p]]]

            cur = await conn.execute("""
                WITH centro AS MATERIALIZED (
                    SELECT c.note_id, avg(c.embedding) AS v
                    FROM chunks c JOIN notes n ON n.id = c.note_id
                    WHERE n.file_path NOT LIKE 'Excalidraw/%%'
                    GROUP BY c.note_id
                )
                SELECT a.note_id, p.file_path, p.cos
                FROM centro a CROSS JOIN LATERAL (
                    SELECT n.file_path, 1 - (a.v <=> b.v) AS cos
                    FROM centro b JOIN notes n ON n.id = b.note_id
                    WHERE b.note_id <> a.note_id
                    ORDER BY a.v <=> b.v, b.note_id
                    LIMIT 3
                ) p
                WHERE a.note_id = ANY(%s);
            """, ([n[0] for n in sorteadas],))
            parecidas = {}
            for nid, caminho, cos in await cur.fetchall():
                parecidas.setdefault(nid, []).append(f"{mask_pii(note_stem(caminho))} ({cos:.2f})".replace(".", ","))

            cur = await conn.execute("SELECT note_id, heading_hierarchy FROM chunks WHERE note_id = ANY(%s) "
                                     "ORDER BY note_id, chunk_index;", ([n[0] for n in sorteadas],))
            hierarquias = {}
            for nid, h in await cur.fetchall():
                hierarquias.setdefault(nid, []).append(h)
    await db.close()

    print(f"{len(notas)} notas elegíveis em {len(por_pasta)} pastas; sorteadas {len(sorteadas)}:")
    print("  " + " · ".join(f"{mask_pii(p)} {cotas[p]}/{len(por_pasta[p])}" for p in sorted(por_pasta)))
    for k, (nid, caminho, trechos) in enumerate(sorteadas, 1):
        lista = secoes(hierarquias.get(nid, []))
        mais = f" (+{len(lista) - MAX_SECOES})" if len(lista) > MAX_SECOES else ""
        print(f"\n[{k:02d}] {mask_pii(caminho)}  ({trechos} trechos)")
        print(f"     seções: {mask_pii(' · '.join(lista[:MAX_SECOES])) or '(nenhuma: nota sem cabeçalho)'}{mais}")
        print(f"     parecidas: {' · '.join(parecidas.get(nid, [])) or '—'}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
