"""Por que a nota certa não aparece? Para cada pergunta do avaliar_busca.py fora do top-5 (--todas: todas):
posição da nota certa no vetor e no FTS contra o índice INTEIRO (sem o corte de 25), quem ficou na frente,
quais termos da pergunta a nota tem e o estado dos trechos dela. No fim, compara o índice com o parser deste código.
Só lê (transação READ ONLY). Não imprime texto das notas: só nomes, seções, posições e tamanhos.
    docker compose exec brain-api python tests/diagnosticar_busca.py [--todas]
Não é teste do pytest."""
import asyncio
import re
import sys
from pathlib import Path

import frontmatter

from avaliar_busca import PERGUNTAS, aceitas, posicao, rotulo
from src.config import settings
from src.database import db
from src.embeddings import OllamaEmbeddingClient
from src.parser import VaultParser
from src.search import HybridSearchEngine, mask_pii, note_stem

CORTE_EMBEDDING = 6000  # caracteres ≈ 2048 tokens (num_ctx): acima disso o embedding só vê o começo do trecho
SEG_POR_TRECHO = 18 * 60 / 7086  # reindexação medida em 2026-09-23: ~18 min para 7.086 trechos
FORA_EXCALIDRAW = "n.file_path NOT LIKE 'Excalidraw/%%'"
RE_DADOS_EMBUTIDOS = re.compile(r'```compressed-json\n[\s\S]*?```|data:[a-z]+/[\w.+-]+;base64,[A-Za-z0-9+/=]+')


def fora_dos_trechos(arquivo: str, trechos: list) -> tuple:
    """(linhas, caracteres) do texto do arquivo que não estão em nenhum trecho. Não conta frontmatter, cabeçalhos,
    desenho do Excalidraw nem imagem base64 (não são texto pesquisável)."""
    try:
        corpo = frontmatter.loads(arquivo).content
    except Exception:
        corpo = arquivo
    corpo = RE_DADOS_EMBUTIDOS.sub("", corpo)
    junto = "\n".join(trechos)
    linhas = [l.strip() for l in corpo.splitlines() if len(l.strip()) >= 15 and not l.lstrip().startswith("#")]
    fora = [l for l in linhas if l not in junto]
    return len(fora), sum(map(len, fora))


def txt_fora(par: tuple) -> str:
    return f"{par[0]} linhas ({par[1]} chars) fora" if par[0] else "nada fora"


def num(x: float, casas: int = 3) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


async def diagnosticar(cur, engine, client, i, pergunta, alvos, todas) -> None:
    v = await client.get_query_embedding(pergunta)
    stems = [note_stem(r.file_path) for r in
             await engine.search(pergunta, limit=5, expand_graph=False, query_vector=v)]
    pos = posicao(stems, alvos)
    if pos and not todas:
        return
    alvo = aceitas(alvos)[0]  # com duas notas aceitas, examina a principal
    print(f"\n[{i}] {pergunta}")
    await cur.execute("SELECT file_path FROM notes n WHERE stem = %s ORDER BY length(file_path) LIMIT 1;", (alvo,))
    linha = await cur.fetchone()
    if not linha:
        print(f"    alvo {alvo}: NÃO ESTÁ NO ÍNDICE (arquivo renomeado/apagado? atualizar perguntas.yaml do perfil)")
        return
    caminho = linha[0]
    print(f"    alvo {rotulo(alvos)} — {f'{pos}º no top-5' if pos else 'fora do top-5'}")

    # Vetor: posição da nota (pelo melhor trecho) entre todas as notas, e quem vem na frente
    await cur.execute(f"""
        WITH r AS (
            SELECT n.stem, c.heading_hierarchy, 1 - (c.embedding <=> %(v)s::vector) AS cos
            FROM chunks c JOIN notes n ON n.id = c.note_id WHERE {FORA_EXCALIDRAW}
        ), notas AS (
            SELECT stem, max(cos) AS cos, (array_agg(heading_hierarchy ORDER BY cos DESC))[1] AS secao,
                   row_number() OVER (ORDER BY max(cos) DESC, stem) AS pos
            FROM r GROUP BY stem
        )
        SELECT stem, cos, secao, pos, (SELECT count(*) + 1 FROM r WHERE r.cos > notas.cos),
               (SELECT count(*) FROM r)
        FROM notas WHERE pos <= 3 OR stem = %(alvo)s ORDER BY pos;
    """, {"v": str(v), "alvo": alvo})
    linhas = await cur.fetchall()
    dados_alvo = next((l for l in linhas if l[0] == alvo), None)
    if dados_alvo is None:
        print("    vetor: a nota não tem trechos no índice")
    else:
        _, cos, secao, pos_nota, pos_trecho, total = dados_alvo
        print(f"    vetor: nota em {pos_nota}º (melhor trecho: {pos_trecho}º de {total}, cos {num(cos)}; "
              f"1º geral {num(linhas[0][1])}) · seção \"{mask_pii(secao or '')}\"")
        na_frente = [f"{mask_pii(s)} {num(c)}" for s, c, _, p, _, _ in linhas if p < pos_nota]
        print(f"    na frente: {' · '.join(na_frente) or 'ninguém'}")

    # FTS: quais termos da pergunta a nota tem, e onde ela fica em cada modo
    await cur.execute("""
        SELECT t.lexeme, bool_or(c.tsv_content @@ t.lexeme_query)
        FROM cortex_fts_terms(%(q)s) t CROSS JOIN chunks c JOIN notes n ON n.id = c.note_id
        WHERE n.file_path = %(caminho)s
        GROUP BY t.lexeme, t.df ORDER BY t.df;
    """, {"q": pergunta, "caminho": caminho})
    termos = await cur.fetchall()
    faltam = [t for t, tem in termos if not tem]
    modos = []
    for modo in ("and", "idf"):
        await cur.execute(f"""
            WITH cand AS (
                SELECT f.chunk_id, n.file_path, rank() OVER (ORDER BY f.score DESC) AS pos
                FROM cortex_fts_candidates(%(q)s, %(modo)s) f
                JOIN chunks c ON c.id = f.chunk_id JOIN notes n ON n.id = c.note_id WHERE {FORA_EXCALIDRAW}
            )
            SELECT (SELECT count(*) FROM cand), (SELECT min(pos) FROM cand WHERE file_path = %(caminho)s);
        """, {"q": pergunta, "modo": modo, "caminho": caminho})
        total, pos_fts = await cur.fetchone()
        modos.append(f"{modo}: " + (f"nota em {pos_fts}º de {total} trechos" if pos_fts
                                    else f"a nota não casa ({total} trechos casam)"))
    print(f"    FTS: a nota tem {len(termos) - len(faltam)} de {len(termos)} termos"
          + (f" (faltam: {', '.join(faltam)})" if faltam else "") + " · " + " · ".join(modos))

    # Trechos da nota: tamanho, corte no embedding, texto do arquivo que não entrou, e o que o parser novo faz
    await cur.execute("SELECT content FROM chunks c JOIN notes n ON n.id = c.note_id WHERE n.file_path = %s;",
                      (caminho,))
    trechos = [r[0] for r in await cur.fetchall()]
    arquivo = Path(settings.vault_path) / caminho
    estado = (f"    trechos: {len(trechos)} no índice, maior {max(map(len, trechos), default=0)} chars, "
              f"{sum(len(t) > CORTE_EMBEDDING for t in trechos)} cortados no embedding")
    if arquivo.exists():
        texto = arquivo.read_text(encoding="utf-8", errors="replace")
        novos = [c.content for c in VaultParser.parse_file(arquivo, Path(settings.vault_path)).chunks]
        estado += (f" · {txt_fora(fora_dos_trechos(texto, trechos))}"
                   f" → parser deste código: {len(novos)} trechos, {txt_fora(fora_dos_trechos(texto, novos))}, "
                   f"maior {max(map(len, novos), default=0)} chars")
    print(estado)


async def resumo_do_indice(cur) -> None:
    """Índice atual x parser deste código, no vault inteiro (se ainda não reindexou, mostra o efeito da mudança)."""
    await cur.execute("SELECT n.file_path, c.content FROM notes n LEFT JOIN chunks c ON c.note_id = n.id;")
    por_nota: dict = {}
    for caminho, conteudo in await cur.fetchall():
        por_nota.setdefault(caminho, []).extend([conteudo] if conteudo is not None else [])
    base = Path(settings.vault_path)
    antes = depois = cortados_antes = cortados_depois = 0
    perdidas_antes, perdidas_depois = [0, 0], [0, 0]  # notas com texto fora, caracteres fora
    maiores = {"antes": [], "depois": []}  # (caracteres fora, nota)
    for caminho, trechos in por_nota.items():
        arquivo = base / caminho
        if not arquivo.exists():
            continue
        novos = [c.content for c in VaultParser.parse_file(arquivo, base).chunks]
        antes, depois = antes + len(trechos), depois + len(novos)
        cortados_antes += sum(len(t) > CORTE_EMBEDDING for t in trechos)
        cortados_depois += sum(len(t) > CORTE_EMBEDDING for t in novos)
        if not caminho.startswith("Excalidraw/"):
            texto = arquivo.read_text(encoding="utf-8", errors="replace")
            for acc, lista, nome in ((perdidas_antes, trechos, "antes"), (perdidas_depois, novos, "depois")):
                _, chars = fora_dos_trechos(texto, lista)
                acc[0] += chars > 0
                acc[1] += chars
                if chars:
                    maiores[nome].append((chars, caminho))
    print(f"\nÍNDICE INTEIRO ({len(por_nota)} notas)            no banco agora   parser deste código")
    print(f"  trechos                                  {antes:>10}   {depois:>10}")
    print(f"  trechos cortados no embedding (>{CORTE_EMBEDDING} chars) {cortados_antes:>7}   {cortados_depois:>10}")
    print(f"  notas com texto fora de todos os trechos  {perdidas_antes[0]:>9}   {perdidas_depois[0]:>10}")
    print(f"  caracteres fora de todos os trechos       {perdidas_antes[1]:>9}   {perdidas_depois[1]:>10}")
    print(f"  reindexar tudo com o parser deste código: ~{depois * SEG_POR_TRECHO / 60:.0f} min de GPU")
    for nome, rotulo in (("antes", "no banco agora"), ("depois", "parser deste código")):
        top = ", ".join(f"{mask_pii(note_stem(c))} ({n})" for n, c in sorted(maiores[nome], reverse=True)[:5])
        print(f"  mais texto fora ({rotulo}): {top or 'nenhuma'}")


async def main(todas: bool) -> None:
    await db.open()
    client = OllamaEmbeddingClient(settings.ollama_base_url, settings.embedding_model)
    engine = HybridSearchEngine(db.pool, client)
    async with db.pool.connection() as conn:
        async with conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            async with conn.cursor() as cur:
                for i, (pergunta, alvo) in enumerate(PERGUNTAS, 1):
                    await diagnosticar(cur, engine, client, i, pergunta, alvo, todas)
                await resumo_do_indice(cur)
    await db.close()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main("--todas" in sys.argv))
