"""Relatório de consolidação da memória (2026-10-06): o "dreaming" do AgentMemoryRepo (agente periódico que junta
duplicatas, tira o que ficou velho e resolve contradições) e as observations do Hindsight (crenças consolidadas com
evidência), sem LLM: o Cortex só aponta, com a evidência, e quem decide e edita o vault é o agente ou o usuário.
Só leitura. Seções:
  duplicatas  pares de notas com ficha quase igual (cosseno ≥ 0,98) + evidência no texto: trechos idênticos e
              semelhança de trigramas do texto inteiro (página raspada com o mesmo menu engana a ficha, não o texto)
  orfas       notas sem nenhum [[link]] para outra nota e sem nenhuma nota apontando para elas
  quebrados   [[links]] para notas que não existem, por nota de origem (link para anexo que existe no vault,
              como .pdf ou imagem, não conta: o Obsidian resolve anexo pelo nome do arquivo)
  lacunas     buscas de agentes com baixa confiança e feedback "não serviu" (o que o vault não respondeu)
  inbox       notas paradas em 00_Inbox/ (a triar), mais antigas primeiro
  uteis       notas que mais resolveram buscas (feedback útil)
  sinalizadas notas que o feedback marcou como desatualizada ou errada cujo conteúdo no vault ainda é o da hora do
              aviso (hash do corpo, sem frontmatter) e sem "resolvida" depois (ideia do ai-memory: sinal stale/wrong
              vale para a versão atual da página e alimenta o lint)
Usado pelo MCP (brain_consolidar) e pelo REST (/consolidar)."""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config import perfil
from .search import mask_pii

LIMIAR_DUPLICATA = 0.98  # fichas: faixa ≥ 0,98 tinha 11 pares em 2026-10-06, quase todos cópia (page/page_final/fetch)
SECOES = ("duplicatas", "orfas", "quebrados", "lacunas", "inbox", "uteis", "sinalizadas")
PROBLEMAS = ("desatualizada", "errada")  # search_feedback.problema
RESOLVIDA = "resolvida"                   # encerra os avisos anteriores (nota que não se edita: PDF, nota oficial)
_RE_FM = re.compile(r"\A---[ \t]*\r?\n.*?^---[ \t]*\r?\n", re.S | re.M)


def hash_conteudo(vault_path: Optional[str], file_path: str) -> Optional[str]:
    """md5 do corpo da nota no vault (sem frontmatter: plugin que mexe em `modified:` não conta como correção); PDF
    e outros = bytes. None se o arquivo não existe. Independe do PARSER_VERSION e do updated_at do índice."""
    if not vault_path:
        return None
    base = Path(vault_path).resolve()
    alvo = (base / file_path).resolve()
    if base not in alvo.parents or not alvo.is_file():
        return None
    dados = alvo.read_bytes()
    if alvo.suffix.lower() == ".md":
        dados = _RE_FM.sub("", dados.decode("utf-8", errors="replace"), count=1).encode("utf-8")
    return hashlib.md5(dados).hexdigest()
FORA = "Excalidraw/%"

SQL_DUPLICATAS = r"""
WITH n AS (
    SELECT id, file_path, ficha_embedding FROM notes
    WHERE ficha_embedding IS NOT NULL AND file_path NOT LIKE %(fora)s
),
pares AS (
    SELECT a.id AS ia, b.id AS ib, a.file_path AS pa, b.file_path AS pb,
           1 - (a.ficha_embedding <=> b.ficha_embedding) AS sim
    FROM n a JOIN n b ON a.id < b.id
    WHERE 1 - (a.ficha_embedding <=> b.ficha_embedding) >= %(limiar)s
)
SELECT p.pa, p.pb, p.sim,
       (SELECT count(*) FROM (
            SELECT regexp_replace(content, '^\[[^\]]*\]\n', '') FROM chunks WHERE note_id = p.ia
            INTERSECT
            SELECT regexp_replace(content, '^\[[^\]]*\]\n', '') FROM chunks WHERE note_id = p.ib) x) AS iguais,
       (SELECT count(*) FROM chunks WHERE note_id = p.ia) AS ta,
       (SELECT count(*) FROM chunks WHERE note_id = p.ib) AS tb,
       similarity((SELECT left(string_agg(content, ' ' ORDER BY chunk_index), 20000) FROM chunks WHERE note_id = p.ia),
                  (SELECT left(string_agg(content, ' ' ORDER BY chunk_index), 20000) FROM chunks WHERE note_id = p.ib))
           AS texto
FROM pares p
ORDER BY p.sim DESC, p.pa, p.pb
LIMIT %(limite)s;
"""

SQL_ORFAS = """
SELECT n.file_path FROM notes n
WHERE n.file_path NOT LIKE %(fora)s
  AND NOT EXISTS (SELECT 1 FROM note_links nl JOIN notes t ON t.stem = nl.target_stem
                  WHERE nl.source_note_id = n.id AND t.id <> n.id)
  AND NOT EXISTS (SELECT 1 FROM note_links nl JOIN notes s ON s.id = nl.source_note_id
                  WHERE nl.target_stem = n.stem AND s.id <> n.id)
ORDER BY n.file_path;
"""

SQL_QUEBRADOS = """
SELECT n.file_path, nl.target_stem
FROM note_links nl JOIN notes n ON n.id = nl.source_note_id
WHERE nl.target_stem <> '' AND n.file_path NOT LIKE %(fora)s
  AND NOT EXISTS (SELECT 1 FROM notes t WHERE t.stem = nl.target_stem)
ORDER BY n.file_path, nl.target_stem;
"""


def anexos_do_vault(vault_path: str) -> Set[str]:
    """Nomes dos arquivos que não são nota (.pdf, imagens, planilhas), fora de pastas com ponto, como o Obsidian."""
    base = Path(vault_path)
    if not base.is_dir():
        return set()
    return {f.name for f in base.rglob("*")
            if f.is_file() and f.suffix.lower() != ".md"
            and not any(p.startswith(".") for p in f.relative_to(base).parts)}


def agrupar_quebrados(linhas: List[tuple], anexos: Set[str]) -> List[tuple]:
    """(nota, alvo) → [(nota, total, [alvos])], sem os alvos que são anexos existentes; mais quebrados primeiro."""
    por_nota: Dict[str, List[str]] = {}
    for nota, alvo in linhas:
        if alvo not in anexos:
            por_nota.setdefault(nota, []).append(alvo)
    return sorted(((n, len(a), sorted(set(a))) for n, a in por_nota.items()), key=lambda x: (-x[1], x[0]))

SQL_LACUNAS = """
SELECT query, count(*) AS vezes, max(round(max_similarity::numeric, 2)) AS sim, max(created_at)::date AS ultima
FROM search_log
WHERE baixa_confianca AND created_at > now() - make_interval(days => %(dias)s)
GROUP BY query
ORDER BY vezes DESC, ultima DESC
LIMIT %(limite)s;
"""

SQL_NAO_SERVIU = """
SELECT l.query, f.file_path, f.comentario, f.created_at::date
FROM search_feedback f JOIN search_log l ON l.id = f.search_id
WHERE NOT f.util AND f.problema IS NULL AND f.created_at > now() - make_interval(days => %(dias)s)
ORDER BY f.created_at DESC
LIMIT %(limite)s;
"""

SQL_INBOX = """
SELECT file_path, updated_at::date FROM notes WHERE file_path LIKE '00_Inbox/%%' ORDER BY updated_at, file_path;
"""

SQL_UTEIS = """
SELECT f.file_path, count(*) AS vezes FROM search_feedback f
WHERE f.util AND f.file_path IS NOT NULL
GROUP BY f.file_path ORDER BY vezes DESC, f.file_path LIMIT %(limite)s;
"""

SQL_SINALIZADAS = """
SELECT f.file_path, f.problema, f.conteudo_hash, f.created_at::date, f.comentario
FROM search_feedback f
WHERE f.problema IN ('desatualizada', 'errada')
  AND f.created_at > coalesce((SELECT max(r.created_at) FROM search_feedback r
                               WHERE r.file_path = f.file_path AND r.problema = 'resolvida'), '-infinity')
ORDER BY f.created_at DESC, f.id DESC;
"""


def agrupar_sinalizadas(linhas: List[tuple], hash_atual) -> List[tuple]:
    """(caminho, problema, hash, data, comentário), mais novas primeiro → [(caminho, problema, vezes, última,
    comentário mais novo)] só dos avisos cujo hash ainda é o do arquivo (conteúdo mudou ou arquivo sumiu: resolvido)."""
    atual: Dict[str, Optional[str]] = {}
    grupos: Dict[tuple, list] = {}
    for caminho, problema, h, data, comentario in linhas:
        if caminho not in atual:
            atual[caminho] = hash_atual(caminho)
        if h is None or h != atual[caminho]:
            continue
        g = grupos.setdefault((caminho, problema), [caminho, problema, 0, data, None])
        g[2] += 1
        g[4] = g[4] or comentario
    return sorted((tuple(g) for g in grupos.values()), key=lambda g: (-g[2], -g[3].toordinal(), g[0]))


@dataclass
class Relatorio:
    duplicatas: List[tuple] = field(default_factory=list)   # (a, b, sim, iguais, trechos_a, trechos_b, texto)
    orfas: List[str] = field(default_factory=list)
    quebrados: List[tuple] = field(default_factory=list)    # (nota, total, [alvos])
    lacunas: List[tuple] = field(default_factory=list)      # (pergunta, vezes, sim, última)
    nao_serviu: List[tuple] = field(default_factory=list)   # (pergunta, nota, comentário, data)
    inbox: List[tuple] = field(default_factory=list)        # (caminho, data)
    uteis: List[tuple] = field(default_factory=list)        # (caminho, vezes)
    sinalizadas: List[tuple] = field(default_factory=list)  # (caminho, problema, vezes, última, comentário)


async def gerar(cur, secoes: Optional[List[str]] = None, limite: int = 10, dias: int = 30,
                limiar: float = LIMIAR_DUPLICATA, vault_path: Optional[str] = None) -> Relatorio:
    """Roda só as seções pedidas (None = todas). Só leitura. vault_path: para reconhecer links a anexos."""
    secoes = list(secoes or SECOES)
    p = {"fora": FORA, "limite": limite, "dias": dias, "limiar": limiar}
    rel = Relatorio()
    if "duplicatas" in secoes:
        await cur.execute(SQL_DUPLICATAS, p)
        rel.duplicatas = [(a, b, float(s), i, ta, tb, float(t or 0)) for a, b, s, i, ta, tb, t in await cur.fetchall()]
    if "orfas" in secoes:
        await cur.execute(SQL_ORFAS, p)
        rel.orfas = [r[0] for r in await cur.fetchall()]
    if "quebrados" in secoes:
        await cur.execute(SQL_QUEBRADOS, p)
        rel.quebrados = agrupar_quebrados(await cur.fetchall(), anexos_do_vault(vault_path) if vault_path else set())
    if "lacunas" in secoes:
        await cur.execute(SQL_LACUNAS, p)
        rel.lacunas = [tuple(r) for r in await cur.fetchall()]
        await cur.execute(SQL_NAO_SERVIU, p)
        rel.nao_serviu = [tuple(r) for r in await cur.fetchall()]
    if "inbox" in secoes:
        await cur.execute(SQL_INBOX)
        rel.inbox = [tuple(r) for r in await cur.fetchall()]
    if "uteis" in secoes:
        await cur.execute(SQL_UTEIS, p)
        rel.uteis = [tuple(r) for r in await cur.fetchall()]
    if "sinalizadas" in secoes:
        await cur.execute(SQL_SINALIZADAS)
        rel.sinalizadas = agrupar_sinalizadas(await cur.fetchall(),
                                              lambda c: hash_conteudo(vault_path, c))[:limite]
    return rel


def _mais(total: int, limite: int) -> str:
    return f"\n- … e mais {total - limite}" if total > limite else ""


def _oficiais() -> str:
    """Pastas de notas oficiais do perfil, entre aspas: "A" ou "A", "B"."""
    return ", ".join(f'"{p}"' for p in perfil.pastas_oficiais)


def formatar(rel: Relatorio, secoes: Optional[List[str]] = None, limite: int = 10, dias: int = 30) -> str:
    """Markdown para o agente: cada seção diz o que fazer. Texto de pergunta passa por mask_pii (LGPD)."""
    secoes = list(secoes or SECOES)
    out = ["# Consolidação da memória (só leitura: quem edita é você, no vault)"]
    if "duplicatas" in secoes:
        out.append(f"\n## Quase duplicatas ({len(rel.duplicatas)} pares com ficha ≥ {LIMIAR_DUPLICATA:.2f})")
        out.append("Abra as duas antes de mexer. Cópia de verdade (trechos idênticos, mesmo assunto): junte na mais "
                   "completa, troque os [[links]] da outra e apague-a. Página raspada (menu do site repetido, assunto "
                   "diferente, ex.: duas páginas raspadas do mesmo site): não apague."
                   + (f" Notas oficiais de {_oficiais()} ficam como estão (só registre o achado)."
                      if perfil.pastas_oficiais else ""))
        out += [f"- `{a}` × `{b}` · ficha {s:.3f} · texto {t:.2f} · {i} trecho(s) idêntico(s) ({ta} e {tb} trechos)"
                for a, b, s, i, ta, tb, t in rel.duplicatas] or ["- nenhuma"]
    if "orfas" in secoes:
        out.append(f"\n## Notas órfãs ({len(rel.orfas)}: sem link de saída nem de entrada)")
        out.append("Ligue cada uma à nota-mapa da pasta (ex.: 00_MAPA) ou a notas do mesmo assunto.")
        out += [f"- `{c}`" for c in rel.orfas[:limite]] or ["- nenhuma"]
        out[-1] += _mais(len(rel.orfas), limite)
    if "quebrados" in secoes:
        total = sum(t for _, t, _ in rel.quebrados)
        out.append(f"\n## Links quebrados ({total} em {len(rel.quebrados)} notas)")
        out.append("Aponte para o nome do arquivo certo ou tire os colchetes.")
        out += [f"- `{c}`: {t} → {', '.join(f'[[{a}]]' for a in alvos[:5])}"
                for c, t, alvos in rel.quebrados[:limite]] or ["- nenhum"]
        out[-1] += _mais(len(rel.quebrados), limite)
    if "lacunas" in secoes:
        out.append(f"\n## Lacunas: buscas sem resposta nos últimos {dias} dias ({len(rel.lacunas)})")
        out.append("O vault provavelmente não cobre: se o assunto é do escopo do vault, vale uma nota nova (brain_store).")
        out += [f"- \"{mask_pii(q)}\" · {v}× · sim {s} · {d}" for q, v, s, d in rel.lacunas] or ["- nenhuma"]
        if rel.nao_serviu:
            out.append("\nFeedback \"não serviu\":")
            out += [f"- \"{mask_pii(q)}\"{f' · `{c}`' if c else ''}{f' · {mask_pii(m)}' if m else ''} · {d}"
                    for q, c, m, d in rel.nao_serviu]
    if "inbox" in secoes:
        out.append(f"\n## Inbox a triar ({len(rel.inbox)} notas em 00_Inbox/)")
        out.append("Mova para a pasta do assunto (o caminho muda; os [[links]] pelo nome continuam valendo).")
        out += [f"- `{c}` · desde {d}" for c, d in rel.inbox[:limite]] or ["- vazia"]
        out[-1] += _mais(len(rel.inbox), limite)
    if "uteis" in secoes:
        out.append(f"\n## Notas que mais resolveram buscas ({len(rel.uteis)})")
        out += [f"- `{c}` · {v}×" for c, v in rel.uteis] or ["- ainda sem feedback (brain_feedback)"]
    if "sinalizadas" in secoes:
        out.append(f"\n## Sinalizadas como desatualizada ou errada ({len(rel.sinalizadas)})")
        out.append("Confira a fonte oficial e corrija a nota no vault; mudar o texto da nota tira o aviso. "
                   + (f"Nota oficial de {_oficiais()} e PDF" if perfil.pastas_oficiais else "PDF")
                   + " não se editam: registre a versão nova em outra nota, "
                   "ligue as duas e encerre o aviso com brain_feedback(nº, util=False, nota=..., problema=\"resolvida\").")
        out += [f"- `{c}` · {pb} · {v}× · {d}{f' · {mask_pii(m)}' if m else ''}"
                for c, pb, v, d, m in rel.sinalizadas] or ["- nenhuma"]
    return "\n".join(out)


def como_dict(rel: Relatorio) -> Dict[str, list]:
    """Para o REST (/consolidar)."""
    return {
        "duplicatas": [{"a": a, "b": b, "similaridade": round(s, 4), "texto": round(t, 3), "trechos_iguais": i,
                        "trechos_a": ta, "trechos_b": tb} for a, b, s, i, ta, tb, t in rel.duplicatas],
        "orfas": rel.orfas,
        "quebrados": [{"nota": c, "total": t, "alvos": list(a)} for c, t, a in rel.quebrados],
        "lacunas": [{"pergunta": mask_pii(q), "vezes": v, "similaridade": float(s) if s is not None else None,
                     "ultima": str(d)} for q, v, s, d in rel.lacunas],
        "nao_serviu": [{"pergunta": mask_pii(q), "nota": c, "comentario": mask_pii(m) if m else None,
                        "data": str(d)} for q, c, m, d in rel.nao_serviu],
        "inbox": [{"nota": c, "desde": str(d)} for c, d in rel.inbox],
        "uteis": [{"nota": c, "vezes": v} for c, v in rel.uteis],
        "sinalizadas": [{"nota": c, "problema": pb, "vezes": v, "ultima": str(d),
                         "comentario": mask_pii(m) if m else None} for c, pb, v, d, m in rel.sinalizadas],
    }
