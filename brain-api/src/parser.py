import io
import logging
import re
import hashlib
from datetime import date, datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import frontmatter

from .config import perfil

logger = logging.getLogger("cortex.parser")

@dataclass
class NoteChunk:
    chunk_index: int
    heading_hierarchy: str
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ParsedNote:
    file_path: str
    title: str
    file_hash: str
    tags: List[str]
    links: List[str]
    chunks: List[NoteChunk]
    # vigente_desde / revogada_em (date) e substituida_por (stem da nota nova) vindos do frontmatter; ausente = None
    vigencia: Dict[str, Any] = field(default_factory=dict)

    def ficha(self) -> str:
        return texto_ficha(self.title, self.file_path, [(c.heading_hierarchy, c.content) for c in self.chunks])

# Entra no file_hash: mudar a divisão em trechos → subir a versão, e o próximo scan reindexa o vault sozinho
# (o indexador reaproveita o vetor de todo trecho com texto igual). v5 = v1 + PREAMBULO + CERCAS, medidas uma a uma
# na busca; títulos pais (v2) e corte de parágrafo gigante pioraram a busca.
PARSER_VERSION = "7"  # 7: vigência do frontmatter em notes; 6: links sem a barra de tabela e sem os de código
                      # (trechos iguais ao v5 nos dois: vetores reaproveitados, a reindexação só regrava as notas)
PDF_PARSER_VERSION = "1"  # entra no file_hash dos PDFs (parse_pdf): mudar o corte em trechos → subir


RE_CABECALHO_TRECHO = re.compile(r"^\[[^\]]*\]\n")


def texto_ficha(title: str, file_path: str, trechos: List[Tuple[str, str]]) -> str:
    """Ficha da nota para a 3ª via do RRF: título, pasta, nomes das seções
    e o começo do 1º trecho (preâmbulo ou resumo) sem o "[cabeçalho]". trechos = [(heading_hierarchy, content)] na
    ordem. Mudar este texto refaz as fichas no próximo scan (ficha_hash); medir antes com tests/avaliar_busca.py."""
    pasta = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
    secoes = list(dict.fromkeys(h.split(" > ", 1)[1] for h, _ in trechos if h and " > " in h))[:40]
    inicio = RE_CABECALHO_TRECHO.sub("", trechos[0][1])[:1200] if trechos else ""
    return f"[{title}]\nPasta: {pasta}\nSeções: {'; '.join(secoes)}\n{inicio}"


def hash_ficha(texto: str) -> str:
    return hashlib.md5(texto.encode("utf-8")).hexdigest()


class VaultParser:
    RE_WIKILINK = re.compile(r'\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]')
    RE_TAG = re.compile(r'(?:^|\s)#([a-zA-Z0-9_\-\/]+)')
    RE_HEADING = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    RE_CERCA = re.compile(r'^ {0,3}(?:```|~~~)', re.MULTILINE)
    PREAMBULO = True  # texto antes do 1º cabeçalho vira trecho da própria nota (o v1 descartava: 204 notas)
    CERCAS = True     # "# comentário" dentro de bloco de código não abre seção nem vira título (32 notas)

    RE_CODIGO_LINHA = re.compile(r'`[^`\n]*`')

    @classmethod
    def _sem_codigo(cls, body: str) -> str:
        """Texto sem blocos ```/~~~ e sem `código` em linha (só para achar links)."""
        fora, dentro = [], False
        for linha in body.split("\n"):
            if cls.RE_CERCA.match(linha):
                dentro = not dentro
            elif not dentro:
                fora.append(linha)
        return cls.RE_CODIGO_LINHA.sub("", "\n".join(fora))

    @classmethod
    def _meta(cls, body: str) -> Dict[str, Any]:
        """Metadados globais da nota (vão em todos os trechos): uma lista por regex de `metadados` no perfil."""
        return {chave: list(set(rx.findall(body))) for chave, rx in perfil.metadados.items()}

    @staticmethod
    def _data(valor) -> Optional[date]:
        """date do YAML (2026-01-01), datetime, texto ISO (2026-01-01, 2026-01-01T10:00) ou dd/mm/aaaa; senão None."""
        if isinstance(valor, datetime):
            return valor.date()
        if isinstance(valor, date):
            return valor
        texto = str(valor).strip()
        for formato, tamanho in (("%Y-%m-%d", 10), ("%d/%m/%Y", 10)):
            try:
                return datetime.strptime(texto[:tamanho], formato).date()
            except ValueError:
                pass
        return None

    @classmethod
    def _vigencia(cls, fm_data: Dict[str, Any], rel_path: str = "") -> Dict[str, Any]:
        """Vigência da nota pelo frontmatter (chaves em perfil.vigencia). substituida_por aceita "[[Nota]]",
        "[[Pasta/Nota#Seção|apelido]]", [[Nota]] sem aspas (o YAML lê lista de lista) ou lista (vale a primeira) e
        vira o stem. Data que não se entende anula a vigência inteira da nota (na dúvida, a nota vale) e vai para o
        log: sem isso, "revogada_em: 31/13/2026" + substituida_por revogaria a nota hoje."""
        def primeiro(campo):
            return next((fm_data[k] for k in perfil.vigencia[campo] if fm_data.get(k) not in (None, "")), None)

        nova = primeiro("substituida_por")
        while isinstance(nova, list):
            nova = nova[0] if nova else None
        if nova is not None:
            nova = str(nova).strip().removeprefix("[[").removesuffix("]]").split("|", 1)[0].split("#", 1)[0]
            nova = nova.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md").strip() or None
        vig = {"vigente_desde": primeiro("vigente_desde"), "revogada_em": primeiro("revogada_em")}
        for campo, valor in vig.items():
            if valor is not None:
                vig[campo] = cls._data(valor)
                if vig[campo] is None:
                    logger.warning("vigência ignorada em %s: %s=%r não é data (use AAAA-MM-DD)", rel_path, campo, valor)
                    return {"vigente_desde": None, "revogada_em": None, "substituida_por": None}
        return {**vig, "substituida_por": nova}

    @classmethod
    def parse_pdf(cls, path: Path, base_vault_path: Path, max_paginas: int = 300) -> Optional["ParsedNote"]:
        """PDF do vault (norma, relatório, manual curto) vira nota: título = nome do arquivo, trechos de
        páginas juntadas até ~2500 caracteres com "[Título (p. N-M)]" no começo. file_path termina em .pdf, então o
        stem (alvo de [[link]]) é "Nome.pdf", como o Obsidian liga anexos. None = PDF longo demais (manual de
        equipamento) ou sem texto (digitalizado). Sem cabeçalhos: a ficha (via de notas) fica com título + começo."""
        from pypdf import PdfReader  # import aqui: o parser de .md não depende do pypdf
        dados = path.read_bytes()
        leitor = PdfReader(io.BytesIO(dados))
        if len(leitor.pages) > max_paginas:
            return None
        paginas = [cls._texto_pdf(p.extract_text() or "") for p in leitor.pages]
        if sum(len(t) for t in paginas) < 200:
            return None
        title = path.stem
        rel_path = str(path.relative_to(base_vault_path)).replace("\\", "/")
        meta = cls._meta("\n".join(paginas))
        chunks: List[NoteChunk] = []
        buffer, inicio, fim = "", 1, 1
        for n, texto in enumerate(paginas, 1):
            for pedaco in cls._pedacos_pdf(texto):
                if buffer and len(buffer) + len(pedaco) > 2500:
                    cls._fecha_pdf(chunks, title, buffer, inicio, fim, meta)
                    buffer, inicio = "", n
                buffer += ("\n\n" if buffer else "") + pedaco
                fim = n
        if buffer.strip():
            cls._fecha_pdf(chunks, title, buffer, inicio, fim, meta)
        return ParsedNote(
            file_path=rel_path, title=title,
            file_hash=hashlib.md5(PDF_PARSER_VERSION.encode() + dados).hexdigest(),
            tags=["pdf"], links=[], chunks=chunks)

    @staticmethod
    def _texto_pdf(texto: str) -> str:
        """Espaços repetidos e linhas em branco em excesso saem; quebra de linha fica (tabelas e listas)."""
        texto = re.sub(r"[ \t ]+", " ", texto)
        return re.sub(r"\n\s*\n\s*\n+", "\n\n", texto).strip()

    @staticmethod
    def _pedacos_pdf(texto: str, limite: int = 2500) -> List[str]:
        """Página maior que o limite (tabela grande) é cortada em linhas inteiras."""
        if len(texto) <= limite:
            return [texto] if texto else []
        pedacos, atual = [], ""
        for linha in texto.split("\n"):
            if atual and len(atual) + len(linha) > limite:
                pedacos.append(atual)
                atual = ""
            atual += ("\n" if atual else "") + linha
        return pedacos + ([atual] if atual else [])

    @staticmethod
    def _fecha_pdf(chunks: list, title: str, texto: str, de: int, ate: int, meta: dict) -> None:
        paginas = f"p. {de}" if de == ate else f"p. {de}-{ate}"
        conteudo = f"[{title} ({paginas})]\n{texto.strip()}"
        chunks.append(NoteChunk(len(chunks), title, conteudo, len(conteudo) // 4, meta))

    @classmethod
    def parse_file(cls, path: Path, base_vault_path: Path) -> ParsedNote:
        raw_content = path.read_text(encoding='utf-8', errors='replace')
        file_hash = hashlib.md5((PARSER_VERSION + raw_content).encode('utf-8')).hexdigest()
        rel_path = str(path.relative_to(base_vault_path)).replace("\\", "/")

        try:
            post = frontmatter.loads(raw_content)
            body, fm_data = post.content, post.metadata
        except Exception:  # YAML inválido no frontmatter: indexa o texto bruto
            body, fm_data = raw_content, {}

        # Extração de Título
        title = path.stem
        if "title" in fm_data and isinstance(fm_data["title"], str):
            title = fm_data["title"]
        else:
            first_h1 = (next((m for m in cls._headings(body) if m.group(1) == "#"), None) if cls.CERCAS
                        else re.search(r'^#\s+(.+)$', body, re.MULTILINE))
            if first_h1:
                title = first_h1.group(first_h1.lastindex).strip()

        # Extração de Tags
        tags = set()
        if "tags" in fm_data:
            t = fm_data["tags"]
            if isinstance(t, list):
                tags.update([str(x).strip() for x in t])
            elif isinstance(t, str):
                tags.update([x.strip() for x in t.split(",")])
        for inline_tag in cls.RE_TAG.findall(body):
            tags.add(inline_tag.strip())

        # Extração de Wiki-links
        # Como o Obsidian: link dentro de código é exemplo, não ligação; em tabela "[[Nota\|apelido]]" deixa "\\" no alvo
        links = set()
        for target, _ in cls.RE_WIKILINK.findall(cls._sem_codigo(body)):
            target = target.strip().rstrip("\\").strip()
            if target:
                links.add(target)

        # Heading-First Chunking
        chunks = cls._chunk_by_headings(title, body, cls._meta(body))

        return ParsedNote(
            file_path=rel_path,
            title=title,
            file_hash=file_hash,
            tags=list(tags),
            links=list(links),
            chunks=chunks,
            vigencia=cls._vigencia(fm_data, rel_path),
        )

    @classmethod
    def _headings(cls, text: str) -> list:
        """Cabeçalhos Markdown; com CERCAS, os que estão dentro de bloco de código não contam."""
        encontrados = list(cls.RE_HEADING.finditer(text))
        if not cls.CERCAS:
            return encontrados
        cercas = [m.start() for m in cls.RE_CERCA.finditer(text)]
        blocos = list(zip(cercas[0::2], cercas[1::2] + [len(text)]))  # bloco sem fechamento vai até o fim
        return [m for m in encontrados if not any(a <= m.start() < b for a, b in blocos)]

    @classmethod
    def _chunk_by_headings(cls, note_title: str, text: str, base_meta: dict) -> List[NoteChunk]:
        headings = cls._headings(text)
        if not headings:
            # Sem cabeçalhos: nota única (qualquer tamanho; cortar parágrafo gigante piorou a busca)
            return [NoteChunk(0, note_title, f"[{note_title}]\n{text.strip()}", len(text) // 4, base_meta)]

        secoes = [(note_title, text[:headings[0].start()].strip())] if cls.PREAMBULO else []
        for i, match in enumerate(headings):
            end_pos = headings[i + 1].start() if (i + 1) < len(headings) else len(text)
            secoes.append((f"{note_title} > {match.group(2).strip()}", text[match.end():end_pos].strip()))

        chunks: List[NoteChunk] = []
        for hierarchy, section_body in secoes:
            if not section_body:
                continue
            for pedaco in cls._pedacos(hierarchy, section_body):
                full_chunk = f"[{hierarchy}]\n{pedaco}"
                chunks.append(NoteChunk(len(chunks), hierarchy, full_chunk, len(full_chunk) // 4, base_meta))
        return chunks

    @classmethod
    def _paragrafos(cls, section_body: str) -> List[str]:
        return section_body.split("\n\n")

    @classmethod
    def _pedacos(cls, hierarchy: str, section_body: str) -> List[str]:
        """Seção de até ~800 tokens fica inteira; maior, junta parágrafos até 2500 chars, com os 200 últimos
        caracteres do trecho anterior no começo do seguinte."""
        if (len(hierarchy) + len(section_body)) // 4 <= 800:
            return [section_body]
        pedacos, buffer = [], ""
        for p in cls._paragrafos(section_body):
            if buffer and len(buffer) + len(p) > 2500:
                pedacos.append(buffer.strip())
                buffer = buffer[-200:] + "\n\n" + p  # 200 chars de overlap com o chunk anterior
            else:
                buffer += ("\n\n" if buffer else "") + p
        if buffer.strip():
            pedacos.append(buffer.strip())
        return pedacos
