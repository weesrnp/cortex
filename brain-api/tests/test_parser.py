import hashlib
import re
from pathlib import Path

import pytest

from src.config import perfil
from src.parser import PARSER_VERSION, VaultParser


@pytest.fixture
def metadados_teste(monkeypatch):
    """Metadados fixos do teste: o resultado não depende do perfil ativo."""
    monkeypatch.setattr(perfil, "metadados", {"tickets": re.compile(r"\b([A-Z]{2,5}-\d{1,5})\b"),
                                              "datas": re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")})


def test_parse_file_chunks_by_heading(tmp_path: Path, metadados_teste):
    vault = tmp_path
    note = vault / "nota.md"
    note.write_text(
        "---\ntags: [projeto]\n---\n"
        "# Intro\nTexto sobre o ticket WEB-12 de 2026-03-10.\n"
        "## Detalhes\nLink para [[Outra Nota]].\n",
        encoding="utf-8"
    )

    parsed = VaultParser.parse_file(note, vault)

    assert parsed.file_path == "nota.md"
    assert "projeto" in parsed.tags
    assert "Outra Nota" in parsed.links
    assert len(parsed.chunks) == 2
    assert parsed.chunks[0].heading_hierarchy.endswith("> Intro")
    assert parsed.chunks[0].metadata == {"tickets": ["WEB-12"], "datas": ["2026-03-10"]}

def test_long_section_overlap_uses_previous_chunk(tmp_path: Path):
    # Regressão: o overlap repetia o parágrafo novo e um 1º parágrafo > 2500 gerava chunk vazio.
    a, b, c = "A" * 3000, "B" * 1500, "C" * 1500
    note = tmp_path / "longa.md"
    note.write_text(f"# Longa\n{a}\n\n{b}\n\n{c}\n", encoding="utf-8")

    chunks = VaultParser.parse_file(note, tmp_path).chunks

    assert all(ch.content.split("\n", 1)[1].strip() for ch in chunks)  # nenhum chunk vazio
    assert chunks[0].content.endswith(a)
    assert chunks[1].content.count(b) == 1 and "A" * 200 in chunks[1].content
    assert "B" * 200 in chunks[2].content and chunks[2].content.endswith(c)

def test_texto_antes_do_primeiro_cabecalho_vira_trecho(tmp_path: Path):
    # v1 descartava: 204 notas do vault tinham descrição fora do índice
    note = tmp_path / "tb_dim_x.md"
    note.write_text("Dimensão da via de administração (intramuscular, subcutânea).\n\n## Colunas\nco_seq\n", encoding="utf-8")

    chunks = VaultParser.parse_file(note, tmp_path).chunks

    assert [c.heading_hierarchy for c in chunks] == ["tb_dim_x", "tb_dim_x > Colunas"]
    assert "intramuscular" in chunks[0].content

def test_comentario_dentro_de_codigo_nao_vira_secao_nem_titulo(tmp_path: Path):
    note = tmp_path / "ficha.md"
    note.write_text("Estrutura Thrift.\n\n```thrift\n# campo obrigatório\nstruct F {}\n```\n## Envio\nPelo LEDI.\n"
                    "```\n# cerca sem fechamento\n", encoding="utf-8")

    parsed = VaultParser.parse_file(note, tmp_path)

    assert parsed.title == "ficha"                                  # não "campo obrigatório"
    assert [c.heading_hierarchy for c in parsed.chunks] == ["ficha", "ficha > Envio"]
    assert "struct F" in parsed.chunks[0].content and "cerca sem fechamento" in parsed.chunks[1].content

def test_versao_do_parser_entra_no_hash(tmp_path: Path):
    note = tmp_path / "n.md"
    note.write_text("texto", encoding="utf-8")
    assert VaultParser.parse_file(note, tmp_path).file_hash == hashlib.md5(f"{PARSER_VERSION}texto".encode()).hexdigest()

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_parse_file_chunks_by_heading(Path(d))
        test_long_section_overlap_uses_previous_chunk(Path(d))
    print("OK: test_parser")


def test_links_como_o_obsidian(tmp_path: Path):
    """Tabela escapa o apelido com "\|" (o alvo não pode levar a barra); link em código é exemplo, não ligação."""
    note = tmp_path / "links.md"
    note.write_text(
        "# Links\n"
        "| unidade | [[Unidade_03_Loja_Centro\\|Loja Centro]] |\n"
        "use o formato `[[Nome_Da_Nota]]` para ligar notas\n"
        "```\n[[SKILL]]\n```\n"
        "e [[Nota Real#Seção|apelido]].\n",
        encoding="utf-8",
    )
    assert sorted(VaultParser.parse_file(note, tmp_path).links) == ["Nota Real#Seção", "Unidade_03_Loja_Centro"]


def test_ficha_da_nota_titulo_pasta_secoes_e_inicio(tmp_path: Path):
    """Ficha (via de notas do RRF): sem o "[cabeçalho]" do trecho, seções sem repetir, início cortado."""
    from src.parser import texto_ficha
    (tmp_path / "Pasta" / "Sub").mkdir(parents=True)
    f = tmp_path / "Pasta" / "Sub" / "n.md"
    f.write_text("# Título\n\n" + "x" * 2000 + "\n\n## A\n\nt\n\n## B\n\nt\n\n## A\n\nt", encoding="utf-8")
    ficha = VaultParser.parse_file(f, tmp_path).ficha()
    assert ficha.startswith("[Título]\nPasta: Pasta/Sub\nSeções: Título; A; B\n")  # o H1 também é seção
    assert ficha.endswith("x" * 1200) and "x" * 1201 not in ficha
    assert texto_ficha("t", "raiz.md", []) == "[t]\nPasta: \nSeções: \n"



def _pdf(paginas) -> bytes:
    """PDF mínimo (Helvetica, uma linha por página) para testar o parse_pdf sem arquivo de fora."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for texto in paginas:
        conteudo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET"
        objs.append(f"<< /Length {len(conteudo)} >>\nstream\n{conteudo}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    saida, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(saida))
        saida += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(saida)
    saida += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    saida += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    saida += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return saida


def test_pdf_vira_nota_com_paginas_no_trecho(tmp_path: Path, metadados_teste):
    from src.parser import PDF_PARSER_VERSION
    (tmp_path / "Docs").mkdir()
    f = tmp_path / "Docs" / "Portaria Teste.pdf"
    paginas = [f"Pagina {i} do ticket WEB-12 " + "texto " * 150 for i in range(1, 4)]
    f.write_bytes(_pdf(paginas))
    nota = VaultParser.parse_pdf(f, tmp_path)
    assert nota.file_path == "Docs/Portaria Teste.pdf" and nota.title == "Portaria Teste" and nota.tags == ["pdf"]
    assert nota.file_hash == hashlib.md5(PDF_PARSER_VERSION.encode() + f.read_bytes()).hexdigest()
    cabecalhos = [c.content.split("\n", 1)[0] for c in nota.chunks]
    assert cabecalhos == ["[Portaria Teste (p. 1-2)]", "[Portaria Teste (p. 3)]"]  # ~2500 caracteres por trecho
    assert nota.chunks[0].metadata["tickets"] == ["WEB-12"]
    assert all(c.heading_hierarchy == "Portaria Teste" for c in nota.chunks)       # sem seções: ficha limpa


def test_pdf_longo_ou_sem_texto_fica_fora(tmp_path: Path):
    longo = tmp_path / "manual.pdf"
    longo.write_bytes(_pdf(["texto suficiente " * 20] * 5))
    assert VaultParser.parse_pdf(longo, tmp_path, max_paginas=4) is None
    vazio = tmp_path / "digitalizado.pdf"
    vazio.write_bytes(_pdf([""]))
    assert VaultParser.parse_pdf(vazio, tmp_path) is None
