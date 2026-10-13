from src.search import cap_per_note, note_stem


def test_no_maximo_dois_trechos_por_nota_mantendo_ordem():
    rows = [("c1", 1), ("c2", 1), ("c3", 1), ("c4", 2), ("c5", 1), ("c6", 3)]
    assert [r[0] for r in cap_per_note(rows, limit=5)] == ["c1", "c2", "c4", "c6"]
    assert [r[0] for r in cap_per_note(rows, limit=2)] == ["c1", "c2"]


def test_note_stem_igual_ao_alvo_de_wikilink():
    assert note_stem("01_Projetos/Nota B2.md") == "Nota B2"
    assert note_stem("Raiz.md") == "Raiz"
    assert note_stem("Excalidraw/Desenho.excalidraw.md") == "Desenho.excalidraw"
