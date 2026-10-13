"""Expansão de siglas na pergunta, nos dois sentidos, só para o vetor (o FTS em modo `and` exigiria os termos novos).
"LGPD" ganha "Lei Geral de Proteção de Dados" e vice-versa: as notas usam ora a sigla, ora o extenso,
e o vetor da sigla sozinha fica longe do extenso. As siglas vêm do perfil da área (src/perfil.py).
Sigla casa só em MAIÚSCULAS e palavra inteira ("sim" não é o SIM); extenso casa sem acento e sem caixa."""
import re
import unicodedata

from .config import perfil

# Lista do perfil (perfis/<nome>/siglas.yaml); sem perfil, nada é expandido.
SIGLAS = perfil.siglas


def _norm(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def _compilar(siglas: dict) -> list:
    """[(sigla, extenso, regex da sigla, regex do extenso normalizado)]."""
    return [(s, e, re.compile(rf"(?<![\w]){s}(?![\w])"), re.compile(rf"(?<![\w]){re.escape(_norm(e))}(?![\w])"))
            for s, e in siglas.items()]


_PADRAO = _compilar(SIGLAS)


def expandir_siglas(pergunta: str, siglas: dict = None) -> str:
    """Anexa ao fim "(SIGLA: Extenso; ...)" para cada sigla ou extenso achado. Sem achado, devolve igual.
    siglas: outro glossário (testes); padrão = o do perfil."""
    norm = _norm(pergunta)
    regras = _PADRAO if siglas is None else _compilar(siglas)
    pares = [f"{s}: {e}" for s, e, rx_s, rx_e in regras if rx_s.search(pergunta) or rx_e.search(norm)]
    return f"{pergunta} ({'; '.join(pares)})" if pares else pergunta
