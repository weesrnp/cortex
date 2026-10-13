"""Conjunto de conferência: perguntas escritas ANTES de medir qualquer variante, sobre notas que não são alvo no
avaliar_busca.py. Serve para conferir se um ganho medido no avaliar_busca.py não é ajuste às perguntas principais:
mudança na busca só entra se não piorar aqui. Não ajustar as perguntas depois da primeira medição.
As listas ficam em perfis/<PERFIL>/perguntas.yaml (`conferencia` e `pdfs`). Não é teste do pytest."""
from src.config import perfil

CONFERENCIA = perfil.perguntas("conferencia")  # (pergunta, nome do arquivo da nota que responde)
# resposta só em PDF do vault (parse_pdf)
PDFS = perfil.perguntas("pdfs")
