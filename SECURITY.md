# Segurança

O Cortex lê e grava o seu cofre do Obsidian e guarda um índice dele num banco local. Por isso:

- todas as portas (API, Postgres, Ollama) escutam só em `127.0.0.1`; não exponha o Cortex na rede sem uma camada de
  autenticação na frente, porque a API e o MCP não têm login;
- os embeddings são gerados pelo Ollama na sua máquina, sem chamada a serviço externo;
- o explorer web não carrega nada de fora (CSP `'self'`).

## Como reportar

Achou uma falha de segurança? Não abra issue pública. Use o reporte privado do GitHub em
**Security → Report a vulnerability** neste repositório. Resposta em até 7 dias.
