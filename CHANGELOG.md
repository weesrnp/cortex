# Mudanças

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e as versões seguem
[SemVer](https://semver.org/lang/pt-BR/). Versão com mudança de schema avisa aqui: rode a seção "Atualizar" do README.

## [0.1.0] - 2026-10-12

Primeira versão pública.

- Indexação de notas `.md` e PDFs de um cofre do Obsidian em PostgreSQL + pgvector, com embeddings do Ollama local
  (`qwen3-embedding:4b`, 1024 dimensões) e reindexação incremental.
- Busca híbrida: vetor dos trechos, FTS em português e ficha da nota, fundidos por RRF e expandidos por `[[links]]`,
  com aviso de baixa confiança.
- API REST, servidor MCP com 9 ferramentas e explorer web sem CDN.
- Perfis por área (`perfis/<nome>/`), com perfil e cofre de exemplo.
- Scripts de avaliação da busca a partir das perguntas do perfil.
