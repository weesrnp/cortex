# Mudanças

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e as versões seguem
[SemVer](https://semver.org/lang/pt-BR/). Versão com mudança de schema avisa aqui: rode a seção "Atualizar" do README.

## [Não lançado]

Muda o schema: rode a seção "Atualizar" do README (três colunas novas em `notes`).

- Vigência das notas pelo frontmatter: `vigente_desde`, `revogada_em` e `substituida_por` (ou `valid_from`,
  `revoked_at`, `superseded_by`; o perfil troca as chaves em `vigencia:`). Nota revogada fica no índice, com o RRF
  multiplicado por `REVOGADA_PESO` (0,5), e o MCP, a API e o explorer avisam qual nota vale. `FUSO` define o "hoje".
- `PARSER_VERSION` 7: a primeira subida relê as notas para preencher a vigência (vetores reaproveitados).

## [0.1.0] - 2026-10-12

Primeira versão pública.

- Indexação de notas `.md` e PDFs de um cofre do Obsidian em PostgreSQL + pgvector, com embeddings do Ollama local
  (`qwen3-embedding:4b`, 1024 dimensões) e reindexação incremental.
- Busca híbrida: vetor dos trechos, FTS em português e ficha da nota, fundidos por RRF e expandidos por `[[links]]`,
  com aviso de baixa confiança.
- API REST, servidor MCP com 9 ferramentas e explorer web sem CDN.
- Perfis por área (`perfis/<nome>/`), com perfil e cofre de exemplo.
- Scripts de avaliação da busca a partir das perguntas do perfil.
