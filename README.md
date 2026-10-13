<p align="center">
  <img src="assets/cortex-banner.png" alt="Cortex: memória semântica local para o seu cofre do Obsidian" width="100%">
</p>

> Memória de longo prazo para agentes de IA, feita do que você já escreveu. O Cortex transforma o seu cofre do
> Obsidian num índice que o agente consulta antes de responder, e tudo roda na sua máquina.

<p align="center">
  <a href="https://github.com/weesrnp/cortex/releases"><img alt="versão" src="https://img.shields.io/github/v/release/weesrnp/cortex?label=vers%C3%A3o"></a>
  <a href="https://github.com/weesrnp/cortex/actions/workflows/testes.yml"><img alt="testes" src="https://github.com/weesrnp/cortex/actions/workflows/testes.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="licença MIT" src="https://img.shields.io/badge/licen%C3%A7a-MIT-blue"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB">
  <img alt="PostgreSQL 16 + pgvector" src="https://img.shields.io/badge/postgres-16%20%2B%20pgvector-336791">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-HTTP-8b5cf6">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/lang-English-lightgrey"></a>
</p>

## Por que o Cortex

Agentes de IA esquecem tudo entre uma sessão e outra, e as memórias que vêm prontas nas ferramentas guardam o que o
agente decidiu anotar, num formato que você não lê nem edita. Enquanto isso, o que você sabe de verdade já está
escrito no Obsidian.

O Cortex liga as duas pontas.

- **Usa o cofre que você já tem.** Notas, PDFs e `[[links]]` viram um índice de busca. O cofre continua sendo a fonte
  da verdade: o índice se reconstrói a partir dele a qualquer momento.
- **Tudo fica na sua máquina.** Vetores pelo Ollama local, banco em Docker, portas só em 127.0.0.1, explorer sem
  nada carregado de fora e números de documento pessoais mascarados nas respostas. Serve para cofres que você nunca
  mandaria para a nuvem.
- **Busca que se mede.** Vetor, palavras e uma ficha por nota, fundidos por RRF e expandidos pelo grafo de links. Cada
  mudança na busca passa por perguntas de avaliação antes de entrar.
- **O agente sabe quando não sabe.** Se o cofre provavelmente não tem a resposta, a busca avisa, e o agente não inventa
  a partir de trechos soltos.
- **Memória que o agente também escreve.** Notas novas e decisões técnicas viram arquivos `.md` no seu cofre, com a
  versão anterior guardada. Um relatório de faxina aponta duplicatas, órfãs, links quebrados e perguntas sem resposta.
- **Um núcleo, várias áreas.** Siglas, pastas, fatos fixos e perguntas de avaliação ficam num perfil por área. O código
  é o mesmo para trabalho, estudo ou vida pessoal.

## O que ele faz

- Busca híbrida com três vias fundidas por RRF: vetor dos trechos, busca textual em português (FTS) e uma ficha por
  nota (título, pasta, seções e começo do texto). Depois expande pelos `[[wiki-links]]`.
- Aviso de baixa confiança quando o cofre provavelmente não tem a resposta, para o agente não inventar.
- Servidor MCP com 9 ferramentas: `brain_search`, `brain_recall`, `brain_graph`, `brain_list`, `brain_store`,
  `brain_record_decision`, `brain_historico`, `brain_feedback` e `brain_consolidar`.
- Memória que se mantém: `brain_store` guarda a versão anterior da nota (`.cortex_historico/`), `brain_feedback`
  registra se a busca resolveu e `brain_consolidar` devolve um relatório de faxina (só leitura).
- API REST: `/health`, `/search`, `/recall/{titulo}`, `/graph/traverse`, `/graph/path` e `/consolidar`.
- Explorer web em `/explorer`: busca com o diagnóstico de cada via e o grafo de links.
- Indexação incremental: só o que mudou volta para o Ollama, e trecho com texto igual reaproveita o vetor.

## Começar

Precisa de Docker com Compose v2.24 ou mais novo, uns 8 GB de RAM livres para o Docker e 6 GB de disco (o modelo de
embeddings tem cerca de 2,5 GB). GPU NVIDIA é opcional.

```bash
git clone https://github.com/weesrnp/cortex.git
cd cortex
cp .env.example .env
docker compose up -d --build
```

Na primeira subida o Ollama baixa o `qwen3-embedding:4b` e o Cortex indexa o cofre de exemplo. Em conexão lenta o
download leva alguns minutos; acompanhe com `docker compose logs -f brain-api`. Depois:

- API: http://127.0.0.1:8900/health
- Busca: http://127.0.0.1:8900/search?q=prazo+de+resposta+do+suporte
- Explorer: http://127.0.0.1:8900/explorer
- MCP (HTTP streamable): http://127.0.0.1:8900/mcp

Com GPU NVIDIA (NVIDIA Container Toolkit; no Windows, Docker Desktop com WSL2):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### Linux

O Cortex grava notas no cofre montado, então o container roda com o uid do dono da pasta. Se o seu usuário não for o
1000, ponha no `.env` o resultado de `id -u` e `id -g` em `CORTEX_UID` e `CORTEX_GID`. Com SELinux (Fedora, RHEL),
acrescente `:z` ao volume do cofre no `docker-compose.yml`.

### Ollama que já roda na máquina

No macOS isso é o recomendado, porque o Docker no Mac não usa a GPU. No `.env`:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_PORT=11435   # a 11434 já é do seu Ollama
```

No Linux, o Ollama do host precisa escutar fora do 127.0.0.1 (`OLLAMA_HOST=0.0.0.0`). O serviço `ollama` do compose
continua subindo, mas fica sem modelo carregado.

### Ligar num agente

Qualquer cliente MCP com transporte HTTP serve. A configuração costuma ser assim:

```json
{
  "mcpServers": {
    "cortex": { "type": "http", "url": "http://127.0.0.1:8900/mcp" }
  }
}
```

Ao conectar, o agente recebe as instruções do perfil: como buscar, quais pastas existem e quais fatos não pode
contradizer.

## Usar com o seu cofre

1. No `.env`, aponte `VAULT_HOST_PATH` para a pasta do cofre (no Windows, use barras normais: `C:/Users/voce/Notas`).
   Use um cofre fora da pasta do repositório; o cofre de exemplo serve só para testar.
2. Copie `perfis/exemplo` para `perfis/minha_area` e ponha `PERFIL=minha_area` no `.env`.
3. Ajuste os arquivos do perfil (veja abaixo) e rode `docker compose up -d`.
4. Escreva perguntas de avaliação e meça (seção "Medir antes de mudar").

O cofre é montado com escrita porque `brain_store` e `brain_record_decision` criam notas. Pastas e arquivos que começam
com ponto ficam fora do índice, como no Obsidian.

## Perfis

O código em `brain-api/src` é o mesmo para qualquer assunto. O que é da sua área fica em `perfis/<nome>/`, e todos os
arquivos são opcionais:

| Arquivo | Para quê |
|---|---|
| `perfil.yaml` | Nome do servidor MCP e título, descrição do cofre, domínio, exemplo de pergunta, regras extras, instrução do modelo de embedding, regex de metadados (viram etiquetas no explorer), pastas oficiais e limiares |
| `siglas.yaml` | Glossário. A pergunta ganha "(SIGLA: extenso)" no fim antes de virar vetor, nos dois sentidos |
| `instrucoes.md` | Texto que entra nas instruções do MCP: mapa das pastas, fatos fixos |
| `perguntas.yaml` | Perguntas de avaliação (`principais`, `conferencia`, `pdfs`) e perguntas que o cofre não responde (`fora`) |

Com `PERFIL=` vazio no `.env`, o Cortex roda neutro: sem siglas, sem metadados e com instruções genéricas. Um perfil
que não existe impede a subida, para você não rodar sem as regras da área sem perceber.

Uma instância por área funciona melhor que um índice com tudo misturado: notas de assuntos diferentes competem na
busca, e dados sensíveis de uma área não precisam dividir banco com as outras. Para isso, rode uma cópia por área com
`COMPOSE_PROJECT_NAME`, `API_PORT` e `POSTGRES_PORT` diferentes e um só Ollama para todas (`OLLAMA_BASE_URL`), ou
também `OLLAMA_PORT` diferente.

## Medir antes de mudar

Mudança na busca só vale se medir melhor. Os scripts em `brain-api/tests` usam as perguntas do perfil:

```bash
docker compose exec brain-api python tests/avaliar_busca.py                          # hit@1, hit@5 e MRR por modo do FTS
docker compose exec brain-api python tests/avaliar_busca.py and --conjunto conferencia
docker compose exec brain-api python tests/calibrar_confianca.py   # limiar de baixa confiança
docker compose exec brain-api python tests/diagnosticar_busca.py   # por que a nota certa não apareceu
docker compose exec brain-api python tests/sortear_notas.py        # notas sorteadas para escrever perguntas novas
docker compose exec brain-api python tests/avaliar_feedback.py     # perguntas reais que os agentes marcaram como úteis
```

Escreva as perguntas antes de mexer na busca e não ajuste as da `conferencia` depois de medir: elas servem para
conferir se um ganho nas `principais` não é ajuste às próprias perguntas. Num cofre real com cerca de 550 notas e
6.700 trechos, as 46 perguntas principais dão MRR 0,700, com a nota certa no top-5 em 38 delas.

## Testes

```bash
docker compose exec brain-api python -m pytest tests -q
```

Os testes de banco usam o Postgres do próprio Cortex, dentro de uma transação desfeita no fim, e não chamam o Ollama.
Fora do Docker, os testes de banco são pulados se `DATABASE_URL` não estiver definida. O CI do GitHub roda tudo contra
um Postgres com pgvector a cada push.

## Atualizar

```bash
git pull
docker compose up -d --build
docker compose exec -T postgres psql -U cortex -d cortex -v ON_ERROR_STOP=1 < db/init.sql
```

O `db/init.sql` cria o schema na primeira subida e pode ser rodado de novo sem perder nada: é assim que uma versão nova
traz tabelas, colunas e funções. O [CHANGELOG](CHANGELOG.md) avisa quando isso é necessário. Usuário e banco
diferentes de `cortex`: troque no comando. No PowerShell, a última linha fica
`Get-Content db/init.sql | docker compose exec -T postgres psql -U cortex -d cortex -v ON_ERROR_STOP=1`.

## Como funciona

```
cofre (.md e .pdf) ──► parser (trechos por cabeçalho, ficha da nota, metadados do perfil)
                          │
                          ▼
                Ollama (qwen3-embedding:4b, vetor de 1024 dimensões)
                          │
                          ▼
     PostgreSQL + pgvector: notas, trechos, links, fichas, log de buscas e feedback
                          │
                          ▼
     busca: vetor exato + FTS português + ficha ──► RRF ──► expansão por [[links]]
                          │
                 REST · MCP (/mcp) · explorer (/explorer)
```

- Os trechos seguem os cabeçalhos de nível 1 a 4 (`Nota > Seção`), com o texto antes do primeiro cabeçalho num trecho
  próprio. Seção longa é cortada por parágrafo; nota sem cabeçalho vira um trecho só.
- O vetor da pergunta é comparado com todos os trechos (busca exata). Até alguns milhares de trechos isso é rápido e não
  perde resultado como o índice HNSW, que é aproximado.
- PDFs entram como notas, com o número da página no trecho. Ficam de fora PDFs com mais de 300 páginas, os sem texto
  (digitalizados), as cópias com texto igual (fica uma) e os de caminho que casa com `PDF_EXCLUDE_PATTERN`.
- As buscas pela API e pelo MCP vão para `search_log`, com a pergunta já mascarada. É a base do `brain_consolidar` e do
  `avaliar_feedback.py`. O log e o feedback não vêm do cofre: apagar o volume do banco apaga os dois.

## Configuração

As variáveis ficam no `.env` (veja `.env.example`) e chegam ao Cortex pelo compose. Os padrões estão em
`brain-api/src/config.py`. As mais usadas:

| Variável | Padrão | O que faz |
|---|---|---|
| `VAULT_HOST_PATH` | `./exemplo-vault` | Pasta do cofre no host |
| `PERFIL` | `exemplo` | Pasta em `perfis/`; vazio = sem perfil |
| `CORTEX_UID` / `CORTEX_GID` | `1000` | Usuário do container (Linux: o dono do cofre) |
| `EMBEDDING_MODEL` | `qwen3-embedding:4b` | Modelo do Ollama |
| `OLLAMA_BASE_URL` | serviço `ollama` do compose | Ollama a usar |
| `API_PORT` / `POSTGRES_PORT` / `OLLAMA_PORT` | 8900 / 5433 / 11434 | Portas no host, só em 127.0.0.1 |
| `INDEX_PDF` | `true` | PDFs do cofre no índice |
| `PDF_EXCLUDE_PATTERN` | `(?i)confidencial\|sigiloso\|nominal` | Regex do caminho de PDF que fica fora do índice |
| `EXPAND_SIGLAS` | `true` | Expansão de siglas na pergunta |
| `LOW_CONFIDENCE_THRESHOLD` / `AUTO_LINK_THRESHOLD` | 0,62 / 0,70 | Limiares (o perfil também define; a variável vence) |

A senha do Postgres vale quando o volume é criado. Para trocar depois, use `ALTER USER` no banco ou apague o volume.

Trocar de modelo de embedding não reindexa sozinho: apague o volume do banco (`docker compose down -v`, que também
apaga o log de buscas) e suba de novo, depois recalibre os limiares com `calibrar_confianca.py`. Outra dimensão de
vetor exige também mudar `VECTOR(1024)` no `db/init.sql`.

Para desenvolver, `docker-compose.dev.yml` monta o código e os testes por cima da imagem:
`docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d`.

## Créditos

O explorer usa [htmx](https://htmx.org) 2.0.11 (BSD Zero Clause) e
[vis-network](https://github.com/visjs/vis-network) 10.1.2 (Apache-2.0 ou MIT), servidos localmente. Algumas ideias de
memória vieram de outros projetos abertos (Cognee, ai-memory, AgentMemoryRepo, Hindsight, OpenMemory), citados no
código onde entram.

## Licença

MIT. Veja [LICENSE](LICENSE). Falhas de segurança: [SECURITY.md](SECURITY.md).
