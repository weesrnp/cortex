-- 1. Extensões do Kernel
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- 2. Configuração de Texto Português sem Acentos
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_ts_config WHERE cfgname = 'pt_unaccent'
    ) THEN
        CREATE TEXT SEARCH CONFIGURATION pt_unaccent (COPY = portuguese);
        ALTER TEXT SEARCH CONFIGURATION pt_unaccent
            ALTER MAPPING FOR hword, hword_part, word
            WITH unaccent, portuguese_stem;
    END IF;
END $$;

-- 3. Tabela de Notas Mestres (Obsidian Files)
CREATE TABLE IF NOT EXISTS notes (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,       -- Caminho relativo ex: '01_Projetos/nota.md'
    title TEXT NOT NULL,                  -- Título limpo
    file_hash CHAR(32) NOT NULL,          -- MD5 para evitar re-embedding desnecessário
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Tabela de Chunks e Vetores
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    note_id INT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    heading_hierarchy TEXT,               -- Ex: "Plano Anual > Metas"
    content TEXT NOT NULL,                -- Conteúdo com contexto herdado
    token_count INT,                      -- Estimativa de tamanho
    embedding VECTOR(1024) NOT NULL,      -- qwen3-embedding:4b cortado em 1024 (Matryoshka)
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, -- Metadados do perfil (regex de perfil.yaml)
    tsv_content TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('pt_unaccent', coalesce(heading_hierarchy, '') || ' ' || content)
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_note_chunk UNIQUE (note_id, chunk_index)
);

-- 5. Tabelas de Grafo e Tags
CREATE TABLE IF NOT EXISTS note_tags (
    note_id INT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (note_id, tag)
);

CREATE TABLE IF NOT EXISTS note_links (
    source_note_id INT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    target_title TEXT NOT NULL,
    PRIMARY KEY (source_note_id, target_title)
);

-- 6. Índices de Alta Performance
-- HNSW: a busca híbrida não usa (lado vetorial exato); fica para voltar atrás sem reconstruir
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN para busca léxica full-text unaccent
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON chunks USING gin (tsv_content);

-- GIN JSONB Path para pre-filtering de metadados
CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON chunks USING gin (metadata jsonb_path_ops);

-- B-Tree com pattern ops para filtro instantâneo por subpasta
CREATE INDEX IF NOT EXISTS idx_notes_file_path_pattern
    ON notes (file_path text_pattern_ops);

-- Índices de suporte a grafo e tags
CREATE INDEX IF NOT EXISTS idx_note_links_target ON note_links(target_title);

-- Links resolvem pelo nome do arquivo, como no Obsidian
ALTER TABLE notes ADD COLUMN IF NOT EXISTS stem TEXT
    GENERATED ALWAYS AS (regexp_replace(file_path, '^.*/|\.md$', '', 'g')) STORED;
ALTER TABLE note_links ADD COLUMN IF NOT EXISTS target_stem TEXT
    GENERATED ALWAYS AS (regexp_replace(regexp_replace(target_title, '#.*$', ''), '^.*/|\.md$', '', 'g')) STORED;
CREATE INDEX IF NOT EXISTS idx_notes_stem ON notes(stem);
CREATE INDEX IF NOT EXISTS idx_note_links_target_stem ON note_links(target_stem);
CREATE INDEX IF NOT EXISTS idx_note_tags_tag ON note_tags(tag);

-- Log das buscas de agentes (brain_search) e do REST (/search); pergunta mascarada
CREATE TABLE IF NOT EXISTS search_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    origem TEXT NOT NULL,                          -- mcp | rest
    query TEXT NOT NULL,                           -- mascarada (mask_pii)
    filtros JSONB NOT NULL DEFAULT '{}'::jsonb,    -- include/exclude/folder_prefix usados
    resultados JSONB NOT NULL DEFAULT '[]'::jsonb, -- [{file_path, similarity, rrf_score}] na ordem devolvida
    max_similarity REAL,                           -- NULL = nenhum resultado
    baixa_confianca BOOLEAN NOT NULL,
    duracao_ms INT
);
CREATE INDEX IF NOT EXISTS idx_search_log_created_at ON search_log(created_at);

-- Feedback dos agentes sobre as buscas: avaliação com perguntas reais, não mexe no ranking
CREATE TABLE IF NOT EXISTS search_feedback (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_id BIGINT NOT NULL REFERENCES search_log(id) ON DELETE CASCADE,
    file_path TEXT,                -- nota avaliada (NULL = a busca inteira não serviu)
    util BOOLEAN NOT NULL,
    comentario TEXT,               -- mascarado (mask_pii)
    problema TEXT CONSTRAINT search_feedback_problema_check
        CHECK (problema IN ('desatualizada', 'errada', 'resolvida')),
    conteudo_hash CHAR(32)         -- md5 do corpo da nota no vault na hora do aviso: vale até o conteúdo mudar
);
CREATE INDEX IF NOT EXISTS idx_search_feedback_search ON search_feedback(search_id);
CREATE INDEX IF NOT EXISTS idx_search_feedback_problema ON search_feedback(file_path) WHERE problema IS NOT NULL;

-- 7. Lado léxico (FTS) da busca: modos and | or | idf
-- Termos da pergunta com a mesma normalização do índice (pt_unaccent), quantos trechos têm cada um (df) e o peso BM25.
CREATE OR REPLACE FUNCTION cortex_fts_terms(query_text TEXT)
RETURNS TABLE (lexeme TEXT, lexeme_query TSQUERY, df BIGINT, idf FLOAT)
LANGUAGE sql STABLE AS $$
WITH total AS (SELECT count(*)::float AS n FROM chunks),
termos AS (
    -- aspas e barras invertidas dobradas: o lexema entra no tsquery como está, sem normalizar de novo
    SELECT lex, ('''' || replace(replace(lex, '\', '\\'), '''', '''''') || '''')::tsquery AS q
    FROM unnest(tsvector_to_array(to_tsvector('pt_unaccent', query_text))) lex
)
SELECT t.lex, t.q, d.df, ln(1 + (total.n - d.df + 0.5) / (d.df + 0.5))
FROM termos t, total, LATERAL (SELECT count(*) AS df FROM chunks c WHERE c.tsv_content @@ t.q) d;
$$;

-- Todos os trechos que o FTS aceita no modo pedido, com o score que os ordena (sem filtro de pasta nem LIMIT).
CREATE OR REPLACE FUNCTION cortex_fts_candidates(query_text TEXT, fts_mode TEXT DEFAULT 'and')
RETURNS TABLE (chunk_id INT, score FLOAT)
LANGUAGE sql STABLE AS $$
    SELECT c.id, ts_rank_cd(c.tsv_content, q)::float
    FROM chunks c, plainto_tsquery('pt_unaccent', query_text) q
    WHERE coalesce(fts_mode, 'and') NOT IN ('or', 'idf') AND c.tsv_content @@ q
UNION ALL
    SELECT c.id, ts_rank(c.tsv_content, o.q)::float
    FROM chunks c, (SELECT string_agg(lexeme_query::text, ' | ')::tsquery AS q FROM cortex_fts_terms(query_text)) o
    WHERE fts_mode = 'or' AND c.tsv_content @@ o.q
UNION ALL
    SELECT c.id, sum(t.idf)
    FROM cortex_fts_terms(query_text) t
    JOIN chunks c ON c.tsv_content @@ t.lexeme_query
    WHERE fts_mode = 'idf'
    GROUP BY c.id;
$$;

-- 8. Função SQL Nativa de Busca Híbrida com RRF (Zero Overhead de Rede)
-- Três vias: vetor EXATO dos trechos, FTS e a ficha da nota aos
-- trechos dela).
ALTER TABLE notes ADD COLUMN IF NOT EXISTS ficha_embedding VECTOR(1024);  -- NULL até o indexador gerar
ALTER TABLE notes ADD COLUMN IF NOT EXISTS ficha_hash CHAR(32);           -- md5 do texto da ficha: mudou → refaz

CREATE OR REPLACE FUNCTION cortex_hybrid_search(
    query_text TEXT,
    query_vector VECTOR(1024),
    match_limit INT DEFAULT 5,
    rrf_k INT DEFAULT 60,
    folder_prefix TEXT DEFAULT NULL,
    include_folders TEXT[] DEFAULT NULL,  -- só estes prefixos de pasta (NULL = todas)
    exclude_folders TEXT[] DEFAULT NULL,  -- nunca estes prefixos
    fts_mode TEXT DEFAULT 'and',          -- and | or | idf (ver cortex_fts_candidates)
    note_weight FLOAT DEFAULT 1.0         -- peso da via de fichas; 0 = só vetor + FTS (como até a 006)
)
RETURNS TABLE (
    chunk_id INT,
    note_id INT,
    file_path TEXT,
    title TEXT,
    heading_hierarchy TEXT,
    content TEXT,
    metadata JSONB,
    vector_rank BIGINT,
    text_rank BIGINT,
    rrf_score FLOAT,
    note_rank BIGINT                      -- posição da nota na via de fichas (NULL = fora das 25)
) LANGUAGE sql STABLE AS $$
WITH notas AS MATERIALIZED (  -- notas que passam nos filtros de pasta
    SELECT n.id, n.ficha_embedding
    FROM notes n
    WHERE (folder_prefix IS NULL OR n.file_path LIKE folder_prefix || '%')
      AND (include_folders IS NULL OR EXISTS (SELECT 1 FROM unnest(include_folders) f WHERE n.file_path LIKE f || '%'))
      AND NOT EXISTS (SELECT 1 FROM unnest(coalesce(exclude_folders, '{}')) f WHERE n.file_path LIKE f || '%')
),
distancias AS MATERIALIZED (  -- sem ORDER BY aqui dentro o planejador não tem como trocar pelo HNSW (vetor exato)
    SELECT c.id, c.note_id, c.embedding <=> query_vector AS dist
    FROM chunks c
    JOIN notas n ON n.id = c.note_id
),
vector_matches AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY dist, id) AS rank
    FROM distancias
    ORDER BY dist, id
    LIMIT 25
),
text_matches AS (
    SELECT
        fts.chunk_id AS id,
        ROW_NUMBER() OVER (ORDER BY fts.score DESC, fts.chunk_id) AS rank
    FROM cortex_fts_candidates(query_text, fts_mode) fts
    JOIN chunks c ON c.id = fts.chunk_id
    JOIN notas n ON n.id = c.note_id
    ORDER BY fts.score DESC, fts.chunk_id
    LIMIT 25
),
note_matches AS (
    SELECT n.id AS note_id, ROW_NUMBER() OVER (ORDER BY n.ficha_embedding <=> query_vector, n.id) AS rank
    FROM notas n
    WHERE n.ficha_embedding IS NOT NULL AND note_weight > 0
    ORDER BY n.ficha_embedding <=> query_vector, n.id
    LIMIT 25
),
note_best AS (  -- nota que só a ficha trouxe entra pelo trecho mais próximo da pergunta
    SELECT DISTINCT ON (d.note_id) d.id
    FROM distancias d
    JOIN note_matches nm ON nm.note_id = d.note_id
    ORDER BY d.note_id, d.dist, d.id
),
candidatos AS (
    SELECT id FROM vector_matches
    UNION SELECT id FROM text_matches
    UNION SELECT id FROM note_best
),
combined AS (
    SELECT
        cand.id AS chunk_id,
        v.rank AS v_rank,
        t.rank AS t_rank,
        nm.rank AS n_rank,
        (coalesce(1.0 / (rrf_k + v.rank), 0.0) + coalesce(1.0 / (rrf_k + t.rank), 0.0))::float
            + coalesce(note_weight / (rrf_k + nm.rank), 0.0) AS rrf
    FROM candidatos cand
    JOIN distancias d ON d.id = cand.id
    LEFT JOIN vector_matches v ON v.id = cand.id
    LEFT JOIN text_matches t ON t.id = cand.id
    LEFT JOIN note_matches nm ON nm.note_id = d.note_id
)
SELECT
    c.id AS chunk_id,
    n.id AS note_id,
    n.file_path,
    n.title,
    c.heading_hierarchy,
    c.content,
    c.metadata,
    comb.v_rank AS vector_rank,
    comb.t_rank AS text_rank,
    comb.rrf AS rrf_score,
    comb.n_rank AS note_rank
FROM combined comb
JOIN chunks c ON c.id = comb.chunk_id
JOIN notes n ON n.id = c.note_id
ORDER BY comb.rrf DESC, comb.chunk_id
LIMIT match_limit;
$$;
