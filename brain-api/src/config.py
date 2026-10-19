from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .perfil import aplicar_limiares, carregar_perfil

FtsMode = Literal["and", "or", "idf"]

class Settings(BaseSettings):
    database_url: str = "postgresql://cortex:cortex123@postgres:5432/cortex"
    ollama_base_url: str = "http://ollama:11434"
    embedding_model: str = "qwen3-embedding:4b"
    embedding_dim: int = 1024       # corte Matryoshka; tem que bater com VECTOR(n) no banco
    auto_link_threshold: float = 0.70  # padrão do qwen3-embedding:4b; o perfil pode trocar (perfis/<nome>/perfil.yaml)
    auto_link_max: int = 3
    embedding_num_ctx: int = 2048  # chunks ~700 tokens; 4096+ transborda uma GPU de 6 GB para a CPU
    note_weight: float = 1.0  # peso da via de fichas no RRF; 0 desliga. Mudar = medir com tests/avaliar_busca.py
    fts_mode: FtsMode = "and"  # lado léxico; trocar só se tests/avaliar_busca.py medir melhor
    revogada_peso: float = Field(0.5, ge=0, le=1)  # RRF de nota revogada (frontmatter revogada_em) × peso; 1 só avisa
    fuso: str = ""  # fuso do "hoje" da vigência (ex.: America/Sao_Paulo); vazio = relógio do container (UTC no Docker)
    low_confidence_threshold: float = 0.62  # maior cosseno abaixo = "vault provavelmente não tem"; perfil pode trocar
    expand_siglas: bool = True  # pergunta ganha "SIGLA: extenso" no vetor (siglas.py); medir com EXPAND_SIGLAS=false
    index_pdf: bool = True  # PDFs do vault entram no índice (parse_pdf); medir com tests/avaliar_busca.py
    pdf_max_pages: int = 300  # PDF maior fica fora (manual de equipamento com milhares de páginas)
    pdf_exclude_pattern: str = r"(?i)confidencial|sigiloso|nominal"  # caminho de PDF que fica fora (dado pessoal); "" = nenhum
    search_log_enabled: bool = True  # tabela search_log: buscas de agentes e REST, pergunta mascarada
    vault_path: str = "/vault"
    perfil: str = ""  # pasta em perfis_path com siglas, instruções, metadados e limiares da área (src/perfil.py)
    perfis_path: str = "/app/perfis"
    api_host: str = "0.0.0.0"
    api_port: int = 8900
    log_level: str = "info"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("fuso")
    @classmethod
    def _fuso_existe(cls, v: str) -> str:
        if v:  # FUSO errado para a subida, em vez de derrubar cada busca com ZoneInfoNotFoundError
            try:
                ZoneInfo(v)
            except (ZoneInfoNotFoundError, ValueError) as e:
                raise ValueError(f"FUSO desconhecido: {v!r} (ex.: America/Sao_Paulo)") from e
        return v

settings = Settings()
perfil = carregar_perfil(settings.perfil, settings.perfis_path)
aplicar_limiares(perfil, settings)
