"""Perfil da área: tudo o que o Cortex sabe sobre o assunto do vault (siglas, instruções do MCP, metadados extraídos
das notas, limiares calibrados e perguntas de avaliação) fica em perfis/<nome>/, fora do código. O núcleo é o mesmo
para qualquer área. Sem perfil (PERFIL vazio), tudo fica neutro: sem siglas, sem metadados, instruções genéricas.
Arquivos (todos opcionais): perfil.yaml, siglas.yaml, instrucoes.md, perguntas.yaml. Ver perfis/exemplo/perfil.yaml."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

import yaml

# Chaves do frontmatter que dizem se a nota ainda vale (src/parser.py: _vigencia). A primeira presente vence.
# O perfil troca a lista inteira de um campo em perfil.yaml (`vigencia: {revogada_em: [revogado]}`).
VIGENCIA_PADRAO: Dict[str, Tuple[str, ...]] = {
    "vigente_desde": ("vigente_desde", "valid_from"),
    "revogada_em": ("revogada_em", "revoked_at"),
    "substituida_por": ("substituida_por", "superseded_by"),
}


@dataclass
class Perfil:
    nome: str = ""
    pasta: Optional[Path] = None
    nome_mcp: str = "cortex"
    titulo: str = "Cortex"
    descricao: str = "Cortex: memória semântica de um vault Obsidian."
    dominio: str = "o assunto deste vault"
    exemplo_reformular: str = ""
    regras: List[str] = field(default_factory=list)
    pastas_oficiais: List[str] = field(default_factory=list)
    instrucao_embedding: str = "Given a question, retrieve note passages that answer it"
    instrucoes_extras: str = ""
    siglas: Dict[str, str] = field(default_factory=dict)
    metadados: Dict[str, Pattern] = field(default_factory=dict)
    limiares: Dict[str, float] = field(default_factory=dict)
    vigencia: Dict[str, List[str]] = field(default_factory=lambda: {k: list(v) for k, v in VIGENCIA_PADRAO.items()})

    def perguntas(self, conjunto: str) -> List[Tuple[str, object]]:
        """[(pergunta, alvo)] de perguntas.yaml; alvo com mais de uma nota aceita vira tupla. Conjunto ausente = []."""
        dados = _yaml(self.pasta / "perguntas.yaml") if self.pasta else {}
        return [(p, tuple(a) if isinstance(a, list) else a) for p, a in dados.get(conjunto) or []]

    def lista(self, chave: str) -> List[str]:
        """Lista simples de perguntas.yaml (ex.: `fora`, perguntas que o vault não responde). Ausente = []."""
        dados = _yaml(self.pasta / "perguntas.yaml") if self.pasta else {}
        return [str(x) for x in dados.get(chave) or []]

    def numero(self, chave: str, padrao: int = 0) -> int:
        """Número solto de perguntas.yaml (ex.: `originais`)."""
        dados = _yaml(self.pasta / "perguntas.yaml") if self.pasta else {}
        return int(dados.get(chave, padrao))


def _yaml(caminho: Path) -> dict:
    if not caminho.is_file():
        return {}
    return yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}


def carregar_perfil(nome: str, base: str) -> Perfil:
    """Lê perfis/<nome>/. Nome dado e pasta inexistente = erro (melhor parar do que rodar sem as regras da área)."""
    if not nome:
        return Perfil()
    pasta = Path(base) / nome
    if not pasta.is_dir():
        raise FileNotFoundError(f"perfil '{nome}' não encontrado em {pasta} (PERFIL e PERFIS_PATH no .env)")
    cfg = _yaml(pasta / "perfil.yaml")
    siglas = _yaml(pasta / "siglas.yaml")
    ruins = [s for s in siglas if not isinstance(s, str)]
    if ruins:  # YAML 1.1 lê NO/ON/YES/OFF sem aspas como booleano
        raise ValueError(f"siglas.yaml do perfil '{nome}': ponha entre aspas as chaves {ruins}")
    instrucoes = pasta / "instrucoes.md"
    p = Perfil(
        nome=nome, pasta=pasta,
        siglas={s: str(e) for s, e in siglas.items()},
        metadados={k: re.compile(v) for k, v in (cfg.get("metadados") or {}).items()},
        instrucoes_extras=instrucoes.read_text(encoding="utf-8").strip() if instrucoes.is_file() else "",
        limiares={k: float(v) for k, v in (cfg.get("limiares") or {}).items()},
    )
    for campo in ("nome_mcp", "titulo", "descricao", "dominio", "exemplo_reformular", "instrucao_embedding"):
        if cfg.get(campo):
            setattr(p, campo, str(cfg[campo]).strip())
    for campo, chaves in (cfg.get("vigencia") or {}).items():
        if campo not in VIGENCIA_PADRAO:
            raise ValueError(f"vigencia do perfil '{nome}': campo desconhecido {campo} (use {', '.join(VIGENCIA_PADRAO)})")
        p.vigencia[campo] = [str(c) for c in (chaves if isinstance(chaves, list) else [chaves])]
    p.regras = [str(r) for r in cfg.get("regras") or []]
    p.pastas_oficiais = [str(r) for r in cfg.get("pastas_oficiais") or []]
    return p


def aplicar_limiares(perfil: Perfil, settings) -> None:
    """Limiar do perfil vale sobre o padrão do config.py; variável de ambiente explícita vence o perfil."""
    for chave, valor in perfil.limiares.items():
        if chave not in type(settings).model_fields:
            raise ValueError(f"limiar desconhecido no perfil '{perfil.nome}': {chave}")
        if chave not in settings.model_fields_set:  # valida como a variável de ambiente (ex.: revogada_peso entre 0 e 1)
            type(settings).__pydantic_validator__.validate_assignment(settings, chave, valor)
