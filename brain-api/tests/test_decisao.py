"""Reranque isolado, sem rede, banco ou provedor real."""
import asyncio
import json
import time
from contextlib import asynccontextmanager

import httpx
import pytest

from src import search
from src.config import Settings, settings
from src.decisao import reranquear
from src.perfil import aplicar_decisao, carregar_perfil
from src.search import HybridSearchEngine, SearchResult, is_low_confidence, log_search, max_similarity


def rodar(coro):
    return asyncio.run(coro)


def resultado(i, sim=0.4):
    return SearchResult(chunk_id=i, note_id=i, file_path=f"pasta/{i}.md", title=f"Nota {i}",
                        heading_hierarchy="Seção", content="texto " * 500, metadata={"x": i}, vector_rank=i,
                        text_rank=None, rrf_score=0.1, outlinks=["ligação"], backlinks=[], similarity=sim,
                        vigencia="revogada" if i == 2 else None)


def escolha(handler, candidatos, timeout=1):
    return rodar(reranquear("pergunta", candidatos, url="https://teste.invalid", modelo="modelo", chave="segredo",
                           enunciado="Which passage best answers the user's question?", timeout_s=timeout,
                           transport=httpx.MockTransport(handler)))


def resposta(request, status=200, dados=None):
    if dados is None:
        dados = {"answers": {"choice": {"probabilities": {"p1": 0.1, "p2": 0.9}}}}
    return httpx.Response(status, json=dados, request=request)


def test_reordena_corpo_e_campos():
    candidatos = [resultado(1), resultado(2)]
    def handler(request):
        corpo = json.loads(request.content)
        assert corpo["model"] == "modelo" and corpo["state"]["question"] == "pergunta"
        assert len(corpo["state"]["passages"][0]["content"]) == 2500
        assert corpo["questions"]["choice"]["criteria"] == {"p1": None, "p2": None}
        return resposta(request)
    saida, estado = escolha(handler, candidatos)
    assert estado == "ok" and saida == candidatos[::-1]
    assert saida[0] is candidatos[1] and saida[0].vigencia == "revogada" and saida[0].metadata == {"x": 2}


def test_pergunta_e_titulo_enviados_com_cpf_mascarado():
    candidato = resultado(1)
    candidato.title = 'Ficha 529.982.247-25'
    def handler(request):
        corpo = json.loads(request.content)
        assert '529.982.247-25' not in json.dumps(corpo, ensure_ascii=False)
        assert '529.***.***-25' in corpo['state']['question']
        assert '529.***.***-25' in corpo['state']['passages'][0]['title']
        return resposta(request, dados={'answers': {'choice': {'probabilities': {'p1': 1.0}}}})
    saida, estado = rodar(reranquear('Paciente 529.982.247-25', [candidato], url='https://teste.invalid',
                                      modelo='modelo', chave='segredo', enunciado='Pergunta?', timeout_s=1,
                                      transport=httpx.MockTransport(handler)))
    assert estado == 'ok' and saida == [candidato]


@pytest.mark.parametrize('status', [400, 401, 422, 500, 503])
def test_reserva_http(status):
    candidatos = [resultado(1), resultado(2)]
    saida, estado = escolha(lambda req: resposta(req, status, {"erro": "segredo"}), candidatos)
    assert saida == candidatos and estado == f"reserva:http_{status}" and "segredo" not in estado


@pytest.mark.parametrize('status', [429, 529])
def test_repeticao_curta_uma_vez(status):
    chamadas = []
    def handler(req):
        chamadas.append(1)
        return resposta(req, status)
    candidatos = [resultado(1), resultado(2)]
    saida, estado = escolha(handler, candidatos)
    assert saida == candidatos and estado == f"reserva:http_{status}" and len(chamadas) == 2


def test_429_recupera_na_segunda_tentativa():
    chamadas = []
    def handler(req):
        chamadas.append(1)
        return resposta(req, 429 if len(chamadas) == 1 else 200)
    candidatos = [resultado(1), resultado(2)]
    assert escolha(handler, candidatos) == (candidatos[::-1], 'ok')
    assert len(chamadas) == 2


@pytest.mark.parametrize('dados', [{}, {"answers": {"choice": {"probabilities": {"p1": 0.4}}}},
                                  {"answers": {"choice": {"probabilities": {"p1": "0.4", "p2": 0.6}}}}])
def test_resposta_invalida(dados):
    candidatos = [resultado(1), resultado(2)]
    saida, estado = escolha(lambda req: resposta(req, dados=dados), candidatos)
    assert saida == candidatos and estado == "reserva:resposta_invalida"


def test_rede_e_timeout_nao_expoem_chave():
    def falha(req):
        raise httpx.ConnectError("segredo", request=req)
    async def lenta(req):
        await asyncio.sleep(1)
        return resposta(req)
    candidatos = [resultado(1), resultado(2)]
    assert escolha(falha, candidatos) == (candidatos, "reserva:rede")
    inicio = time.monotonic()
    assert escolha(lenta, candidatos, timeout=0.03) == (candidatos, "reserva:timeout")
    assert time.monotonic() - inicio < 0.5


def test_http_400_reduz_contexto_e_preserva_opcoes():
    corpos = []
    def handler(req):
        corpo = json.loads(req.content)
        corpos.append(corpo)
        if len(corpos) == 1:
            return resposta(req, 400, {"error": "max_tokens_exceeded"})
        return resposta(req)
    candidatos = [resultado(1), resultado(2)]
    saida, estado = escolha(handler, candidatos)
    assert estado == "ok" and saida == candidatos[::-1]
    assert len(corpos[0]['state']['passages'][0]['content']) == 2500
    assert len(corpos[1]['state']['passages'][0]['content']) == 1250
    assert corpos[0]['questions'] == corpos[1]['questions']


@pytest.mark.parametrize('ativa,chave', [(False, True), (True, False)])
def test_desligada_ou_sem_chave_nao_chama_cliente(monkeypatch, ativa, chave):
    monkeypatch.setattr(settings, 'decisao_ativa', ativa)
    monkeypatch.setattr(settings, 'decisao_chave_env', 'CHAVE_TESTE_DECISAO')
    if chave: monkeypatch.setenv('CHAVE_TESTE_DECISAO', 'segredo')
    else: monkeypatch.delenv('CHAVE_TESTE_DECISAO', raising=False)
    candidatos = [resultado(1), resultado(2)]
    limites = []
    async def atual(self, query, limit, *args):
        limites.append(limit)
        return candidatos
    async def proibido(*args, **kwargs):
        raise AssertionError('cliente não deve ser chamado')
    monkeypatch.setattr(HybridSearchEngine, '_search_atual', atual)
    monkeypatch.setattr(search, 'reranquear', proibido)
    saida = rodar(HybridSearchEngine(None, None).search('q', limit=2))
    assert saida == candidatos and limites == [2]
    if ativa:
        assert saida.decisao == 'reserva:sem_chave' and saida.confianca_sim == 0.4
        class Conn:
            async def execute(self, sql, params):
                assert params[2].obj == {'decisao': 'reserva:sem_chave'}
                return self
            async def fetchone(self): return (1,)
        class Pool:
            @asynccontextmanager
            async def connection(self): yield Conn()
        assert rodar(log_search(Pool(), 'rest', 'q', {}, saida, 1)) == 1
    else:
        assert saida is candidatos and not hasattr(saida, 'decisao')


def test_busca_confianca_antes_da_escolha_e_log(monkeypatch):
    monkeypatch.setattr(settings, 'decisao_ativa', True)
    monkeypatch.setattr(settings, 'decisao_n', 6)
    monkeypatch.setattr(settings, 'decisao_chave_env', 'CHAVE_TESTE_DECISAO')
    monkeypatch.setenv('CHAVE_TESTE_DECISAO', 'segredo')
    candidatos = [resultado(i, 0.95 if i == 1 else 0.1) for i in range(1, 7)]
    candidatos[1].vigencia = None
    async def atual(self, query, limit, *args):
        assert limit == 6
        return candidatos
    async def escolher(query, trechos, **kwargs):
        assert kwargs['chave'] == 'segredo'
        return trechos[1:] + trechos[:1], 'ok'
    monkeypatch.setattr(HybridSearchEngine, '_search_atual', atual)
    monkeypatch.setattr(search, 'reranquear', escolher)
    saida = rodar(HybridSearchEngine(None, None).search('q', limit=5))
    assert saida == candidatos[1:] and max_similarity(saida) == 0.95 and not is_low_confidence(saida)
    class Conn:
        async def execute(self, sql, params):
            assert params[2].obj == {'decisao': 'ok'} and params[4] == 0.95 and params[5] is False
            assert 'segredo' not in repr(params)
            return self
        async def fetchone(self): return (17,)
    class Pool:
        @asynccontextmanager
        async def connection(self): yield Conn()
    assert rodar(log_search(Pool(), 'rest', 'q', {}, saida, 5)) == 17


def test_reserva_na_busca_preserva_ordem_e_estado(monkeypatch):
    monkeypatch.setattr(settings, 'decisao_ativa', True)
    monkeypatch.setattr(settings, 'decisao_n', 6)
    monkeypatch.setattr(settings, 'decisao_chave_env', 'CHAVE_TESTE_DECISAO')
    monkeypatch.setenv('CHAVE_TESTE_DECISAO', 'segredo')
    candidatos = [resultado(i, 0.8 if i == 6 else 0.1) for i in range(1, 7)]
    async def atual(self, query, limit, *args): return candidatos
    async def falha(req):
        raise httpx.ConnectError('segredo', request=req)
    async def reranque(query, trechos, **kwargs):
        return await reranquear(query, trechos, **kwargs, transport=httpx.MockTransport(falha))
    monkeypatch.setattr(HybridSearchEngine, '_search_atual', atual)
    monkeypatch.setattr(search, 'reranquear', reranque)
    saida = rodar(HybridSearchEngine(None, None).search('q', limit=5))
    assert saida == candidatos[:5] and saida.decisao == 'reserva:rede'
    assert max_similarity(saida) == 0.8 and not is_low_confidence(saida)


def test_revogadas_ficam_apos_validas_em_ordem_estavel(monkeypatch):
    monkeypatch.setattr(settings, 'decisao_ativa', True)
    monkeypatch.setattr(settings, 'decisao_n', 6)
    monkeypatch.setattr(settings, 'decisao_chave_env', 'CHAVE_TESTE_DECISAO')
    monkeypatch.setenv('CHAVE_TESTE_DECISAO', 'segredo')
    candidatos = [resultado(i) for i in range(1, 7)]
    candidatos[3].vigencia = 'revogada'
    async def atual(self, query, limit, *args): return candidatos
    async def escolher(query, trechos, **kwargs):
        return [trechos[i - 1] for i in (4, 3, 2, 6, 1, 5)], 'ok'
    monkeypatch.setattr(HybridSearchEngine, '_search_atual', atual)
    monkeypatch.setattr(search, 'reranquear', escolher)
    saida = rodar(HybridSearchEngine(None, None).search('q', limit=6))
    assert [r.chunk_id for r in saida] == [3, 6, 1, 5, 4, 2]


def test_buscas_concorrentes_nao_misturam_estado(monkeypatch):
    monkeypatch.setattr(settings, 'decisao_ativa', True)
    monkeypatch.setattr(settings, 'decisao_chave_env', 'CHAVE_TESTE_DECISAO')
    monkeypatch.setenv('CHAVE_TESTE_DECISAO', 'segredo')
    async def atual(self, query, limit, *args):
        return [resultado(1, 0.9 if query == 'alto' else 0.1)]
    async def escolher(query, trechos, **kwargs):
        await asyncio.sleep(0)
        return trechos, 'ok' if query == 'alto' else 'reserva:timeout'
    monkeypatch.setattr(HybridSearchEngine, '_search_atual', atual)
    monkeypatch.setattr(search, 'reranquear', escolher)
    async def executar():
        engine = HybridSearchEngine(None, None)
        return await asyncio.gather(engine.search('alto'), engine.search('baixo'))
    alto, baixo = rodar(executar())
    assert (alto.decisao, max_similarity(alto)) == ('ok', 0.9)
    assert (baixo.decisao, max_similarity(baixo)) == ('reserva:timeout', 0.1)


def test_perfil_e_precedencia_de_ambiente(tmp_path, monkeypatch):
    pasta = tmp_path / 'x'
    pasta.mkdir()
    (pasta / 'perfil.yaml').write_text('decisao:\n  n: 7\n  enunciado: "pergunta?"\n', encoding='utf-8')
    perfil = carregar_perfil('x', str(tmp_path))
    monkeypatch.setenv('DECISAO_N', '9')
    monkeypatch.delenv('DECISAO_ATIVA', raising=False)  # o compose pode ligar a opção no ambiente do container
    config = Settings(_env_file=None)
    aplicar_decisao(perfil, config)
    assert config.decisao_n == 9 and config.decisao_enunciado == 'pergunta?'
    assert config.decisao_ativa is False
