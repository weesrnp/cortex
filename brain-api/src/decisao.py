"""Reranque opcional de trechos; falhas devolvem a ordem recebida, sem expor credenciais."""
import asyncio
import json
import math

import httpx


def _corpo(pergunta, trechos, modelo, enunciado):
    # Importação local evita o ciclo: search usa este módulo para o reranque.
    from .search import mask_pii
    return {
        "model": modelo,
        "state": {"question": mask_pii(pergunta), "passages": [
            {"n": i, "title": mask_pii(r.title), "heading": r.heading_hierarchy or "", "content": r.content[:2500]}
            for i, r in enumerate(trechos, 1)]},
        "questions": {"choice": {"type": "choice", "instructions": enunciado,
                                  "criteria": {f"p{i}": None for i in range(1, len(trechos) + 1)}}},
    }


def _tokens(corpo):
    return len(json.dumps(corpo, sort_keys=True, ensure_ascii=False)) / 2.5


def _reduzir(corpo):
    """Reduz o contexto em ordem de prioridade; nunca altera o enunciado ou as opções."""
    state = corpo["state"]
    for campo in ("content", "heading", "title"):
        alterou = False
        for passagem in state["passages"]:
            valor = passagem[campo]
            if valor:
                passagem[campo] = valor[:len(valor) // 2]
                alterou = True
        if alterou:
            return True
    if state["question"]:
        state["question"] = state["question"][:len(state["question"]) // 2]
        return True
    return False


def _ajustar(corpo):
    """Como na bancada, conserva o máximo de texto igualmente por trecho até 20 mil tokens estimados."""
    if _tokens(corpo) <= 20000:
        return
    passagens = corpo["state"]["passages"]
    conteudos = [p["content"] for p in passagens]
    baixo, alto = 0, max(map(len, conteudos), default=0)
    while baixo < alto:
        meio = (baixo + alto + 1) // 2
        for passagem, conteudo in zip(passagens, conteudos):
            passagem["content"] = conteudo[:meio]
        if _tokens(corpo) <= 20000:
            baixo = meio
        else:
            alto = meio - 1
    for passagem, conteudo in zip(passagens, conteudos):
        passagem["content"] = conteudo[:baixo]
    for campo in ("heading", "title"):
        if _tokens(corpo) <= 20000:
            break
        for passagem in passagens:
            passagem[campo] = passagem[campo][:250]
    if _tokens(corpo) > 20000:
        corpo["state"]["question"] = corpo["state"]["question"][:2500]
    while _tokens(corpo) > 20000 and _reduzir(corpo):
        pass


async def reranquear(pergunta, trechos, *, url, modelo, chave, enunciado, timeout_s, transport=None):
    """Devolve (trechos, estado); o prazo cobre cliente, tentativas e esperas de rede."""
    if not trechos:
        return trechos, "ok"

    async def executar():
        corpo = _corpo(pergunta, trechos, modelo, enunciado)
        _ajustar(corpo)
        if _tokens(corpo) > 20000:
            return trechos, "reserva:resposta_invalida"
        retentou = False
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as cliente:
            while True:
                resposta = await cliente.post(url, json=corpo, headers={"Authorization": f"Bearer {chave}"})
                if resposta.status_code == 400 and "max_tokens_exceeded" in resposta.text and _reduzir(corpo):
                    continue
                if resposta.status_code in (429, 529) and not retentou:
                    retentou = True
                    await asyncio.sleep(0.1)
                    continue
                if resposta.is_error:
                    return trechos, f"reserva:http_{resposta.status_code}"
                try:
                    probs = resposta.json()["answers"]["choice"]["probabilities"]
                    valores = [probs[f"p{i}"] for i in range(1, len(trechos) + 1)]
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                           not math.isfinite(v) or not 0 <= v <= 1 for v in valores) or not sum(valores):
                        raise ValueError
                except (KeyError, TypeError, ValueError, AttributeError):
                    return trechos, "reserva:resposta_invalida"
                return [r for _, r in sorted(zip(valores, trechos), key=lambda x: -x[0])], "ok"

    try:
        return await asyncio.wait_for(executar(), timeout=timeout_s)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        return trechos, "reserva:timeout"
    except Exception:
        # Não propagar mensagens de transportes nem respostas, que podem conter a chave.
        return trechos, "reserva:rede"
