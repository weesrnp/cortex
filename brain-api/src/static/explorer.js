// Explorer do Cortex: grafo (vis-network) e cola entre os fragmentos htmx.
// Sem eval nem script inline: a CSP da página só aceita scripts da própria origem.
(function () {
  "use strict";

  const form = document.getElementById("busca");
  const caixaGrafo = document.getElementById("grafo");
  const nos = new vis.DataSet();
  const arestas = new vis.DataSet();
  let rede = null;
  let notasDosResultados = [];
  let pedidoAtual = 0;
  const DICA = "O grafo das notas encontradas aparece aqui. Clique numa nota para lê-la · duplo clique expande as vizinhas.";

  // /explorer?q=... abre já com a busca (o gatilho "load" do form lê o campo depois deste script).
  const qInicial = new URLSearchParams(location.search).get("q");
  if (qInicial) document.getElementById("q").value = qInicial;

  function cor(nome) {
    return getComputedStyle(document.documentElement).getPropertyValue(nome).trim();
  }

  function opcoes() {
    const fonte = { color: cor("--texto"), size: 13, face: "system-ui, sans-serif",
                    strokeWidth: 3, strokeColor: cor("--fundo") };
    return {
      nodes: { shape: "dot", size: 11, borderWidth: 2, font: fonte, widthConstraint: { maximum: 150 } },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.45 } },
        color: { color: cor("--aresta"), highlight: cor("--acento"), hover: cor("--acento") },
        width: 1.2, smooth: false,
      },
      groups: {
        centro: { size: 17, color: { background: cor("--acento"), border: cor("--acento-forte"),
                  highlight: { background: cor("--acento"), border: cor("--texto") } } },
        nota: { color: { background: cor("--no"), border: cor("--no-borda"),
                highlight: { background: cor("--no"), border: cor("--texto") } } },
        ausente: { size: 8, color: { background: cor("--fundo"), border: cor("--ausente") },
                   shapeProperties: { borderDashes: [3, 3] }, font: { color: cor("--apagado") } },
      },
      physics: { solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -120, springLength: 130, avoidOverlap: 0.6 },
                 stabilization: { iterations: 200 } },
      interaction: { hover: true, tooltipDelay: 120 },
    };
  }

  function montarRede() {
    if (rede) return;
    caixaGrafo.textContent = "";
    rede = new vis.Network(caixaGrafo, { nodes: nos, edges: arestas }, opcoes());
    rede.on("click", function (p) {
      const no = p.nodes.length && nos.get(p.nodes[0]);
      if (no && no.nota) htmx.ajax("GET", "/explorer/nota/" + no.nota, { target: "#nota", swap: "innerHTML" });
    });
    rede.on("doubleClick", function (p) {
      const no = p.nodes.length && nos.get(p.nodes[0]);
      if (no && no.nota) carregarGrafo([no.nota], true);
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      rede.setOptions(opcoes());
    });
  }

  function avisoGrafo(texto) {
    ++pedidoAtual;  // descarta grafo que ainda estava a caminho
    if (rede) { rede.destroy(); rede = null; }
    nos.clear(); arestas.clear();
    caixaGrafo.innerHTML = "";
    const p = document.createElement("p");
    p.className = "grafo-vazio";
    p.textContent = texto;
    caixaGrafo.appendChild(p);
  }

  // somar=false troca o grafo; somar=true expande (mantém o que já está na tela).
  async function carregarGrafo(ids, somar) {
    if (!ids.length) { avisoGrafo(DICA); return; }
    const meu = ++pedidoAtual;  // busca rápida em sequência: só a última resposta vale
    let g;
    try {
      const resp = await fetch("/explorer/grafo?" + ids.map(function (i) { return "ids=" + i; }).join("&"));
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      g = await resp.json();
    } catch (e) {
      if (meu === pedidoAtual) avisoGrafo("Não foi possível carregar o grafo (" + e.message + ").");
      return;
    }
    if (meu !== pedidoAtual) return;
    montarRede();
    if (!somar) { nos.clear(); arestas.clear(); }
    g.nodes.forEach(function (n) {  // vizinha que já era centro continua centro
      const antigo = nos.get(n.id);
      if (antigo && antigo.group === "centro") n.group = "centro";
    });
    nos.update(g.nodes);
    arestas.update(g.edges);
    if (!somar) rede.once("stabilized", function () { rede.fit({ animation: { duration: 300 } }); });
    rede.selectNodes(ids.map(function (i) { return "n" + i; }).filter(function (id) { return nos.get(id); }));
  }

  document.body.addEventListener("htmx:afterSwap", function (e) {
    const alvo = e.detail.target;
    if (alvo.id === "resultados") {
      const lista = alvo.querySelector("[data-notas]");
      notasDosResultados = lista ? lista.dataset.notas.split(",").map(Number) : [];
      carregarGrafo(notasDosResultados, false);  // sem resultados: limpa o grafo anterior
      const q = new FormData(form).get("q").trim();
      history.replaceState(null, "", q ? "?q=" + encodeURIComponent(q) : location.pathname);
    } else if (alvo.id === "nota") {
      // Rola só o painel da nota até o trecho do resultado; a página só rola se o painel estiver fora da tela (celular).
      const caixa = alvo.getBoundingClientRect();
      if (caixa.top < 0 || caixa.top > innerHeight - 120) alvo.scrollIntoView({ block: "start", behavior: "smooth" });
      const destaque = alvo.querySelector(".destaque");
      alvo.scrollTop = destaque ? alvo.scrollTop + destaque.getBoundingClientRect().top - caixa.top - 8 : 0;
    }
  });

  document.addEventListener("click", function (e) {
    const focar = e.target.closest("[data-focar]");
    if (focar) { carregarGrafo([Number(focar.dataset.focar)], false); return; }
    const trecho = e.target.closest(".trecho");
    if (trecho) trecho.classList.toggle("aberto");
  });

  document.getElementById("grafo-resultados").addEventListener("click", function () {
    if (notasDosResultados.length) carregarGrafo(notasDosResultados, false);
  });
})();
