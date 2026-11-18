"""Módulo de Decisão do EcoSort — sistema baseado em regras (agente reativo).

PAPEL REAL NO SISTEMA
---------------------
Os módulos de Visão e de PLN apenas *percebem* o mundo: um diz "qual material
provavelmente é" (com uma confiança) e o outro diz "o que o usuário quer saber"
(também com confiança). Nenhum dos dois sabe o que RESPONDER. Quem decide é este
módulo, que funciona como um agente reativo baseado em regras:

    percepção (classe, confiança_visão, top2, intenção, confiança_pln)
        -> regras SE-ENTÃO avaliadas em ordem de prioridade
        -> ação (resposta textual + regra acionada + justificativa)

Ele é responsável por:
  1. Consultar a base de conhecimento (dados/regras_descarte.json) — cor da
     lixeira, reciclabilidade, preparo e destino de cada material;
  2. Tratar a INCERTEZA das percepções: se a visão não tem confiança suficiente,
     o agente não "chuta": pede outra foto ou apresenta as 2 classes mais
     prováveis; se as duas levam à MESMA lixeira (ex.: papel x papelão), ele
     consegue responder com segurança mesmo assim;
  3. Tratar perguntas fora do escopo e intenções ambíguas;
  4. Tornar a decisão explicável: toda resposta carrega o ID e a descrição da
     regra que a gerou (exibidos no app).

Sem este componente o sistema seria apenas dois classificadores desconectados.

REGRAS (ordem de prioridade — a primeira cuja condição é verdadeira dispara)
---------------------------------------------------------------------------
R01  SE intenção = fora_do_escopo E confiança_pln >= 0.4  ENTÃO explica o escopo do assistente.
R02  SE confiança_visão < limiar_mínimo (0.35)            ENTÃO pede uma nova foto.
R03  SE confiança_visão < limiar (0.6) E top1 e top2 têm a mesma lixeira
                                                          ENTÃO responde com base no grupo comum.
R04  SE confiança_visão < limiar (0.6)                    ENTÃO mostra as 2 classes prováveis + resposta de cada uma e pede outra foto.
R05  SE confiança_pln < limiar_pln (0.4)                  ENTÃO dá um resumo geral do material e sugere reformular.
     (inclui perguntas classificadas como fora_do_escopo com baixa confiança: na dúvida,
      o agente prefere ajudar com um resumo a recusar a pergunta)
R10..R33  SE classe = X E intenção = Y E confiança_visão >= limiar
                                                          ENTÃO resposta específica (X, Y) — 6 classes × 4 intenções = 24 regras.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from src.machine_learning.utils import RAIZ

CAMINHO_REGRAS: Path = RAIZ / "dados" / "regras_descarte.json"
CLASSES: list[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
INTENCOES_DOMINIO: list[str] = ["como_descartar", "e_reciclavel", "onde_levar", "preparo_antes"]
INTENCOES: list[str] = INTENCOES_DOMINIO + ["fora_do_escopo"]


@dataclass
class Percepcao:
    """Entradas do agente: saídas dos módulos de Visão e PLN."""

    classe: str
    confianca_visao: float
    top2: list[tuple[str, float]]
    intencao: str
    confianca_pln: float = 1.0


@dataclass
class Decisao:
    """Saída do agente: resposta + rastreabilidade da regra acionada."""

    regra_id: str
    regra_descricao: str
    resposta: str
    classe_considerada: str | None
    precisa_nova_foto: bool = False
    detalhes: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def carregar_base(caminho: str = str(CAMINHO_REGRAS)) -> dict[str, Any]:
    """Carrega a base de conhecimento JSON."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Geradores de resposta (ações) para cada intenção do domínio
# --------------------------------------------------------------------------
def _resp_como_descartar(m: dict[str, Any]) -> str:
    return f"Isso parece ser **{m['nome_pt']}**. {m['como_descartar']}"


def _resp_e_reciclavel(m: dict[str, Any]) -> str:
    if m["reciclavel"]:
        return (f"Sim! **{m['nome_pt'].capitalize()}** é reciclável. "
                f"Coloque na lixeira **{m['lixeira_cor'].upper()}**. {m['observacao']}")
    return (f"Não. Isso parece ser **{m['nome_pt']}**, que não tem reciclagem viável. "
            f"Descarte na lixeira **{m['lixeira_cor'].upper()}** (rejeito). {m['observacao']}")


def _resp_onde_levar(m: dict[str, Any]) -> str:
    return f"Para **{m['nome_pt']}**: {m['destino']}"


def _resp_preparo_antes(m: dict[str, Any]) -> str:
    passos = "\n".join(f"- {p}" for p in m["preparo"])
    return f"Antes de descartar **{m['nome_pt']}**:\n{passos}"


ACOES: dict[str, Callable[[dict[str, Any]], str]] = {
    "como_descartar": _resp_como_descartar,
    "e_reciclavel": _resp_e_reciclavel,
    "onde_levar": _resp_onde_levar,
    "preparo_antes": _resp_preparo_antes,
}


@dataclass(frozen=True)
class RegraEspecifica:
    """Regra explícita: SE classe=X E intenção=Y E confiança>=limiar ENTÃO resposta."""

    id: str
    classe: str
    intencao: str

    def descricao(self, limiar: float) -> str:
        return (f"SE classe={self.classe} E intenção={self.intencao} E confiança>={limiar:.2f} "
                f"ENTÃO resposta '{self.intencao}' para '{self.classe}'")

    def acionar(self, base: dict[str, Any]) -> str:
        return ACOES[self.intencao](base["materiais"][self.classe])


def gerar_regras_especificas() -> dict[tuple[str, str], RegraEspecifica]:
    """Tabela explícita das 24 regras (classe × intenção), IDs R10..R33."""
    tabela: dict[tuple[str, str], RegraEspecifica] = {}
    n = 10
    for c in CLASSES:
        for i in INTENCOES_DOMINIO:
            tabela[(c, i)] = RegraEspecifica(f"R{n:02d}", c, i)
            n += 1
    return tabela


REGRAS_ESPECIFICAS: dict[tuple[str, str], RegraEspecifica] = gerar_regras_especificas()


def decidir(p: Percepcao, base: dict[str, Any] | None = None) -> Decisao:
    """Motor de inferência: avalia as regras em ordem de prioridade e retorna a 1ª que dispara."""
    base = base or carregar_base()
    meta = base["_meta"]
    limiar = float(meta["limiar_confianca_visao"])
    limiar_min = float(meta["limiar_confianca_minima_visao"])
    limiar_pln = float(meta["limiar_confianca_pln"])
    materiais = base["materiais"]
    det = {"limiar_visao": limiar, "limiar_minimo_visao": limiar_min, "limiar_pln": limiar_pln}

    # R01 — pergunta fora do escopo (com confiança suficiente do PLN)
    if p.intencao == "fora_do_escopo" and p.confianca_pln >= limiar_pln:
        return Decisao("R01", f"SE intenção=fora_do_escopo E confiança_pln>={limiar_pln:.2f} "
                              "ENTÃO explicar o escopo",
                       "Sou o EcoSort e só consigo ajudar com o descarte de resíduos. "
                       "Pergunte, por exemplo: *como descarto isso?*, *isso é reciclável?*, "
                       "*onde posso levar?* ou *precisa lavar antes?*", None, detalhes=det)

    # R02 — visão muito incerta: não arrisca resposta
    if p.confianca_visao < limiar_min:
        return Decisao("R02", f"SE confiança_visão<{limiar_min:.2f} ENTÃO pedir nova foto",
                       f"Não consegui identificar o material com segurança (confiança de "
                       f"{p.confianca_visao:.0%}). Envie outra foto com o objeto sozinho, "
                       "bem iluminado e sobre um fundo neutro.", None, precisa_nova_foto=True, detalhes=det)

    (c1, p1), (c2, p2) = p.top2[0], p.top2[1]
    if p.confianca_visao < limiar:
        # R03 — incerto, mas as duas hipóteses levam à MESMA lixeira
        if materiais[c1]["lixeira_cor"] == materiais[c2]["lixeira_cor"] and p.intencao in ACOES:
            resp = ACOES[p.intencao](materiais[c1])
            return Decisao("R03", f"SE confiança_visão<{limiar:.2f} E lixeira(top1)=lixeira(top2) "
                                  "ENTÃO responder pelo grupo comum",
                           f"Não tenho certeza se é **{materiais[c1]['nome_pt']}** ({p1:.0%}) ou "
                           f"**{materiais[c2]['nome_pt']}** ({p2:.0%}), mas os dois vão para a "
                           f"lixeira **{materiais[c1]['lixeira_cor'].upper()}**, então a orientação é a mesma.\n\n"
                           + resp, c1, detalhes=det)
        # R04 — incerto entre materiais com destinos diferentes
        intencao = p.intencao if p.intencao in ACOES else "como_descartar"
        resp = (f"Fiquei em dúvida entre duas opções (confiança abaixo de {limiar:.0%}):\n\n"
                f"**1) {materiais[c1]['nome_pt'].capitalize()} ({p1:.0%})** — {ACOES[intencao](materiais[c1])}\n\n"
                f"**2) {materiais[c2]['nome_pt'].capitalize()} ({p2:.0%})** — {ACOES[intencao](materiais[c2])}\n\n"
                "Se puder, envie outra foto para eu confirmar.")
        return Decisao("R04", f"SE confiança_visão<{limiar:.2f} E lixeiras diferentes "
                              "ENTÃO mostrar top-2 e pedir nova foto",
                       resp, None, precisa_nova_foto=True, detalhes=det)

    # R05 — imagem confiável, mas a intenção da pergunta é ambígua
    if p.confianca_pln < limiar_pln:
        m = materiais[p.classe]
        resumo = (f"Não entendi bem a pergunta, então aqui vai um resumo sobre **{m['nome_pt']}**:\n\n"
                  f"- **Reciclável?** {'Sim' if m['reciclavel'] else 'Não'}\n"
                  f"- **Lixeira:** {m['lixeira_cor'].upper()}\n"
                  f"- **Preparo:** {m['preparo'][0]}\n"
                  f"- **Destino:** {m['destino']}\n\n"
                  "Se quiser algo específico, pergunte por exemplo *precisa lavar antes?*")
        return Decisao("R05", f"SE confiança_pln<{limiar_pln:.2f} ENTÃO resumo geral do material",
                       resumo, p.classe, detalhes=det)

    # R10..R33 — regra específica classe × intenção
    regra = REGRAS_ESPECIFICAS[(p.classe, p.intencao)]
    return Decisao(regra.id, regra.descricao(limiar), regra.acionar(base), p.classe, detalhes=det)
