"""Fluxo integrado do EcoSort: Visão -> PLN -> Decisão (regras).

    imagem ──► predict() ──► (classe, confiança, top2) ─┐
                                                       ├─► decidir() ──► resposta + regra
    pergunta ─► interpretar() ─► (intenção, confiança) ┘
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.decisao.motor_regras import Percepcao, decidir
from src.pln.interpretar import interpretar
from src.visao_computacional.predict import EntradaImagem, predict


def processar(imagem: EntradaImagem, pergunta: str) -> dict[str, Any]:
    """Executa o fluxo completo e devolve todas as saídas intermediárias."""
    classe, conf_visao, top2 = predict(imagem)
    intencao, conf_pln = interpretar(pergunta)
    decisao = decidir(Percepcao(classe, conf_visao, top2, intencao, conf_pln))
    return {
        "pergunta": pergunta,
        "visao": {"classe": classe, "confianca": conf_visao, "top2": top2},
        "pln": {"intencao": intencao, "confianca": conf_pln},
        "decisao": asdict(decisao),
    }
