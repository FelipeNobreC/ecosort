"""Testes automatizados do EcoSort (pytest).

Executar na raiz do projeto:
    python -m pytest testes -v
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.decisao.motor_regras import (
    CLASSES,
    INTENCOES,
    INTENCOES_DOMINIO,
    REGRAS_ESPECIFICAS,
    Percepcao,
    carregar_base,
    decidir,
)
from src.machine_learning.utils import RAIZ
from src.pipeline import processar
from src.pln.interpretar import interpretar
from src.pln.preprocessamento import preprocessar
from src.visao_computacional.predict import predict

DIR_EXEMPLOS = RAIZ / "dados" / "exemplos"
IMAGENS = sorted(DIR_EXEMPLOS.glob("*.jpg"))


# ----------------------------- Visão -----------------------------
@pytest.mark.parametrize("img", IMAGENS, ids=[p.name for p in IMAGENS])
def test_predict_retorna_classe_valida(img: Path) -> None:
    classe, conf, top2 = predict(img)
    assert classe in CLASSES
    assert 0.0 <= conf <= 1.0
    assert len(top2) == 2 and top2[0][0] == classe and top2[0][1] >= top2[1][1]


def test_predict_aceita_pil_e_imagem_qualquer_tamanho() -> None:
    img = Image.new("RGB", (640, 120), color=(120, 90, 40))
    classe, conf, _ = predict(img)
    assert classe in CLASSES and 0.0 <= conf <= 1.0


def test_existem_imagens_de_exemplo() -> None:
    assert len(IMAGENS) >= 6, "dados/exemplos/ deve ter ao menos uma imagem por classe"


# ----------------------------- PLN -----------------------------
@pytest.mark.parametrize("texto", [
    "como descarto isso?", "isso é reciclável?", "onde posso levar?", "precisa lavar antes?",
    "qual a previsão do tempo?", "", "!!!", "RECICLA ISSO AÍ???",
])
def test_interpretar_retorna_intencao_valida(texto: str) -> None:
    intencao, conf = interpretar(texto)
    assert intencao in INTENCOES
    assert 0.0 <= conf <= 1.0


@pytest.mark.parametrize("texto,esperado", [
    ("como descarto isso?", "como_descartar"),
    ("isso é reciclável?", "e_reciclavel"),
    ("onde posso levar?", "onde_levar"),
    ("precisa lavar antes?", "preparo_antes"),
])
def test_interpretar_exemplos_do_enunciado(texto: str, esperado: str) -> None:
    assert interpretar(texto)[0] == esperado


def test_preprocessamento() -> None:
    assert preprocessar("Isso é RECICLÁVEL???") == "reciclavel"
    assert "onde" in preprocessar("Onde posso levar?")  # 'onde' não é stopword (pista de intenção)


# ----------------------------- Decisão -----------------------------
def test_base_de_conhecimento_completa() -> None:
    base = carregar_base()
    cores = {"cardboard": "azul", "paper": "azul", "plastic": "vermelha",
             "glass": "verde", "metal": "amarela", "trash": "cinza"}
    for c in CLASSES:
        m = base["materiais"][c]
        assert m["lixeira_cor"] == cores[c]
        for chave in ("reciclavel", "como_descartar", "preparo", "destino", "nome_pt"):
            assert chave in m
    assert base["materiais"]["trash"]["reciclavel"] is False


def test_regras_cobrem_todas_combinacoes_classe_intencao() -> None:
    """Toda combinação classe × intenção (alta confiança) aciona uma regra e gera resposta."""
    assert len(REGRAS_ESPECIFICAS) == len(CLASSES) * len(INTENCOES_DOMINIO)
    ids_vistos = set()
    for c in CLASSES:
        outra = next(x for x in CLASSES if x != c)
        for i in INTENCOES:
            d = decidir(Percepcao(c, 0.95, [(c, 0.95), (outra, 0.03)], i, 0.9))
            assert d.resposta.strip()
            if i == "fora_do_escopo":
                assert d.regra_id == "R01"
            else:
                assert d.regra_id == REGRAS_ESPECIFICAS[(c, i)].id
                ids_vistos.add(d.regra_id)
    assert len(ids_vistos) == 24  # cada regra específica é distinta


def test_baixa_confianca_pede_nova_foto() -> None:
    d = decidir(Percepcao("glass", 0.20, [("glass", 0.20), ("plastic", 0.18)], "como_descartar", 0.9))
    assert d.regra_id == "R02" and d.precisa_nova_foto


def test_confianca_media_mostra_top2() -> None:
    d = decidir(Percepcao("glass", 0.50, [("glass", 0.50), ("plastic", 0.40)], "como_descartar", 0.9))
    assert d.regra_id == "R04" and d.precisa_nova_foto
    assert "vidro" in d.resposta.lower() and "plástico" in d.resposta.lower()


def test_confianca_media_mesma_lixeira_responde() -> None:
    d = decidir(Percepcao("paper", 0.50, [("paper", 0.50), ("cardboard", 0.45)], "como_descartar", 0.9))
    assert d.regra_id == "R03" and not d.precisa_nova_foto and "AZUL" in d.resposta


def test_intencao_ambigua_gera_resumo() -> None:
    d = decidir(Percepcao("metal", 0.9, [("metal", 0.9), ("glass", 0.05)], "onde_levar", 0.25))
    assert d.regra_id == "R05"


def test_fora_do_escopo_tem_prioridade() -> None:
    d = decidir(Percepcao("metal", 0.10, [("metal", 0.10), ("glass", 0.09)], "fora_do_escopo", 0.9))
    assert d.regra_id == "R01"


def test_fora_do_escopo_incerto_vira_resumo() -> None:
    d = decidir(Percepcao("metal", 0.9, [("metal", 0.9), ("glass", 0.05)], "fora_do_escopo", 0.3))
    assert d.regra_id == "R05"


# ----------------------------- Fluxo integrado -----------------------------
def test_fluxo_ponta_a_ponta() -> None:
    img = next(p for p in IMAGENS if p.name.startswith("glass"))
    r = processar(img, "isso é reciclável?")
    assert r["visao"]["classe"] in CLASSES
    assert r["pln"]["intencao"] == "e_reciclavel"
    assert r["decisao"]["regra_id"].startswith("R")
    assert r["decisao"]["resposta"].strip()


def test_fluxo_ponta_a_ponta_fora_do_escopo() -> None:
    r = processar(IMAGENS[0], "qual a capital da frança?")
    assert r["pln"]["intencao"] == "fora_do_escopo" and r["decisao"]["regra_id"] == "R01"
