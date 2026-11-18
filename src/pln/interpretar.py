"""Inferência do módulo de PLN: interpretar(texto) -> (intencao, confianca)."""
from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

from src.pln.preprocessamento import preprocessar
from src.pln.treinar import CAMINHO_MODELO_PLN


@lru_cache(maxsize=1)
def carregar_modelo_pln() -> dict[str, Any]:
    """Carrega o pipeline TF-IDF + Regressão Logística salvo (cache em memória)."""
    if not CAMINHO_MODELO_PLN.exists():
        raise FileNotFoundError(f"Modelo de PLN não encontrado. Rode: python -m src.pln.treinar")
    with open(CAMINHO_MODELO_PLN, "rb") as f:
        return pickle.load(f)


def interpretar(texto: str) -> tuple[str, float]:
    """Reconhece a intenção da pergunta do usuário.

    Texto vazio (ou que vira vazio após o pré-processamento) é tratado como
    'fora_do_escopo' com confiança 1.0, sem consultar o modelo.
    """
    if not texto or not preprocessar(texto):
        return "fora_do_escopo", 1.0
    pipe = carregar_modelo_pln()["pipeline"]
    probs = pipe.predict_proba([texto])[0]
    idx = int(probs.argmax())
    return str(pipe.classes_[idx]), float(probs[idx])
