"""Inferência do módulo de Visão: predict(imagem) -> (classe, confianca, top2)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from src.visao_computacional.modelo import CAMINHO_MODELO, CLASSES, criar_modelo, transform_avaliacao

EntradaImagem = Union[str, Path, Image.Image]


@lru_cache(maxsize=1)
def carregar_modelo(caminho: str = str(CAMINHO_MODELO)) -> torch.nn.Module:
    """Carrega o modelo treinado uma única vez (cache em memória)."""
    if not Path(caminho).exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}. Rode: python -m src.visao_computacional.treinar"
        )
    ckpt = torch.load(caminho, map_location="cpu", weights_only=False)
    modelo = criar_modelo(pretreinado=False, n_classes=len(ckpt["classes"]))
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo


def probabilidades(imagem: EntradaImagem) -> dict[str, float]:
    """Retorna a distribuição de probabilidade (softmax) sobre as 6 classes."""
    if not isinstance(imagem, Image.Image):
        imagem = Image.open(imagem)
    x = transform_avaliacao()(imagem.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(carregar_modelo()(x), dim=1)[0].tolist()
    return dict(zip(CLASSES, p))


def predict(imagem: EntradaImagem) -> tuple[str, float, list[tuple[str, float]]]:
    """Classifica o resíduo da imagem.

    Retorna (classe_mais_provável, confiança, [(classe1, p1), (classe2, p2)]).
    """
    probs = probabilidades(imagem)
    ordenado = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    classe, conf = ordenado[0]
    return classe, float(conf), [(c, float(p)) for c, p in ordenado[:2]]
