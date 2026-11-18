"""Utilitários compartilhados de Machine Learning do EcoSort.

Reúne funções usadas tanto pelo módulo de Visão quanto pelo de PLN:
- fixação de seeds (reprodutibilidade);
- divisão estratificada treino/validação/teste;
- cálculo de métricas de classificação;
- geração de gráficos (matriz de confusão e curvas de treino).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (funciona em servidor/CPU/Colab)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# Raiz do repositório (…/ecosort)
RAIZ: Path = Path(__file__).resolve().parents[2]
SEED_PADRAO: int = 42


def fixar_seeds(seed: int = SEED_PADRAO) -> None:
    """Fixa as seeds de random, numpy e (se disponível) torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch é opcional para o módulo de PLN
        pass


def dividir_estratificado(
    itens: Sequence[Any],
    rotulos: Sequence[Any],
    prop_treino: float = 0.70,
    prop_val: float = 0.15,
    seed: int = SEED_PADRAO,
) -> dict[str, tuple[list[Any], list[Any]]]:
    """Divide (itens, rótulos) em treino/validação/teste de forma estratificada.

    A proporção de teste é o restante (1 - treino - validação).
    Retorna um dicionário {"treino": (X, y), "validacao": (X, y), "teste": (X, y)}.
    """
    prop_teste = 1.0 - prop_treino - prop_val
    x_treino, x_resto, y_treino, y_resto = train_test_split(
        list(itens), list(rotulos), train_size=prop_treino, stratify=rotulos, random_state=seed
    )
    # Do restante, separa validação e teste mantendo a proporção relativa
    frac_teste = prop_teste / (prop_val + prop_teste)
    x_val, x_teste, y_val, y_teste = train_test_split(
        x_resto, y_resto, test_size=frac_teste, stratify=y_resto, random_state=seed
    )
    return {
        "treino": (x_treino, y_treino),
        "validacao": (x_val, y_val),
        "teste": (x_teste, y_teste),
    }


def calcular_metricas(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """Calcula acurácia, F1 macro e o relatório por classe (precision/recall/F1)."""
    relatorio = classification_report(
        y_true, y_pred, labels=list(classes), output_dict=True, zero_division=0
    )
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)),
        "por_classe": {c: relatorio[c] for c in classes},
        "matriz_confusao": confusion_matrix(y_true, y_pred, labels=list(classes)).tolist(),
    }


def salvar_json(dados: dict[str, Any], caminho: Path) -> None:
    """Salva um dicionário em JSON (UTF-8, indentado)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")


def plotar_matriz_confusao(
    matriz: Sequence[Sequence[int]],
    classes: Sequence[str],
    titulo: str,
    caminho: Path,
    normalizar: bool = False,
) -> None:
    """Gera e salva o PNG de uma matriz de confusão anotada."""
    m = np.asarray(matriz, dtype=float)
    if normalizar:
        m = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 1.0 * len(classes) + 2))
    im = ax.imshow(m, cmap="Greens")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    limiar = m.max() / 2 if m.max() > 0 else 0.5
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            texto = f"{m[i, j]:.2f}" if normalizar else f"{int(m[i, j])}"
            ax.text(j, i, texto, ha="center", va="center",
                    color="white" if m[i, j] > limiar else "black", fontsize=9)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def plotar_curvas_treino(historico: dict[str, list[float]], caminho: Path, marco: int | None = None) -> None:
    """Plota curvas de loss e acurácia (treino x validação) por época.

    `marco` indica a época em que começou o fine-tuning (linha vertical tracejada).
    """
    epocas = range(1, len(historico["loss_treino"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epocas, historico["loss_treino"], "o-", label="treino")
    ax1.plot(epocas, historico["loss_val"], "s-", label="validação")
    ax1.set_title("Loss por época")
    ax1.set_xlabel("Época")
    ax2.plot(epocas, historico["acc_treino"], "o-", label="treino")
    ax2.plot(epocas, historico["acc_val"], "s-", label="validação")
    ax2.set_title("Acurácia por época")
    ax2.set_xlabel("Época")
    for ax in (ax1, ax2):
        if marco is not None:
            ax.axvline(marco + 0.5, color="gray", ls="--", label="início fine-tuning")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
