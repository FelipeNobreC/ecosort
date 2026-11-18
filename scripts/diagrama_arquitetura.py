"""Desenha o diagrama de arquitetura do EcoSort (PNG) e copia as figuras usadas no artigo.

Uso:
    python -m scripts.diagrama_arquitetura
"""
from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.machine_learning.utils import RAIZ

FIGURAS_ARTIGO = [
    "dados_distribuicao_classes.png",
    "visao_curvas_treino.png",
    "visao_matriz_confusao_normalizada.png",
    "limiar_cobertura.png",
]


def caixa(ax, x: float, y: float, w: float, h: float, texto: str, cor: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=cor, ec="#333333", lw=1.0))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=8.5)


def seta(ax, a: tuple[float, float], b: tuple[float, float], rotulo: str = "") -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#333333"))
    if rotulo:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, rotulo, ha="center", fontsize=7, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    caixa(ax, 0.1, 2.5, 1.3, 0.9, "Foto do\nresíduo", "#eeeeee")
    caixa(ax, 0.1, 0.6, 1.3, 0.9, "Pergunta\n(PT-BR)", "#eeeeee")
    caixa(ax, 2.0, 2.4, 2.4, 1.1, "Visão Computacional\nResize 224, norm. ImageNet\nMobileNetV2 (transfer learning)", "#c8e6c9")
    caixa(ax, 2.0, 0.5, 2.4, 1.1, "PLN\nnormalização + stopwords\nTF-IDF (1-2) + Reg. Logística", "#bbdefb")
    caixa(ax, 5.2, 1.3, 2.2, 1.4, "Decisão\nagente baseado em regras\nR01–R05, R10–R33", "#ffe0b2")
    caixa(ax, 5.35, 0.05, 1.9, 0.75, "Base de conhecimento\nregras_descarte.json", "#fff3e0")
    caixa(ax, 8.1, 1.5, 1.8, 1.0, "Interface\nStreamlit\n(resposta + regra)", "#e1bee7")
    seta(ax, (1.4, 2.95), (2.0, 2.95))
    seta(ax, (1.4, 1.05), (2.0, 1.05))
    seta(ax, (4.4, 2.95), (5.2, 2.4))
    ax.text(4.8, 3.15, "classe,\nconfiança, top-2", ha="center", fontsize=7, style="italic")
    seta(ax, (4.4, 1.05), (5.2, 1.6))
    ax.text(4.8, 0.6, "intenção,\nconfiança", ha="center", fontsize=7, style="italic")
    seta(ax, (6.3, 0.8), (6.3, 1.3))
    seta(ax, (7.4, 2.0), (8.1, 2.0))
    fig.tight_layout()
    destino = RAIZ / "resultados" / "graficos" / "arquitetura.png"
    fig.savefig(destino, dpi=200)
    plt.close(fig)

    pasta = RAIZ / "artigo" / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)
    for nome in FIGURAS_ARTIGO + ["arquitetura.png"]:
        shutil.copy(RAIZ / "resultados" / "graficos" / nome, pasta / nome)
    print("Figuras copiadas para", pasta)


if __name__ == "__main__":
    main()
