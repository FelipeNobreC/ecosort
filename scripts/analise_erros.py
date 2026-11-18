"""Coleta dados quantitativos para a análise de erros (Visão e PLN).

Gera:
  resultados/analise_erros_dados.md            — tabelas (pares confundidos, confiança, erros de PLN)
  resultados/graficos/visao_erros_exemplos.png — grade com imagens classificadas errado
  resultados/graficos/visao_confianca_hist.png — histograma de confiança (acertos x erros)
  resultados/graficos/limiar_cobertura.png     — cobertura x acurácia em função do limiar

Uso:
    python -m scripts.analise_erros
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.machine_learning.utils import RAIZ

DIR_GRAF = RAIZ / "resultados" / "graficos"


def main() -> None:
    pred = pd.read_csv(RAIZ / "resultados" / "predicoes_teste_visao.csv")
    pred["acerto"] = pred["classe"] == pred["predito"]
    erros = pred[~pred["acerto"]]
    out: list[str] = ["# Dados da análise de erros (gerado automaticamente)\n"]

    out.append(f"Conjunto de teste: {len(pred)} imagens, {len(erros)} erros "
               f"(acurácia {pred['acerto'].mean():.4f}).\n")
    pares = erros.groupby(["classe", "predito"]).size().sort_values(ascending=False).reset_index(name="n")
    pares["% dos erros"] = (100 * pares["n"] / len(erros)).round(1)
    out.append("## Pares (real → predito) mais confundidos\n\n" + pares.to_markdown(index=False) + "\n")

    por_classe = pred.groupby("classe").agg(n=("acerto", "size"), acertos=("acerto", "sum"))
    por_classe["recall"] = (por_classe["acertos"] / por_classe["n"]).round(4)
    out.append("## Recall por classe\n\n" + por_classe.to_markdown() + "\n")

    conf = pred.groupby("acerto")["confianca"].describe()[["count", "mean", "50%", "min", "max"]].round(4)
    conf.index = ["erros" if not i else "acertos" for i in conf.index]
    out.append("## Confiança da visão: acertos x erros\n\n" + conf.to_markdown() + "\n")

    # Cobertura x acurácia por limiar (justifica o limiar de 0.6)
    linhas = []
    for lim in [0.0, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sel = pred[pred["confianca"] >= lim]
        linhas.append({"limiar": lim, "cobertura": round(len(sel) / len(pred), 4),
                       "acuracia_acima_limiar": round(sel["acerto"].mean(), 4) if len(sel) else None,
                       "erros_acima_limiar": int((~sel["acerto"]).sum())})
    tab_lim = pd.DataFrame(linhas)
    out.append("## Efeito do limiar de confiança\n\n"
               "Cobertura = fração das imagens em que o sistema responde direto (sem pedir nova foto).\n\n"
               + tab_lim.to_markdown(index=False) + "\n")

    # PLN
    met_pln = json.loads((RAIZ / "resultados" / "metricas_pln.json").read_text(encoding="utf-8"))
    for modelo, m in met_pln.items():
        e = pd.DataFrame(m["erros_teste"])
        out.append(f"## Erros de PLN no teste — {modelo} ({len(e)} erros)\n\n"
                   + (e.to_markdown(index=False) if len(e) else "Nenhum erro.") + "\n")

    (RAIZ / "resultados" / "analise_erros_dados.md").write_text("\n".join(out), encoding="utf-8")

    # ---------- Gráficos ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(pred.loc[pred.acerto, "confianca"], bins=bins, alpha=0.7, label="acertos", color="tab:green")
    ax.hist(pred.loc[~pred.acerto, "confianca"], bins=bins, alpha=0.7, label="erros", color="tab:red")
    ax.axvline(0.6, ls="--", color="k", label="limiar 0.6")
    ax.set_xlabel("Confiança (probabilidade máxima)")
    ax.set_ylabel("Nº de imagens")
    ax.set_title("Visão — distribuição de confiança no teste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIR_GRAF / "visao_confianca_hist.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tab_lim["limiar"], tab_lim["cobertura"], "o-", label="cobertura")
    ax.plot(tab_lim["limiar"], tab_lim["acuracia_acima_limiar"], "s-", label="acurácia (respostas diretas)")
    ax.axvline(0.6, ls="--", color="gray")
    ax.set_xlabel("Limiar de confiança")
    ax.set_title("Trade-off do limiar de confiança")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIR_GRAF / "limiar_cobertura.png", dpi=120)
    plt.close(fig)

    amostra = erros.sort_values("confianca", ascending=False).head(16)
    fig, axs = plt.subplots(4, 4, figsize=(13, 11))
    for ax in axs.flat:
        ax.axis("off")
    for ax, (_, l) in zip(axs.flat, amostra.iterrows()):
        ax.imshow(Image.open(RAIZ / l["caminho"]))
        ax.set_title(f"real={l['classe']}  pred={l['predito']} ({l['confianca']:.2f})", fontsize=9, color="red")
    fig.suptitle("Visão — erros no teste (maior confiança primeiro)")
    fig.tight_layout()
    fig.savefig(DIR_GRAF / "visao_erros_exemplos.png", dpi=100)
    plt.close(fig)
    print("\n".join(out))


if __name__ == "__main__":
    main()
