"""Gera exemplos de entrada/saída do fluxo completo em resultados/exemplos/.

Seleciona imagens do CONJUNTO DE TESTE (nunca vistas no treino) cobrindo:
acertos com alta confiança, erros da visão, baixa confiança (R02/R03/R04),
perguntas fora do escopo, com erros de digitação e ambíguas.

Uso:
    python -m scripts.gerar_exemplos
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.decisao.motor_regras import carregar_base
from src.machine_learning.utils import RAIZ
from src.pipeline import processar

DIR = RAIZ / "resultados" / "exemplos"
LIMIAR = carregar_base()["_meta"]["limiar_confianca_visao"]
LIMIAR_MIN = carregar_base()["_meta"]["limiar_confianca_minima_visao"]


def selecionar_casos(pred: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Retorna lista (caminho_imagem, pergunta, descrição do caso)."""
    pred = pred.copy()
    pred["acerto"] = pred["classe"] == pred["predito"]
    # 2ª classe mais provável (para separar R03 x R04)
    cols = [c for c in pred.columns if c.startswith("p_")]
    pred["segunda"] = pred[cols].apply(lambda r: r.sort_values(ascending=False).index[1][2:], axis=1)
    azul = {"paper", "cardboard"}
    casos: list[tuple[str, str, str]] = []
    perguntas_acerto = {
        "cardboard": ("como descarto isso?", "Acerto, alta confiança — como descartar"),
        "glass": ("precisa lavar antes?", "Acerto, alta confiança — preparo"),
        "metal": ("onde posso levar?", "Acerto, alta confiança — onde levar"),
        "plastic": ("isso é reciclável?", "Acerto, alta confiança — é reciclável"),
        "trash": ("da pra reciclar isso?", "Acerto, classe rejeito (não reciclável)"),
        "paper": ("presisa lava antes?", "Pergunta com erros de digitação"),
    }
    for classe, (perg, desc) in perguntas_acerto.items():
        linha = pred[(pred.classe == classe) & pred.acerto & (pred.confianca >= 0.8)].sort_values("confianca").iloc[-1]
        casos.append((linha.caminho, perg, desc))
    # Erros da visão com confiança acima do limiar (erro "confiante" — o pior caso)
    erros_conf = pred[~pred.acerto & (pred.confianca >= LIMIAR)].sort_values("confianca", ascending=False)
    for _, l in erros_conf.head(2).iterrows():
        casos.append((l.caminho, "como descarto isso?", f"ERRO da visão com confiança alta ({l.classe}→{l.predito})"))
    # Baixa confiança: R04 (lixeiras diferentes), R03 (papel x papelão) e R02 (muito baixa)
    medio = pred[(pred.confianca >= LIMIAR_MIN) & (pred.confianca < LIMIAR)]
    r04 = medio[~(medio.predito.isin(azul) & medio.segunda.isin(azul))]
    r03 = medio[medio.predito.isin(azul) & medio.segunda.isin(azul)]
    if len(r04):
        casos.append((r04.iloc[0].caminho, "isso é reciclável?", "Baixa confiança, lixeiras diferentes (top-2)"))
    if len(r03):
        casos.append((r03.iloc[0].caminho, "como descarto isso?", "Baixa confiança, mas top-2 com mesma lixeira"))
    muito_baixa = pred[pred.confianca < LIMIAR_MIN]
    if len(muito_baixa):
        casos.append((muito_baixa.iloc[0].caminho, "como descarto isso?", "Confiança muito baixa — pede nova foto"))
    # Casos de PLN
    base = pred[pred.acerto & (pred.confianca >= 0.8)]
    casos.append((base.iloc[3].caminho, "qual a previsão do tempo hoje?", "Pergunta fora do escopo"))
    casos.append((base.iloc[7].caminho, "mano essa parada vai em qual lixo?", "Gíria/informal"))
    casos.append((base.iloc[11].caminho, "me explica esse item", "Pergunta ambígua (intenção incerta)"))
    return casos


def main() -> None:
    pred = pd.read_csv(RAIZ / "resultados" / "predicoes_teste_visao.csv")
    casos = selecionar_casos(pred)
    dir_img = DIR / "imagens"
    if dir_img.exists():
        shutil.rmtree(dir_img)
    dir_img.mkdir(parents=True)

    registros, linhas_md = [], []
    for n, (caminho, pergunta, desc) in enumerate(casos, 1):
        real = pred.loc[pred.caminho == caminho, "classe"].iloc[0]
        r = processar(RAIZ / caminho, pergunta)
        nome_img = f"{n:02d}_{Path(caminho).name}"
        shutil.copy(RAIZ / caminho, dir_img / nome_img)
        r.update({"caso": n, "descricao": desc, "imagem": f"imagens/{nome_img}", "classe_real": real})
        registros.append(r)
        v, p, d = r["visao"], r["pln"], r["decisao"]
        top2 = ", ".join(f"{c} {pr:.2f}" for c, pr in v["top2"])
        linhas_md.append(
            f"### Caso {n:02d} — {desc}\n\n"
            f"![]({r['imagem']})\n\n"
            f"| Campo | Valor |\n|---|---|\n"
            f"| Classe real | `{real}` |\n| Pergunta | \"{pergunta}\" |\n"
            f"| Visão | `{v['classe']}` (conf. {v['confianca']:.3f}) — top-2: {top2} |\n"
            f"| PLN | `{p['intencao']}` (conf. {p['confianca']:.3f}) |\n"
            f"| Regra acionada | **{d['regra_id']}** — {d['regra_descricao']} |\n"
            f"| Visão acertou? | {'sim' if v['classe'] == real else '**NÃO**'} |\n\n"
            f"**Resposta:**\n\n> " + d["resposta"].replace("\n", "\n> ") + "\n"
        )

    (DIR / "exemplos.json").write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")
    (DIR / "exemplos.md").write_text(
        "# Exemplos de entrada/saída do fluxo completo\n\n"
        "Gerado automaticamente por `python -m scripts.gerar_exemplos` com imagens do conjunto de TESTE.\n\n"
        + "\n".join(linhas_md), encoding="utf-8")

    # Painel visual com todos os casos
    cols = 4
    linhas = (len(registros) + cols - 1) // cols
    fig, axs = plt.subplots(linhas, cols, figsize=(4 * cols, 3.6 * linhas))
    for ax in axs.flat:
        ax.axis("off")
    for ax, r in zip(axs.flat, registros):
        ax.imshow(Image.open(DIR / r["imagem"]))
        ok = r["visao"]["classe"] == r["classe_real"]
        ax.set_title(f"#{r['caso']:02d} real={r['classe_real']} pred={r['visao']['classe']} "
                     f"({r['visao']['confianca']:.2f})\n\"{r['pergunta'][:30]}\" → {r['pln']['intencao']}\n"
                     f"regra {r['decisao']['regra_id']}", fontsize=8, color="green" if ok else "red")
    fig.tight_layout()
    fig.savefig(DIR / "painel_exemplos.png", dpi=110)
    plt.close(fig)
    print(f"{len(registros)} exemplos salvos em {DIR}")
    for r in registros:
        print(f"#{r['caso']:02d} {r['descricao'][:45]:45s} real={r['classe_real']:9s} "
              f"pred={r['visao']['classe']:9s} {r['visao']['confianca']:.2f} "
              f"int={r['pln']['intencao']:15s} {r['pln']['confianca']:.2f} {r['decisao']['regra_id']}")


if __name__ == "__main__":
    main()
