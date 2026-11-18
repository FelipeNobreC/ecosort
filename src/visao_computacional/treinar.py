"""Treino e avaliação do classificador de resíduos (MobileNetV2, transfer learning).

Estratégia em duas fases (viável em CPU):
  1. Base congelada, treina só a cabeça (lr alto, poucas épocas).
  2. Fine-tuning: descongela os últimos blocos convolucionais (lr baixo).
O melhor checkpoint (maior acurácia de validação) é salvo e avaliado no TESTE.

Uso:
    python -m src.visao_computacional.treinar [--epocas-cabeca 6] [--epocas-ft 10]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.machine_learning.utils import (
    RAIZ,
    SEED_PADRAO,
    calcular_metricas,
    fixar_seeds,
    plotar_curvas_treino,
    plotar_matriz_confusao,
    salvar_json,
)
from src.visao_computacional.modelo import (
    CAMINHO_MODELO,
    CLASSES,
    DatasetResiduos,
    congelar_base,
    criar_modelo,
    descongelar_ultimas,
    transform_avaliacao,
    transform_treino,
)

DIR_GRAF = RAIZ / "resultados" / "graficos"


def carregar_split() -> dict[str, pd.DataFrame]:
    """Lê dados/split_imagens.csv (gerado por preparar_dados.py)."""
    df = pd.read_csv(RAIZ / "dados" / "split_imagens.csv")
    return {nome: df[df["conjunto"] == nome].reset_index(drop=True) for nome in ("treino", "validacao", "teste")}


def rodar_epoca(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer | None,
    base_congelada: bool = False,
) -> tuple[float, float]:
    """Executa uma época de treino (se houver otimizador) ou de avaliação."""
    treinando = otimizador is not None
    modelo.train(treinando)
    if treinando and base_congelada:
        modelo.features.eval()  # mantém estatísticas do BatchNorm do ImageNet
    perda_total, acertos, n = 0.0, 0, 0
    with torch.set_grad_enabled(treinando):
        for x, y in loader:
            saida = modelo(x)
            perda = criterio(saida, y)
            if treinando:
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
            perda_total += perda.item() * len(y)
            acertos += (saida.argmax(1) == y).sum().item()
            n += len(y)
    return perda_total / n, acertos / n


@torch.no_grad()
def prever_loader(modelo: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (probabilidades, rótulos verdadeiros) para todo o loader."""
    modelo.eval()
    probs, alvos = [], []
    for x, y in loader:
        probs.append(torch.softmax(modelo(x), 1).numpy())
        alvos.append(y.numpy())
    return np.concatenate(probs), np.concatenate(alvos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas-cabeca", type=int, default=6)
    ap.add_argument("--epocas-ft", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    fixar_seeds(SEED_PADRAO)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    split = carregar_split()
    g = torch.Generator().manual_seed(SEED_PADRAO)

    def loader(nome: str, treino: bool) -> DataLoader:
        df = split[nome]
        ds = DatasetResiduos(df["caminho"].tolist(), df["classe"].tolist(),
                             transform_treino() if treino else transform_avaliacao())
        return DataLoader(ds, batch_size=args.batch, shuffle=treino, generator=g if treino else None,
                          num_workers=args.workers, persistent_workers=args.workers > 0)

    dl_treino, dl_val, dl_teste = loader("treino", True), loader("validacao", False), loader("teste", False)

    # Pesos por classe (inverso da frequência) para compensar a classe "trash" (minoritária)
    contagem = split["treino"]["classe"].value_counts().reindex(CLASSES).to_numpy()
    pesos = torch.tensor(contagem.sum() / (len(CLASSES) * contagem), dtype=torch.float32)
    criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.1)

    modelo = criar_modelo(pretreinado=True)
    historico: dict[str, list[float]] = {"loss_treino": [], "acc_treino": [], "loss_val": [], "acc_val": []}
    melhor_acc, melhor_estado = -1.0, None

    def registrar(fase: str, ep: int, tr: tuple[float, float], va: tuple[float, float], t0: float) -> None:
        nonlocal melhor_acc, melhor_estado
        for k, v in zip(historico, (*tr, *va)):
            historico[k].append(v)
        print(f"[{fase}] época {ep:02d} | loss {tr[0]:.4f} acc {tr[1]:.4f} | "
              f"val_loss {va[0]:.4f} val_acc {va[1]:.4f} | {time.time() - t0:.0f}s", flush=True)
        if va[1] > melhor_acc:
            melhor_acc = va[1]
            melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}

    # ---------- Fase 1: só a cabeça ----------
    congelar_base(modelo)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, modelo.parameters()), lr=1e-3, weight_decay=1e-4)
    for ep in range(1, args.epocas_cabeca + 1):
        t0 = time.time()
        tr = rodar_epoca(modelo, dl_treino, criterio, opt, base_congelada=True)
        registrar("cabeça", ep, tr, rodar_epoca(modelo, dl_val, criterio, None), t0)

    # ---------- Fase 2: fine-tuning dos últimos blocos ----------
    descongelar_ultimas(modelo, n_blocos=5)
    opt = torch.optim.AdamW([
        {"params": [p for p in modelo.features.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": modelo.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epocas_ft)
    for ep in range(1, args.epocas_ft + 1):
        t0 = time.time()
        tr = rodar_epoca(modelo, dl_treino, criterio, opt)
        sched.step()
        registrar("fine-tuning", args.epocas_cabeca + ep, tr, rodar_epoca(modelo, dl_val, criterio, None), t0)

    # ---------- Salvar melhor modelo ----------
    modelo.load_state_dict(melhor_estado)
    CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": modelo.state_dict(), "classes": CLASSES, "arquitetura": "mobilenet_v2",
                "melhor_acc_val": melhor_acc}, CAMINHO_MODELO)
    print(f"Modelo salvo em {CAMINHO_MODELO} (melhor val_acc={melhor_acc:.4f})")

    # ---------- Avaliação no teste ----------
    probs, alvos = prever_loader(modelo, dl_teste)
    y_true = [CLASSES[i] for i in alvos]
    y_pred = [CLASSES[i] for i in probs.argmax(1)]
    metricas = calcular_metricas(y_true, y_pred, CLASSES)
    metricas["melhor_acc_validacao"] = melhor_acc
    metricas["historico"] = historico
    metricas["hiperparametros"] = vars(args) | {"lr_cabeca": 1e-3, "lr_ft_base": 1e-4, "blocos_ft": 5}
    salvar_json(metricas, RAIZ / "resultados" / "metricas_visao.json")

    # Predições do teste (usadas na análise de erros)
    df_teste = split["teste"].copy()
    df_teste["predito"] = y_pred
    df_teste["confianca"] = probs.max(1)
    for i, c in enumerate(CLASSES):
        df_teste[f"p_{c}"] = probs[:, i]
    df_teste.to_csv(RAIZ / "resultados" / "predicoes_teste_visao.csv", index=False)

    plotar_matriz_confusao(metricas["matriz_confusao"], CLASSES, "Visão — matriz de confusão (teste)",
                           DIR_GRAF / "visao_matriz_confusao.png")
    plotar_matriz_confusao(metricas["matriz_confusao"], CLASSES, "Visão — matriz normalizada (teste)",
                           DIR_GRAF / "visao_matriz_confusao_normalizada.png", normalizar=True)
    plotar_curvas_treino(historico, DIR_GRAF / "visao_curvas_treino.png", marco=args.epocas_cabeca)
    print(f"\nTESTE: acurácia={metricas['acuracia']:.4f}  F1 macro={metricas['f1_macro']:.4f}")
    for c in CLASSES:
        m = metricas["por_classe"][c]
        print(f"  {c:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f} n={int(m['support'])}")


if __name__ == "__main__":
    main()
