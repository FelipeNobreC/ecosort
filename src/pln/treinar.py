"""Treino e comparação dos modelos de intenção: TF-IDF + Regressão Logística vs. Naive Bayes.

Protocolo:
  1. Split estratificado 70/15/15 (seed fixa) de dados/intencoes.csv.
  2. Hiperparâmetro (C da RegLog / alpha do NB) escolhido pelo F1 macro na VALIDAÇÃO.
  3. Avaliação final no TESTE (acurácia, F1 macro, matriz de confusão).
  4. Validação cruzada estratificada 5-fold no dataset inteiro (robustez da comparação).
  5. O modelo vencedor é re-treinado em treino+validação e salvo em modelos/.

Uso:
    python -m src.pln.treinar
"""
from __future__ import annotations

import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.machine_learning.utils import (
    RAIZ,
    SEED_PADRAO,
    calcular_metricas,
    dividir_estratificado,
    fixar_seeds,
    plotar_matriz_confusao,
    salvar_json,
)
from src.pln.preprocessamento import preprocessar

INTENCOES: list[str] = ["como_descartar", "e_reciclavel", "onde_levar", "preparo_antes", "fora_do_escopo"]
CAMINHO_MODELO_PLN = RAIZ / "modelos" / "pln_intencoes.pkl"
DIR_GRAF = RAIZ / "resultados" / "graficos"


def criar_pipeline(tipo: str, param: float) -> Pipeline:
    """Pipeline TF-IDF (unigramas + bigramas) + classificador."""
    vetorizador = TfidfVectorizer(preprocessor=preprocessar, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    if tipo == "regressao_logistica":
        clf = LogisticRegression(C=param, max_iter=2000, random_state=SEED_PADRAO)
    else:
        clf = MultinomialNB(alpha=param)
    return Pipeline([("tfidf", vetorizador), ("clf", clf)])


GRADES: dict[str, list[float]] = {
    "regressao_logistica": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
    "naive_bayes": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
}


def main() -> None:
    fixar_seeds()
    df = pd.read_csv(RAIZ / "dados" / "intencoes.csv")
    partes = dividir_estratificado(df["texto"].tolist(), df["intencao"].tolist(), 0.70, 0.15, SEED_PADRAO)
    (xt, yt), (xv, yv), (xs, ys) = partes["treino"], partes["validacao"], partes["teste"]
    print(f"Frases: total={len(df)} treino={len(xt)} validação={len(xv)} teste={len(xs)}")

    resultados: dict[str, dict] = {}
    linhas_tabela: list[dict] = []
    for tipo, grade in GRADES.items():
        # --- Seleção de hiperparâmetro na validação ---
        f1s_val = {p: f1_score(yv, criar_pipeline(tipo, p).fit(xt, yt).predict(xv), average="macro") for p in grade}
        melhor_p = max(f1s_val, key=f1s_val.get)
        # --- Avaliação no teste (treinado só no treino) ---
        modelo = criar_pipeline(tipo, melhor_p).fit(xt, yt)
        met = calcular_metricas(ys, modelo.predict(xs), INTENCOES)
        # --- Validação cruzada 5-fold no dataset inteiro ---
        cv = cross_validate(criar_pipeline(tipo, melhor_p), df["texto"], df["intencao"],
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED_PADRAO),
                            scoring=["accuracy", "f1_macro"])
        met.update({
            "hiperparametro": melhor_p,
            "f1_val_por_hiperparametro": f1s_val,
            "cv5_acuracia_media": float(cv["test_accuracy"].mean()),
            "cv5_acuracia_desvio": float(cv["test_accuracy"].std()),
            "cv5_f1_macro_media": float(cv["test_f1_macro"].mean()),
            "cv5_f1_macro_desvio": float(cv["test_f1_macro"].std()),
        })
        resultados[tipo] = met
        plotar_matriz_confusao(met["matriz_confusao"], INTENCOES, f"PLN — {tipo} (teste)",
                               DIR_GRAF / f"pln_matriz_confusao_{tipo}.png")
        linhas_tabela.append({
            "modelo": tipo, "hiperparametro": melhor_p,
            "acuracia_teste": round(met["acuracia"], 4), "f1_macro_teste": round(met["f1_macro"], 4),
            "cv5_acuracia": f"{met['cv5_acuracia_media']:.4f} ± {met['cv5_acuracia_desvio']:.4f}",
            "cv5_f1_macro": f"{met['cv5_f1_macro_media']:.4f} ± {met['cv5_f1_macro_desvio']:.4f}",
        })
        # Erros no teste (para análise)
        erros = [(x, y, p) for x, y, p in zip(xs, ys, modelo.predict(xs)) if y != p]
        met["erros_teste"] = [{"texto": x, "real": y, "predito": p} for x, y, p in erros]

    tabela = pd.DataFrame(linhas_tabela)
    tabela.to_csv(RAIZ / "resultados" / "pln_comparacao_modelos.csv", index=False)
    (RAIZ / "resultados" / "pln_comparacao_modelos.md").write_text(tabela.to_markdown(index=False), encoding="utf-8")
    salvar_json(resultados, RAIZ / "resultados" / "metricas_pln.json")
    print(tabela.to_string(index=False))

    # --- Modelo final: Regressão Logística (modelo principal exigido), re-treinada em treino+validação ---
    p_final = resultados["regressao_logistica"]["hiperparametro"]
    final = criar_pipeline("regressao_logistica", p_final).fit(xt + xv, yt + yv)
    CAMINHO_MODELO_PLN.parent.mkdir(parents=True, exist_ok=True)
    with open(CAMINHO_MODELO_PLN, "wb") as f:
        pickle.dump({"pipeline": final, "intencoes": INTENCOES, "C": p_final}, f)
    print(f"\nModelo final (RegLog, C={p_final}) salvo em {CAMINHO_MODELO_PLN}")


if __name__ == "__main__":
    main()
