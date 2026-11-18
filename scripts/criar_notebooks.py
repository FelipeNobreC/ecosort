"""Cria os notebooks de treino e análise em notebooks/ (formato .ipynb).

Uso:
    python -m scripts.criar_notebooks
"""
from __future__ import annotations

import nbformat as nbf

from src.machine_learning.utils import RAIZ

PREAMBULO = """# Ajuste de caminho: permite rodar o notebook a partir de notebooks/ ou da raiz (Colab)
import os, sys
from pathlib import Path
raiz = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(raiz); sys.path.insert(0, str(raiz))
print("Raiz do projeto:", raiz)"""

COLAB = """# (Somente no Google Colab) clonar o repositório e instalar dependências:
# !git clone <URL_DO_REPOSITORIO> ecosort && cd ecosort
# %cd ecosort
# !pip install -q -r requirements.txt"""


def nb_treino() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# EcoSort — 01. Treino dos modelos\n\n"
                                 "Executa, em ordem: download/split do TrashNet, treino da visão (MobileNetV2) "
                                 "e treino/comparação do PLN. Em CPU o treino da visão leva ~10 min "
                                 "(16 épocas). Defina `RETREINAR=False` para só reaproveitar os modelos salvos."),
        nbf.v4.new_code_cell(COLAB),
        nbf.v4.new_code_cell(PREAMBULO + "\nRETREINAR = False"),
        nbf.v4.new_markdown_cell("## 1. Dados de imagem (TrashNet)"),
        nbf.v4.new_code_cell("from src.visao_computacional import preparar_dados\npreparar_dados.main()"),
        nbf.v4.new_markdown_cell("## 2. Visão computacional — transfer learning em 2 fases"),
        nbf.v4.new_code_cell(
            "import subprocess\nfrom src.visao_computacional.modelo import CAMINHO_MODELO\n"
            "if RETREINAR or not CAMINHO_MODELO.exists():\n"
            "    subprocess.run([sys.executable, '-m', 'src.visao_computacional.treinar'], check=True)\n"
            "else:\n    print('Modelo já existe:', CAMINHO_MODELO)"),
        nbf.v4.new_code_cell("from IPython.display import Image, display\n"
                             "display(Image('resultados/graficos/visao_curvas_treino.png'))\n"
                             "display(Image('resultados/graficos/visao_matriz_confusao.png'))"),
        nbf.v4.new_markdown_cell("## 3. PLN — TF-IDF + Regressão Logística vs. Naive Bayes"),
        nbf.v4.new_code_cell("from src.pln import treinar as treinar_pln\ntreinar_pln.main()"),
        nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('resultados/pln_comparacao_modelos.csv')"),
        nbf.v4.new_code_cell("display(Image('resultados/graficos/pln_matriz_confusao_regressao_logistica.png'))"),
    ]
    return nb


def nb_analise() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# EcoSort — 02. Análise de resultados e do fluxo integrado"),
        nbf.v4.new_code_cell(PREAMBULO),
        nbf.v4.new_markdown_cell("## Métricas da visão (conjunto de teste)"),
        nbf.v4.new_code_cell(
            "import json, pandas as pd\n"
            "mv = json.load(open('resultados/metricas_visao.json', encoding='utf-8'))\n"
            "print(f\"Acurácia: {mv['acuracia']:.4f} | F1 macro: {mv['f1_macro']:.4f}\")\n"
            "pd.DataFrame(mv['por_classe']).T.round(3)"),
        nbf.v4.new_code_cell("from IPython.display import Image, display\n"
                             "for g in ['visao_matriz_confusao_normalizada', 'visao_confianca_hist', "
                             "'limiar_cobertura', 'visao_erros_exemplos']:\n"
                             "    display(Image(f'resultados/graficos/{g}.png'))"),
        nbf.v4.new_markdown_cell("## Comparação dos modelos de PLN"),
        nbf.v4.new_code_cell("pd.read_csv('resultados/pln_comparacao_modelos.csv')"),
        nbf.v4.new_markdown_cell("## Fluxo integrado: Visão → PLN → Decisão"),
        nbf.v4.new_code_cell(
            "from src.pipeline import processar\n"
            "r = processar('dados/exemplos/glass183.jpg', 'precisa lavar antes?')\n"
            "print(r['visao']); print(r['pln']); print(r['decisao']['regra_id'], '-', r['decisao']['regra_descricao'])\n"
            "print(); print(r['decisao']['resposta'])"),
        nbf.v4.new_code_cell(
            "for pergunta in ['isso é reciclável?', 'onde posso levar?', 'qual a capital da frança?']:\n"
            "    r = processar('dados/exemplos/metal174.jpg', pergunta)\n"
            "    print(f\"{pergunta!r:32} -> {r['pln']['intencao']:15} regra {r['decisao']['regra_id']}\")"),
        nbf.v4.new_markdown_cell("## Exemplos salvos"),
        nbf.v4.new_code_cell("display(Image('resultados/exemplos/painel_exemplos.png'))"),
    ]
    return nb


def main() -> None:
    destino = RAIZ / "notebooks"
    destino.mkdir(exist_ok=True)
    nbf.write(nb_treino(), destino / "01_treino_modelos.ipynb")
    nbf.write(nb_analise(), destino / "02_analise_resultados.ipynb")
    print("Notebooks criados em", destino)


if __name__ == "__main__":
    main()
