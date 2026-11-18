"""Baixa, organiza e divide o dataset TrashNet (6 classes).

Uso:
    python -m src.visao_computacional.preparar_dados

Saídas:
    dados/bruto/trashnet/<classe>/*.jpg   (imagens originais 512x384, não versionadas)
    dados/split_imagens.csv               (caminho, classe, conjunto) — split 70/15/15 estratificado
"""
from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.machine_learning.utils import RAIZ, SEED_PADRAO, dividir_estratificado

CLASSES: list[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Fontes em ordem de preferência (se a primeira falhar, tenta a próxima)
FONTES: list[str] = [
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
]

DIR_BRUTO: Path = RAIZ / "dados" / "bruto" / "trashnet"
CSV_SPLIT: Path = RAIZ / "dados" / "split_imagens.csv"


def baixar_zip() -> bytes:
    """Tenta baixar o zip do TrashNet de cada fonte até conseguir."""
    erros: list[str] = []
    for url in FONTES:
        try:
            print(f"Baixando {url} ...")
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            if len(resp.content) < 10_000_000:  # proteção contra páginas de erro/LFS pointer
                raise ValueError(f"arquivo pequeno demais ({len(resp.content)} bytes)")
            return resp.content
        except Exception as exc:  # noqa: BLE001 - queremos tentar a próxima fonte
            erros.append(f"{url}: {exc}")
            print(f"  falhou: {exc}")
    raise RuntimeError("Nenhuma fonte do TrashNet funcionou:\n" + "\n".join(erros))


def extrair(conteudo: bytes) -> None:
    """Extrai apenas os .jpg de cada classe para dados/bruto/trashnet/<classe>/."""
    if DIR_BRUTO.exists():
        shutil.rmtree(DIR_BRUTO)
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        for nome in zf.namelist():
            partes = Path(nome).parts
            if "__MACOSX" in partes or not nome.lower().endswith(".jpg"):
                continue
            classe = partes[-2]
            if classe not in CLASSES:
                continue
            destino = DIR_BRUTO / classe / partes[-1]
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(zf.read(nome))


def gerar_split(seed: int = SEED_PADRAO) -> pd.DataFrame:
    """Cria o CSV com a divisão estratificada 70/15/15 (caminhos relativos à raiz)."""
    caminhos: list[str] = []
    rotulos: list[str] = []
    for classe in CLASSES:
        for img in sorted((DIR_BRUTO / classe).glob("*.jpg")):
            caminhos.append(img.relative_to(RAIZ).as_posix())
            rotulos.append(classe)
    partes = dividir_estratificado(caminhos, rotulos, 0.70, 0.15, seed)
    linhas = [
        {"caminho": c, "classe": y, "conjunto": nome}
        for nome, (xs, ys) in partes.items()
        for c, y in zip(xs, ys)
    ]
    df = pd.DataFrame(linhas)
    df.to_csv(CSV_SPLIT, index=False)
    return df


def plotar_distribuicao(df: pd.DataFrame) -> None:
    """Análise inicial: nº de imagens por classe em cada conjunto (barras empilhadas)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = pd.crosstab(df["classe"], df["conjunto"]).reindex(CLASSES)[["treino", "validacao", "teste"]]
    ax = tab.plot(kind="bar", stacked=True, figsize=(7, 4), color=["#2e7d32", "#81c784", "#c8e6c9"],
                  edgecolor="black", linewidth=0.4)
    for i, total in enumerate(tab.sum(axis=1)):
        ax.text(i, total + 5, str(total), ha="center", fontsize=9)
    ax.set_title("TrashNet — imagens por classe e conjunto (split 70/15/15)")
    ax.set_xlabel("")
    ax.set_ylabel("Nº de imagens")
    plt.xticks(rotation=0)
    plt.tight_layout()
    destino = RAIZ / "resultados" / "graficos" / "dados_distribuicao_classes.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destino, dpi=120)
    plt.close()


def main() -> None:
    if not all((DIR_BRUTO / c).exists() and any((DIR_BRUTO / c).iterdir()) for c in CLASSES):
        extrair(baixar_zip())
    else:
        print("Imagens já presentes em", DIR_BRUTO)
    df = gerar_split()
    plotar_distribuicao(df)
    print("\nDistribuição por conjunto e classe:")
    print(pd.crosstab(df["classe"], df["conjunto"], margins=True))
    print(f"\nSplit salvo em {CSV_SPLIT}")


if __name__ == "__main__":
    main()
