# Artigo — EcoSort

- **Fonte LaTeX:** [`ecosort_artigo.tex`](ecosort_artigo.tex) (IEEE Conference, duas colunas, português)
- **Figuras:** [`figuras/`](figuras/) (geradas por `python -m scripts.diagrama_arquitetura`, que copia os gráficos de `resultados/graficos/`)
- **Link do Overleaf:** `<COLE_AQUI_O_LINK_DO_OVERLEAF>`
- **PDF final:** salvar aqui como `ecosort_artigo.pdf` depois de compilar.

## Como publicar no Overleaf
1. No Overleaf: **New Project → Upload Project** e envie um `.zip` com `ecosort_artigo.tex` e a pasta `figuras/`.
2. Compilador: **pdfLaTeX** (padrão). A classe `IEEEtran` já vem instalada no Overleaf.
3. Preencha o **curso e os e-mails** (os nomes já estão) no bloco `\author` (linhas marcadas com `PREENCHER`).
4. Confira se o PDF ficou com **no máximo 6 páginas** (estimativa: 4–5 páginas).
5. **Share → Turn on link sharing → link de visualização (view only)** e cole o link aqui e no `README.md` principal.
6. Baixe o PDF e salve como `artigo/ecosort_artigo.pdf`.

## Estrutura
Resumo (191 palavras) · 1 Introdução · 2 Fundamentação e trabalhos relacionados · 3 Metodologia
(arquitetura, dados, visão, PLN, decisão, métricas) · 4 Resultados e discussão (visão, análise de erros,
limiar, PLN, fluxo integrado, limitações) · 5 Conclusão e trabalhos futuros · 11 referências.
Todos os números vêm de `resultados/metricas_visao.json`, `resultados/metricas_pln.json` e `resultados/analise_erros_dados.md`.
