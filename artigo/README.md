# Artigo — EcoSort

- **Link do Overleaf:** `<COLE_AQUI_O_LINK_DO_OVERLEAF>`
- Coloque aqui o PDF final (`ecosort_artigo.pdf`) e, se quiser, o `.tex`.

## Roteiro sugerido (formato SBC/IEEE)

1. **Introdução** — problema do descarte incorreto de resíduos no Brasil; objetivo do EcoSort.
2. **Trabalhos relacionados** — TrashNet (Yang & Thung, 2016), transfer learning com MobileNet, chatbots de intenção com TF-IDF.
3. **Metodologia**
   - 3.1 Visão: TrashNet, split 70/15/15, MobileNetV2 em 2 fases (`resultados/graficos/visao_curvas_treino.png`).
   - 3.2 PLN: dataset próprio de 335 frases, TF-IDF + RegLog vs. Naive Bayes.
   - 3.3 Decisão: agente reativo baseado em regras + base de conhecimento JSON (regras R01–R05, R10–R33).
   - 3.4 Integração e interface (Streamlit).
4. **Resultados** — usar os números de `resultados/metricas_visao.json`, `resultados/pln_comparacao_modelos.md` e as figuras de `resultados/graficos/`.
5. **Análise de erros** — `resultados/analise_erros.md`.
6. **Limitações e trabalhos futuros**.
7. **Conclusão**.
