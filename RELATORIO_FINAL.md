# Relatório final — EcoSort

## 1. O que foi feito
| Item do checklist | Status | Evidência |
|---|---|---|
| Dataset de imagens baixado, organizado e dividido | ✅ | TrashNet (2.527 imagens) em `dados/bruto/trashnet/` (não versionado); split 70/15/15 estratificado em `dados/split_imagens.csv` |
| Modelo de visão treinado, avaliado e salvo, com gráficos | ✅ | `modelos/visao_mobilenetv2.pt`; `resultados/metricas_visao.json`; `resultados/graficos/visao_*.png`; log em `resultados/log_treino_visao.txt` |
| Dataset de intenções (300+), PLN treinado e comparado, com gráficos | ✅ | `dados/intencoes.csv` (335 frases); `modelos/pln_intencoes.pkl`; `resultados/pln_comparacao_modelos.md`; `resultados/graficos/pln_matriz_confusao_*.png` |
| Base de regras e módulo de decisão | ✅ | `dados/regras_descarte.json`; `src/decisao/motor_regras.py` (29 regras) |
| Fluxo integrado ponta a ponta | ✅ | `src/pipeline.py` (`processar(imagem, pergunta)`) |
| App Streamlit funcional | ✅ | `app.py` iniciado com `streamlit run`, sem erros no log; testado no navegador com upload de imagem (lata → Metal 60,8% → regra R18 → lixeira amarela) |
| Testes pytest passando | ✅ | 37 testes passando (`python -m pytest`) |
| 10+ exemplos de entrada/saída e análise de erros | ✅ | 14 casos em `resultados/exemplos/` (`exemplos.md`, `exemplos.json`, `painel_exemplos.png`, `imagens/`); `resultados/analise_erros.md` |
| README completo com métricas reais | ✅ | `README.md` (inclui público-alvo, objetivos, justificativa do aprendizado supervisionado, origem/licença/limpeza dos dados e lista de integrantes) |
| Artigo científico (IEEE, LaTeX) | ✅ (falta compilar no Overleaf) | `artigo/ecosort_artigo.tex` + `artigo/figuras/` — ver `artigo/README.md` |
| DECISOES.md e RELATORIO_FINAL.md | ✅ | este arquivo e `DECISOES.md` (29 decisões) |

Extras: notebooks `notebooks/01_treino_modelos.ipynb` e `notebooks/02_analise_resultados.ipynb` (este executado, com saídas);
gráficos de distribuição de confiança e de cobertura × acurácia por limiar; roteiro do artigo em `artigo/README.md`.

## 2. Métricas obtidas (todas de execuções reais)

**Visão (MobileNetV2, teste com 380 imagens):** acurácia **0,8684**, F1 macro **0,8461**, melhor acurácia de
validação 0,8865. F1 por classe: cardboard 0,897 · glass 0,882 · metal 0,882 · paper 0,875 · plastic 0,875 · trash 0,667.
Acurácia por lixeira: 0,8947. Com limiar 0,6: cobertura 76,3% e 96,6% de acerto nas respostas diretas.
Tempo de treino: ~23 s/época (cabeça) e ~33 s/época (fine-tuning) em CPU, 16 épocas.

**PLN (teste com 51 frases):**
| Modelo | Acurácia | F1 macro | CV 5-fold acurácia |
|---|---:|---:|---:|
| TF-IDF + Regressão Logística (C=3) | 0,8431 | 0,8400 | 0,8328 ± 0,0466 |
| TF-IDF + Naive Bayes (α=0,5) | 0,7451 | 0,7365 | 0,7522 ± 0,0307 |

**Testes:** 37/37 passando.

## 3. Principais decisões (detalhes em `DECISOES.md`)
- PyTorch CPU + MobileNetV2 em 2 fases (cabeça → fine-tuning dos 5 últimos blocos), pesos por classe e label smoothing.
- TrashNet redimensionado do Hugging Face (espelho oficial), com fallback para o GitHub.
- Stopwords próprias (as listas padrão removem "onde", "como", "antes", que são pistas de intenção).
- Motor de regras com dois limiares de visão (0,35 e 0,6) e um de PLN (0,4); regra R03 que responde com segurança quando as duas hipóteses levam à mesma lixeira; `fora_do_escopo` incerto vira resumo (R05) em vez de recusa.

## 4. Problemas encontrados e soluções
| Problema | Solução |
|---|---|
| `python -m venv` falhou ao instalar o pip (caminho da pasta > 260 caracteres, limite MAX_PATH do Windows) | Ambiente virtual criado num caminho curto fora do projeto; documentado no README |
| `fora_do_escopo` disparava mesmo com baixa confiança do PLN (ex.: "me explica esse item") | R01 passou a exigir confiança ≥ 0,4; abaixo disso cai em R05 (resumo) — teste adicionado |
| Frase do teste automatizado "quem ganhou o jogo ontem?" era classificada como `como_descartar` ("jogo" de "jogo isso fora") | Mantido como erro documentado na análise; o teste usa outra frase fora do escopo |
| `st.image(use_container_width=...)` está depreciado no Streamlit 1.67 | Trocado por `width="stretch"` |

## 5. Como executar — passo a passo

Na pasta `ecosort/`:

1. **Criar o ambiente e instalar dependências**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1          # Windows PowerShell  (Linux/macOS: source .venv/bin/activate)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
2. **(Opcional, só para retreinar) Baixar e dividir o dataset**
   ```bash
   python -m src.visao_computacional.preparar_dados
   ```
3. **(Opcional) Treinar os modelos** — os modelos treinados já estão em `modelos/`
   ```bash
   python -m src.visao_computacional.treinar
   python -m src.pln.treinar
   ```
4. **Rodar os testes**
   ```bash
   python -m pytest -v
   ```
5. **(Opcional, requer passo 2) Regenerar exemplos e análise de erros**
   ```bash
   python -m scripts.gerar_exemplos
   python -m scripts.analise_erros
   ```
6. **Abrir o app**
   ```bash
   streamlit run app.py
   ```
   Acesse http://localhost:8501, envie uma foto (há exemplos em `dados/exemplos/`), digite a pergunta e clique em **Analisar**.

## 6. Pendências para a equipe
- Preencher o link do Overleaf no `README.md` e em `artigo/README.md`.
- Compilar `artigo/ecosort_artigo.tex` no Overleaf, preencher os autores, conferir o limite de 6 páginas e salvar o PDF em `artigo/`.
