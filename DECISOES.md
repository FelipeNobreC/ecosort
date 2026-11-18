# Registro de decisões — EcoSort

Cada item: **decisão** — justificativa curta.

## Ambiente
1. **Python 3.12 + PyTorch 2.x versão CPU** — o enunciado exige rodar em CPU/Colab gratuito; a wheel CPU é ~10× menor que a CUDA e torna a instalação reproduzível. O código usa apenas CPU (funciona igual no Colab).
2. **Ambiente virtual fora da pasta do projeto (no Windows de desenvolvimento)** — a pasta do projeto tinha um caminho muito longo e o `pip` falhou por limite de 260 caracteres do Windows (MAX_PATH) ao instalar o próprio pip/torch. Solução: criar o venv num caminho curto. Isso não afeta o projeto; o README recomenda `python -m venv .venv` normalmente (em caminhos curtos funciona).
3. **PyTorch/torchvision em vez de TensorFlow/Keras** — instalação mais leve em CPU, API de transfer learning simples com `torchvision.models`, pesos do ImageNet baixados automaticamente.

## Dados de imagem
4. **TrashNet via Hugging Face (`garythung/trashnet`, arquivo `dataset-resized.zip`, 42 MB)** — é o espelho oficial do autor; o GitHub original fica como fonte alternativa automática no script. Usamos a versão redimensionada (512×384) porque a original tem 3,6 GB e a rede recebe 224×224 de qualquer forma.
5. **Split guardado em `dados/split_imagens.csv` em vez de copiar imagens para pastas treino/val/teste** — evita duplicar arquivos, deixa o split auditável e versionável (o CSV é pequeno, as imagens não são versionadas).
6. **Split 70/15/15 estratificado em duas etapas (seed 42)** — garante a mesma proporção de classes (inclusive a minoritária `trash`, 137 imagens) nos três conjuntos.
7. **12 imagens do conjunto de TESTE copiadas para `dados/exemplos/` (2 por classe, ~240 KB)** — os testes automatizados e o notebook precisam de imagens reais versionadas, já que o dataset bruto não é versionado.

## Visão computacional
8. **MobileNetV2 (ImageNet)** — rede leve (3,5 M parâmetros, ~9 MB) adequada para CPU, com bom desempenho em transfer learning.
9. **Treino em 2 fases: 6 épocas só da cabeça (lr 1e-3) + 10 épocas de fine-tuning dos 5 últimos blocos (lr 1e-4 na base, 3e-4 na cabeça, cosine annealing)** — fase 1 adapta o classificador sem destruir as features; fase 2 especializa as camadas mais semânticas. ~23 s/época na fase 1 e ~33 s/época na fase 2 em CPU de 8 threads (~8 min no total).
10. **BatchNorm da base mantido em modo `eval` durante a fase 1** — com a base congelada, atualizar as estatísticas de BN com lotes pequenos degradaria as features do ImageNet.
11. **CrossEntropy com pesos por classe (inverso da frequência) + label smoothing 0.1** — compensa o desbalanceamento (`trash` tem ~1/4 das imagens de `paper`) e reduz excesso de confiança, o que torna o limiar de confiança mais útil.
12. **Augmentation só no treino: RandomResizedCrop(0.7–1.0), flips horizontal/vertical, rotação ±15°, ColorJitter** — o TrashNet tem fundo branco e iluminação uniforme; augmentations aumentam a robustez para fotos reais. Avaliação usa Resize(256)+CenterCrop(224).
13. **Checkpoint escolhido pela melhor acurácia de VALIDAÇÃO** — o teste é usado uma única vez, no final, evitando vazamento.
14. **`predict()` retorna `(classe, confianca, top2)` com confiança = probabilidade softmax máxima** — simples e suficiente para o componente de decisão aplicar limiares.

## PLN
15. **Dataset próprio escrito à mão (335 frases, 65–70 por intenção)** — inclui gírias ("mano", "vei", "parada"), abreviações ("cm", "q", "tem q"), erros de digitação ("presisa", "recilavel", "discarto") e frases fora do escopo que compartilham palavras com o domínio ("como lavar roupa branca?", "onde fica o hospital?") para forçar o modelo a não depender de uma única palavra.
16. **Lista de stopwords própria em vez da do NLTK** — as listas padrão removem "onde", "como", "não", "antes", que são exatamente as pistas mais fortes de intenção. Também evita download de recursos externos.
17. **TF-IDF de palavras (uni+bigramas, `sublinear_tf`) + Regressão Logística; baseline MultinomialNB** — conforme o enunciado. Não adicionamos n-gramas de caracteres (ajudariam com erros de digitação) para manter a especificação; fica registrado como melhoria futura.
18. **Hiperparâmetros (C e alpha) escolhidos pelo F1 macro na validação + validação cruzada 5-fold como medida de robustez** — o dataset é pequeno (51 frases no teste), então uma única partição tem variância alta; o CV 5-fold complementa.
19. **Modelo final re-treinado em treino+validação** — usa mais dados para o modelo implantado; as métricas reportadas são do modelo treinado só no treino e avaliado no teste.
20. **Texto vazio ou só com stopwords/pontuação → `fora_do_escopo` com confiança 1.0** — sem vocabulário útil, não há o que classificar.

## Decisão (sistema baseado em regras)
21. **Motor de regras próprio em Python, com prioridade explícita e regra acionada rastreável** — sem dependências extras; cada resposta informa o ID e a descrição da regra (explicabilidade exigida na interface).
22. **Base de conhecimento em JSON com cores da Resolução CONAMA 275/2001** (papel=azul, plástico=vermelho, vidro=verde, metal=amarelo, orgânico=marrom, rejeito=cinza) — `cardboard` e `paper` foram mapeados para azul; `trash` para cinza (rejeito).
23. **Dois limiares de visão: 0.35 (mínimo) e 0.6 (confiável)** — abaixo de 0.35 o agente só pede outra foto (R02); entre 0.35 e 0.6 mostra as 2 classes prováveis (R04) ou, se as duas vão para a mesma lixeira (papel × papelão), responde com segurança (R03). O limiar 0.6 foi sugerido no enunciado e validado com a curva cobertura × acurácia (`resultados/graficos/limiar_cobertura.png`).
24. **Limiar de 0.4 para a confiança do PLN** — com 5 classes, 0.4 é o dobro do acaso (0.2); abaixo disso o agente dá um resumo geral (R05) em vez de arriscar uma resposta específica.
25. **`fora_do_escopo` só é aceito com confiança ≥ 0.4; se incerto, vai para R05** — perguntas vagas como "me explica esse item" eram classificadas como fora do escopo com baixa confiança; recusar seria pior para o usuário do que dar um resumo.
26. **24 regras específicas geradas a partir da tabela classe × intenção (R10–R33)** — garante cobertura completa por construção, verificada em teste automatizado.

## Interface e testes
27. **Streamlit com layout em duas colunas e todas as saídas intermediárias** (classe, confiança, top-2, intenção, regra acionada, JSON completo).
28. **Testes do pytest usam os modelos treinados reais** (não mocks) — validam o comportamento real do sistema; exigem rodar o treino antes (ou usar os modelos versionados).
29. **Modelos versionados em `modelos/` (9 MB + <1 MB)** — bem abaixo do limite de 100 MB; permite rodar o app sem retreinar.
