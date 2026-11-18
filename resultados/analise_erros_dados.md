# Dados da análise de erros (gerado automaticamente)

Conjunto de teste: 380 imagens, 50 erros (acurácia 0.8684).

## Pares (real → predito) mais confundidos

| classe    | predito   |   n |   % dos erros |
|:----------|:----------|----:|--------------:|
| cardboard | paper     |   6 |            12 |
| plastic   | glass     |   5 |            10 |
| paper     | trash     |   5 |            10 |
| glass     | plastic   |   5 |            10 |
| paper     | cardboard |   4 |             8 |
| plastic   | trash     |   3 |             6 |
| glass     | metal     |   3 |             6 |
| trash     | paper     |   3 |             6 |
| metal     | glass     |   3 |             6 |
| paper     | metal     |   2 |             4 |
| plastic   | metal     |   2 |             4 |
| cardboard | metal     |   1 |             2 |
| cardboard | glass     |   1 |             2 |
| metal     | trash     |   1 |             2 |
| glass     | trash     |   1 |             2 |
| metal     | paper     |   1 |             2 |
| metal     | plastic   |   1 |             2 |
| paper     | plastic   |   1 |             2 |
| trash     | metal     |   1 |             2 |
| trash     | plastic   |   1 |             2 |

## Recall por classe

| classe    |   n |   acertos |   recall |
|:----------|----:|----------:|---------:|
| cardboard |  60 |        52 |   0.8667 |
| glass     |  76 |        67 |   0.8816 |
| metal     |  62 |        56 |   0.9032 |
| paper     |  89 |        77 |   0.8652 |
| plastic   |  73 |        63 |   0.863  |
| trash     |  20 |        15 |   0.75   |

## Confiança da visão: acertos x erros

|         |   count |   mean |    50% |    min |    max |
|:--------|--------:|-------:|-------:|-------:|-------:|
| erros   |      50 | 0.4703 | 0.4657 | 0.2702 | 0.8257 |
| acertos |     330 | 0.7806 | 0.8312 | 0.2652 | 0.9976 |

## Efeito do limiar de confiança

Cobertura = fração das imagens em que o sistema responde direto (sem pedir nova foto).

|   limiar |   cobertura |   acuracia_acima_limiar |   erros_acima_limiar |
|---------:|------------:|------------------------:|---------------------:|
|     0    |      1      |                  0.8684 |                   50 |
|     0.35 |      0.9658 |                  0.8883 |                   41 |
|     0.4  |      0.9316 |                  0.904  |                   34 |
|     0.5  |      0.8447 |                  0.9502 |                   16 |
|     0.6  |      0.7632 |                  0.9655 |                   10 |
|     0.7  |      0.6421 |                  0.9918 |                    2 |
|     0.8  |      0.4921 |                  0.9947 |                    1 |
|     0.9  |      0.2421 |                  1      |                    0 |

## Erros de PLN no teste — regressao_logistica (8 erros)

| texto                              | real           | predito        |
|:-----------------------------------|:---------------|:---------------|
| entra na coleta seletiva?          | e_reciclavel   | onde_levar     |
| isso e recilavel?                  | e_reciclavel   | fora_do_escopo |
| onde fica a praia mais bonita?     | fora_do_escopo | onde_levar     |
| vai chover amanhã?                 | fora_do_escopo | como_descartar |
| onde fica o hospital mais próximo? | fora_do_escopo | onde_levar     |
| isso é rejeito?                    | e_reciclavel   | fora_do_escopo |
| quem ganhou o jogo ontem?          | fora_do_escopo | como_descartar |
| isso é aceito na coleta seletiva?  | e_reciclavel   | onde_levar     |

## Erros de PLN no teste — naive_bayes (13 erros)

| texto                              | real           | predito        |
|:-----------------------------------|:---------------|:---------------|
| kkkkkk                             | fora_do_escopo | como_descartar |
| entra na coleta seletiva?          | e_reciclavel   | onde_levar     |
| alguém recolhe isso?               | onde_levar     | e_reciclavel   |
| isso e recilavel?                  | e_reciclavel   | como_descartar |
| onde fica a praia mais bonita?     | fora_do_escopo | onde_levar     |
| vai chover amanhã?                 | fora_do_escopo | como_descartar |
| me recomenda um filme              | fora_do_escopo | como_descartar |
| onde fica o hospital mais próximo? | fora_do_escopo | onde_levar     |
| tô com fome                        | fora_do_escopo | como_descartar |
| isso é rejeito?                    | e_reciclavel   | como_descartar |
| desmonto o papelão?                | preparo_antes  | e_reciclavel   |
| quem ganhou o jogo ontem?          | fora_do_escopo | como_descartar |
| isso é aceito na coleta seletiva?  | e_reciclavel   | onde_levar     |
