# Notas para a análise (não entregar como está; escrever com minhas palavras)

## Q inicial × γ = 1 (hipótese refutada)

- **Hipótese original:** Q inicial = 0 é otimista e explora melhor; −100 e −300 (pessimistas)
  pioram.
- **Resultado parcial:** −100 e −300 tiveram desempenho igual ou melhor que 0. −100 e −300 deram
  **exatamente o mesmo episódio do 1º sucesso** em cada algoritmo (ex.: Q-learning `baixa_resolucao`: 186,8
  nos dois).
- **Antes do 1º sucesso:** com γ = 1 e recompensa −1, se tudo vale c, o alvo é −1 + c, que fica
  **abaixo** de c para qualquer c.
  - A ação tentada fica α abaixo das não tentadas, seja c = 0 ou c = −300.
  - Só as diferenças entre os valores importam; c não aparece na atualização → comportamento igual
    para qualquer Q inicial até achar o objetivo. Isso explica o 1º sucesso idêntico.
  - Q inicial = 0 deu um pouco diferente: provavelmente arredondamento de ponto flutuante mudando
    empates exatos no desempate aleatório (suposição, não verificada).
- **Com γ < 1 a intuição de "pessimista" vale:** o alvo −1 + γc fica acima de c quando
  c < −1/(1−γ) (= −10 para γ = 0,9).
  - Abaixo disso, a ação tentada parece melhor que as não tentadas → o agente repete o que conhece
    (pessimista).
  - Com γ = 1 essa fronteira está em −∞ → todo Q inicial finito é "otimista" antes do 1º sucesso.
- **Depois do 1º sucesso:** transições que chegam ao objetivo usam alvo = r (sem bootstrap), e os
  valores do caminho se ancoram nos retornos reais (≈ −k passos até o objetivo),
  independentemente de c. A partir daí c importa, pelo contraste com o que ainda não foi tentado:
  - c = 0: o não tentado (0) parece melhor que o caminho conhecido (−k) → continua saindo do
    caminho para explorar → explora mais e explora o caminho conhecido mais tarde.
  - c = −300: o caminho conhecido (−k, com k < 200) parece muito melhor que o não tentado (−300)
    → se compromete com ele cedo → melhor desempenho final.
- **−100 vs −300:** −100 fica *dentro* da faixa de valores reais (pessimista para estados a menos
  de ≈100 passos do objetivo, otimista para os mais distantes); −300 fica abaixo de todos.
  - Na prática foram estatisticamente iguais → depois que c fica abaixo dos valores do caminho
    aprendido, ir mais para baixo não muda nada.
- **Experimento opcional (não executado):** γ = 0,9 com Q inicial = −100. Abaixo da fronteira
  −10, seria pessimista "de verdade", e a hipótese original preveria exploração pior.

## λ = 0,99 pior que 0,9 (troca viés × variância)

- λ pequeno: depende das estimativas (bootstrapping), que no começo estão erradas → atualizações
  com **viés**.
- λ → 1: se aproxima de Monte Carlo, usa a trajetória longa real → menos viés, **mais variância**.
  - Com γλ = 0,99 o traço quase não decai num episódio (0,99²⁰⁰ ≈ 0,13): o vagar sem rumo 150
    passos antes do objetivo recebe quase o mesmo crédito que o balanço que funcionou.
- Por isso o melhor costuma ser um λ intermediário (curvas em "U" no capítulo de traços de
  elegibilidade do Sutton & Barto) → aqui, 0,9.

## γ = 1 vs γ = 0,9

- Com γ < 1 a recompensa terminal ainda se propaga, mas encolhe por γ a cada passo para trás.
- A ≈ 100 passos do objetivo, 0,9¹⁰⁰ ≈ 0,00003: balanços bons e ruins parecem quase iguais vistos
  do início, e o sinal se perde no ruído.
- γ = 1 mantém o sinal intacto ao longo do balanço longo.

## α

- α é o tamanho do passo, não tem a ver com exploração (isso é o ε).
- Grande: aprende rápido, mas cada amostra nova sobrescreve o que foi aprendido (pior com
  aliasing, porque amostras da mesma célula discordam).
- Pequeno: média estável, mas lento.
- Hipótese: α é o parâmetro mais sensível. Indício já visto: α = 0,3 derruba o Dyna-Q na baixa
  resolução (199,6), enquanto as variações de exploração ficam em 178–179.
- Atenção (diário): a hipótese foi formulada *depois* de ver α = 0,3; esse resultado a inspirou e
  não conta como evidência. O primeiro teste de verdade é α = 0,05 (resultado desconhecido quando
  a hipótese foi formulada).

## Resultados da busca completa (82 configurações × 10 seeds; só fatos, sem interpretação)

Métrica: média de passos nos últimos 50 episódios (menor = melhor; 200 = nunca chega).
Tabela completa: `resumo_busca.csv` / `resumo_saida.txt`.

- **Efeito piso:**
  - Em `alta_resolucao`, Q-learning e SARSA(λ) ficam em 200,0 em **todas** as variações, inclusive
    λ até 0,99 (no máximo 3/10 seeds chegaram ao objetivo alguma vez).
  - Em `media_resolucao` e `pos_vel_detalhada`, ≈ 199,6.
  - 500 episódios não distinguem variações dos agentes sem modelo nessas grades.
  - Dyna-Q: 10/10 seeds chegaram em todas as grades.
  - Nenhum agente perto de convergir (melhor resultado ≈ 147).
  - Decisão: não estender o orçamento. Formular como resultado limitado ao orçamento ("em 500
    episódios, Q-learning e SARSA(λ) não aprenderam nas grades mais finas"), não como "não
    conseguem aprender". Limitação a citar no diário.
  - **Correção depois da fase 2:** o SARSA(λ) *ajustado* (Q inicial −100/−300, λ = 0,9) aprende na
    `media_resolucao` (151,7; padrão: 199,0) e na `pos_vel_detalhada` (154,3). Só a
    `alta_resolucao` continua em 200. O piso era em grande parte das configurações **não
    ajustadas**. O Q-learning (que ficou no padrão) continua no piso fora da baixa resolução.
  - Se uma das grades do notebook for fina, as curvas sem modelo ficam planas em 200 nos
    gráficos 4.1 (válido, e mostra a eficiência amostral do Dyna-Q).
  - Impacto nos finalistas: na grade fina do notebook não dá para escolher a melhor configuração
    de Q-learning e SARSA(λ) (todas empatam em ≈ 200). Opção: reaproveitar os finalistas desses
    algoritmos na baixa resolução e declarar isso explicitamente. Dyna-Q e a baixa resolução não
    são afetados.
- **Q inicial:** −100 / −300 iguais ou melhores que 0 (Q-learning, baixa resolução: 178,6 / 180,6
  vs 188,2). Hipótese não confirmada.
- **Exploração:**
  - Dyna-Q, baixa resolução: todas as variações de ε/decaimento entre 178,0 e 179,0 (dentro do IC).
  - Mudam muito o 1º sucesso (episódio 34 a 81), pouco o desempenho final.
- **λ (SARSA, baixa resolução):** 0 → 184,6; 0,7 → 174,0; 0,9 → 150,4; 0,99 → 168,6 (melhora até
  0,9, piora em 0,99). Na alta resolução: sem efeito (piso).
- **Traço acumulativo:** igual ou melhor que o de substituição em todo λ (λ = 0,9: 147,3 ± 3,5 vs
  150,4 ± 10,8; melhor resultado geral). Hipótese de "estourar" não confirmada.
- **Modelo `ultimo`:** baixa resolução 199,5 (7/10 seeds chegaram alguma vez) vs 178,6 amostral;
  alta resolução 182,2 vs 165,7. Efeito bem maior na grade grosseira. Hipótese confirmada.
- **N = 20 vs 10:**
  - Alta resolução: 153,2 vs 165,7 (melhor).
  - Baixa resolução: 178,9 vs 178,6 (igual).
  - Custo: ≈ 1,8–2× atualizações de Q.
- **γ = 0,9:** pior onde dá para distinguir (Dyna-Q, baixa resolução: 197,1).
- **α = 0,3:**
  - Dyna-Q, baixa resolução: colapsa (199,6).
  - Dyna-Q, alta resolução: um pouco pior (168,9).
  - Sem modelo, baixa resolução: parecido com o padrão.
- **Discretizações (padrão):**
  - Dyna-Q: media 158,8 < alta 165,7 < pos_vel 171,9 < vel 176,0 < baixa 178,6 < pos 192,1.
  - Sem modelo: razoável só na baixa resolução.
  - Velocidade detalhada > posição detalhada para os três agentes.

## Fase 2 (combinações dos melhores valores; só fatos)

Tabela: `resumo_fase2.csv`; gráfico: `fase2_grades.png`.

- **Teste da independência** (combinação vs melhor variação isolada, na grade de origem):
  - SARSA(λ), baixa resolução: melhor isolada λ 0,9 acumulativo 147,3 ± 3,5; combinações 151,9 a
    159,7 → nenhuma melhor.
  - Dyna-Q, alta resolução: N = 20 isolado 153,2 ± 2,7; Q −100 + N 20 → 152,6 ± 2,4 (igual).
  - Dyna-Q, baixa resolução: só α = 0,05 mudou → mesma execução (165,7).
  - Em nenhum caso combinar superou a melhor mudança isolada.
- **Melhor resultado por grade** (melhor combinação de cada algoritmo; Q-learning = padrão):

  | Grade | Dyna-Q | SARSA(λ) | Q-learning |
  |---|---|---|---|
  | baixa_resolucao | 165,7 | 151,9 | 188,2 |
  | media_resolucao | 153,2 | 151,7 | 199,7 |
  | alta_resolucao | 152,6 | 200,0 | 200,0 |
  | pos_detalhada | 180,0 | 149,9 | 193,5 |
  | vel_detalhada | 159,1 | 145,1 | 197,7 |
  | pos_vel_detalhada | 154,1 | 154,3 | 199,7 |

- Pela métrica: melhor grade do Dyna-Q = alta (152,6), empatada com media (153,2) e pos_vel
  (154,1). Sem modelo: SARSA(λ) melhor na vel_detalhada (145,1); Q-learning na baixa (188,2). Os
  dois agentes sem modelo discordam sobre a melhor grade.
## Rodada 2 (busca coordenada; só fatos)

Tabela: `rodada2_resultados.txt` / `resumo_rodada2.csv`. Valor-p: Welch contra a nova base.

- **Única variação significativamente melhor que a nova base:** Dyna-Q, alta resolução, N = 40:
  148,0 ± 3,0 vs 153,2 (N = 20), p = 0,009. Ainda na borda da faixa.
  - Tendência: N 10 → 165,7; 20 → 153,2; 40 → 148,0 (ganho diminuindo).
  - Custo: 1,0 M → 1,8 M → 3,4 M atualizações de Q; 16 s → 24 s → 47 s por seed.
  - 1º sucesso: episódio 115 → 91 → 83.
- **Perto, mas não significativo:**
  - Q-learning, decaimento 0,9: 170,0 vs 178,6 (p = 0,16).
  - Dyna-Q, baixa resolução: Q −300 → 162,0 vs 165,7 (p = 0,08); α = 0,02 → 163,4 (p = 0,23).
  - SARSA(λ), α = 0,05: 145,0 vs 147,3 (p = 0,58).
- **Interações reveladas pela mudança de base:**
  - SARSA(λ), Q inicial −100: com λ = 0 era melhor que Q = 0 (175,4 vs 184,6); com λ = 0,9
    acumulativo é significativamente pior (159,0 vs 147,3, p = 0,005). Explica por que as
    combinações da fase 2 (Q −100 + λ 0,9) não superaram λ 0,9 sozinho.
  - Dyna-Q, alta resolução, γ = 0,9: pior com N = 10; com N = 20 indistinguível (150,9 vs 153,2,
    p = 0,14).

## α = 0,05

- **α = 0,05 (1º teste real da hipótese de sensibilidade):** Dyna-Q baixa resolução 165,7 vs 178,6
  (melhor, p < 0,05); Q-learning 198,1 vs 188,2 e SARSA 191,0 vs 184,6 (piores, p < 0,05).
