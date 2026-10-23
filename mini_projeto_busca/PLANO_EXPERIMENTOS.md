# Plano de experimentos: mini-projeto MountainCar (agentes tabulares)

> **Rascunho.** As seções 1–4 e 7 descrevem o que é executado. As seções 5 e 6 são notas
> para eu escrever com minhas palavras na justificativa e no diário de bordo do notebook.

## 1. Objetivo

O objetivo destes experimentos é comparar a eficiência amostral e a eficiência computacional de
três agentes tabulares no `MountainCar-v0` discretizado, e testar hipóteses sobre o efeito de
cada escolha de modelagem e de hiperparâmetro. As hipóteses não são tratadas como resultados
esperados: cada uma é posta à prova para ser confirmada ou refutada pelos dados. Para valores de
γ diferentes de 1 não há hipótese prévia, e a análise é exploratória, feita depois de ver os
resultados. O uso de γ = 1 na configuração padrão, por outro lado, é uma escolha deliberada,
discutida na seção 5. A taxa de aprendizado α é tratada como ajuste fino, com a hipótese de que
é o parâmetro ao qual o desempenho é mais sensível.

## 2. Metodologia

São comparados três agentes: Q-learning, usado como baseline, SARSA(λ) e Dyna-Q. O desenho
experimental é do tipo "um fator por vez": parte-se de uma configuração padrão e cada variação
altera um único fator em relação a ela, o que permite atribuir a diferença observada àquele
fator. Há duas exceções, em que pequenas grades completas permitem observar interações: o ε
inicial é combinado com mais de um decaimento, e cada valor de λ é testado com os dois tipos de
traço de elegibilidade.

Cada configuração é executada com 10 seeds (0 a 9), as mesmas para todas as configurações. Cada
execução tem 500 episódios, com o limite padrão do ambiente de 200 passos por episódio, ou seja,
no máximo 100.000 passos de interação. Em cada episódio são registrados a recompensa, o número de
passos até o sucesso, os passos acumulados no ambiente desde o início do treino, as atualizações
de valores Q (no episódio e acumuladas), o valor de ε e o tempo de execução.

A métrica usada para escolher a melhor configuração foi definida antes de ver os resultados: o
desempenho final, isto é, a média de passos por episódio nos últimos 50 episódios (quanto menor,
melhor). Além dela, todas as configurações são avaliadas por métricas complementares, que não
decidem a escolha: o episódio do primeiro sucesso, que mede a rapidez em encontrar o objetivo; a
média de passos em todos os episódios, equivalente à área sob a curva de aprendizado e portanto
uma medida da eficiência amostral ao longo de todo o treino; e o número de atualizações de
valores Q, que mede o custo computacional. Como o primeiro sucesso não existe nas seeds que nunca
chegam ao objetivo, ele é sempre acompanhado do número de seeds que tiveram algum sucesso.

## 3. Discretizações

O estado do MountainCar tem duas dimensões: a posição do carro, entre −1,2 e 0,6 (o objetivo é
atingido em 0,5), e a velocidade, entre −0,07 e 0,07. Todas as discretizações são grades sobre
essas duas dimensões, e cada dimensão pode ser dividida de forma uniforme ou não uniforme.

Três discretizações são uniformes e diferem apenas na resolução, para medir o efeito de quantas
células existem. A grade `baixa_resolucao` divide cada dimensão em 10 intervalos (largura 0,18 na posição
e 0,014 na velocidade), totalizando 100 células. A grade `media_resolucao` usa 20 intervalos por dimensão
(400 células), e a grade `alta_resolucao` usa 40 (1.600 células, com largura 0,045 na posição e 0,0035 na
velocidade).

Outras duas discretizações são não uniformes e têm aproximadamente o mesmo número de células que
a grade de baixa resolução, para medir o efeito de onde as células estão, e não de quantas são. A
`pos_detalhada` divide a posição em 18 intervalos, com células de 0,05 perto da parede da
esquerda, em torno do fundo do vale (por volta de −0,52) e no trecho final antes do objetivo
(0,4 a 0,5), e células de até 0,2 no restante da pista; a velocidade fica com 6 intervalos
uniformes, totalizando 108 células. A `vel_detalhada` faz o oposto: a posição tem 6 intervalos
uniformes de 0,3, e a velocidade tem 16 intervalos não uniformes, de 0,002 perto de zero até 0,02
perto dos extremos, totalizando 96 células. Por fim, a `pos_vel_detalhada` combina as duas,
usando os 18 intervalos de posição da primeira e os 16 de velocidade da segunda (288 células). As
bordas internas exatas estão em `DISCRETIZACOES_EXTRAS`, no arquivo `busca.py`.

| Nome | Posição | Velocidade | Células | O que testa |
|---|---|---|---|---|
| `baixa_resolucao` | 10 intervalos uniformes (0,18) | 10 uniformes (0,014) | 100 | resolução baixa |
| `media_resolucao` | 20 uniformes (0,09) | 20 uniformes (0,007) | 400 | resolução intermediária |
| `alta_resolucao` | 40 uniformes (0,045) | 40 uniformes (0,0035) | 1.600 | resolução alta |
| `pos_detalhada` | 18 não uniformes: 0,05 perto da parede esquerda, do fundo do vale (≈ −0,52) e de 0,4–0,5; até 0,2 no meio | 6 uniformes | 108 | **onde** ficam as células (≈ mesmo nº da `baixa_resolucao`) |
| `vel_detalhada` | 6 uniformes (0,3) | 16 não uniformes: 0,002 perto de 0, até 0,02 nos extremos | 96 | idem, detalhando a velocidade |
| `pos_vel_detalhada` | as 18 da `pos_detalhada` | as 16 da `vel_detalhada` | 288 | combinação das duas |

## 4. Configuração padrão e variações

Na configuração padrão, o fator de desconto γ é 1, todos os valores Q começam em 0, a taxa de
aprendizado α é 0,1 e a exploração ε-gulosa começa com ε = 0,5, multiplicado por 0,99 ao fim de
cada episódio até o piso de 0,01, que é atingido por volta do episódio 389. No SARSA(λ), o padrão
é λ = 0, ou seja, o SARSA de um passo, com traços de substituição. No Dyna-Q, o padrão é N = 10
passos de planejamento por passo real, com o modelo amostral, que guarda a contagem de todos os
desfechos observados para cada par estado-ação.

As variações que se aplicam aos três agentes alteram γ para 0,9; o valor Q inicial para −100 e
para −300; α para 0,3; e a exploração. Na exploração, o decaimento de ε é testado com 0,95 e 0,9,
que levam ε ao piso por volta dos episódios 76 e 37, e o ε inicial é testado com 0,1, tanto com o
decaimento padrão (piso por volta do episódio 229) quanto com decaimento 0,95 (piso por volta do
episódio 45). As variações exclusivas do SARSA(λ) testam λ = 0,7, 0,9 e 0,99, cada um com traços
de substituição e com traços acumulativos. As variações exclusivas do Dyna-Q testam N = 20 e o
modelo "último desfecho", que guarda apenas a transição observada mais recente para cada par
estado-ação.

Depois da primeira rodada, o valor Q inicial passou a ser cruzado com outros fatores, formando
duas pequenas matrizes. Na primeira, os valores −100 e −300 são combinados com γ = 0,9, para
testar a inicialização num caso em que o desconto torna esses valores pessimistas. Na segunda,
os mesmos valores são combinados com as configurações de exploração (ε inicial 0,1, decaimento
0,95, e os dois juntos), para verificar qual combinação de exploração e inicialização funciona
melhor. As células dessas matrizes com Q inicial = 0 já existiam entre as variações anteriores.

Todas as variações rodam nas grades `baixa_resolucao` e `alta_resolucao`. Nas outras quatro discretizações roda
apenas a configuração padrão, para comparar as discretizações entre si. Ao todo são 136 execuções,
cada uma com 10 seeds e 500 episódios.

### Fase 2: combinação dos melhores valores

Como as variações mudam um fator por vez, elas não dizem como os fatores interagem, e as
discretizações só foram comparadas com a configuração padrão. Na segunda fase, para cada
algoritmo, combinam-se os melhores valores de cada fator encontrados na primeira fase, supondo
que os fatores são independentes (mesmo sabendo que podem não ser). O melhor valor de cada fator
é o que tiver a menor média de desempenho final nas variações de um fator por vez, e considera-se
empatado com ele o valor cuja diferença não é significativa num teste t de Welch a 5% (10 seeds
de cada lado). A seleção é condicionada à evidência: se o valor padrão de um fator está empatado
com o melhor, o fator permanece no padrão, porque não há evidência para mudá-lo; caso contrário,
entram o melhor valor e todos os valores empatados com ele. As combinações são todas as misturas
dos valores selecionados de cada fator.

Essa regra substituiu uma versão anterior, em que todos os valores empatados com o melhor
entravam, mesmo quando o padrão também estava empatado. Com só 10 seeds o teste tem pouco poder,
e quase todos os valores empatavam: a versão anterior gerava 236 combinações (1.416 execuções,
cerca de 24 horas), a maioria misturando mudanças sem evidência de efeito. Com a regra
condicionada à evidência ficam 8 combinações: a configuração padrão para o Q-learning (nenhum
fator tem o padrão significativamente pior), quatro para o SARSA(λ) (Q inicial −100 ou −300 com
λ = 0,9, traço de substituição ou acumulativo), uma para o Dyna-Q a partir da `baixa_resolucao`
(α = 0,05) e duas a partir da `alta_resolucao` (Q inicial −100 ou −300 com N = 20). Para o Dyna-Q, que tem
resultados distinguíveis nas duas grades testadas, monta-se uma combinação a partir da
`baixa_resolucao` e outra a partir da `alta_resolucao`. Para Q-learning e SARSA(λ), que só têm
resultados distinguíveis na `baixa_resolucao`, monta-se uma combinação a partir dela. Cada
combinação é executada nas seis discretizações (8 combinações × 6 grades = 48 execuções).

Essa fase serve a dois propósitos. Primeiro, testa a suposição de independência: se os fatores
fossem independentes, a combinação deveria ser pelo menos tão boa quanto a melhor variação
isolada, e um resultado pior é evidência de interação entre os fatores. Segundo, compara as
discretizações com cada algoritmo usando seus melhores valores, e não só a configuração padrão,
o que serve de base para escolher as duas discretizações usadas no notebook: a melhor para o
agente baseado em modelo e a melhor para os agentes sem modelo, com os três agentes executados
nas duas.

### Rodada 2: busca coordenada a partir da melhor mudança isolada

Como última etapa da busca, a melhor mudança isolada de cada algoritmo na fase 1 passa a ser a nova
configuração base, e todos os outros fatores são variados de novo, um por vez, em torno dela, na
mesma grade de origem. As novas bases são Q inicial = −100 para o Q-learning (melhor média, embora
sem diferença significativa para o padrão), λ = 0,9 com traço acumulativo para o SARSA(λ), α = 0,05
para o Dyna-Q na `baixa_resolucao` e N = 20 para o Dyna-Q na `alta_resolucao`. Nos dois casos em que
o melhor valor estava na borda da faixa testada, a faixa é estendida com um valor além dela
(α = 0,02 e N = 40). Para cada variação são registrados a média, o IC 95% e o valor-p do teste t de
Welch contra a nova base. A decisão sobre o que conta como um resultado novo, e se ele justifica
outra rodada, é tomada depois de ver os resultados; por causa do prazo, se nada relevante aparecer,
a busca termina aqui. São 48 execuções, das quais 26 são novas (as demais já existiam no cache).

**Padrão:**

| Parâmetro | Valor |
|---|---|
| γ | 1 |
| Q inicial | 0 |
| ε inicial | 0,5 |
| decaimento de ε | 0,99 por episódio |
| ε mínimo | 0,01 (atingido no episódio ≈ 389) |
| α | 0,1 |
| λ | 0 |
| tipo de traço | substituição |
| N | 10 |
| tipo de modelo | amostral |

**Variações** (todas rodam em `baixa_resolucao` e `alta_resolucao`; nas outras 4 discretizações roda só o padrão):

| Rótulo | Mudança | Algoritmos | ε mínimo atingido no episódio |
|---|---|---|---|
| `padrao` | — | todos | ≈ 389 |
| `gamma=0.9` | γ = 0,9 | todos | ≈ 389 |
| `q_inicial=-100` | Q inicial = −100 | todos | ≈ 389 |
| `q_inicial=-300` | Q inicial = −300 | todos | ≈ 389 |
| `eps=0.1` | ε inicial = 0,1 | todos | ≈ 229 |
| `eps=0.1,decay=0.95` | ε inicial = 0,1; decaimento 0,95 | todos | ≈ 45 |
| `decay=0.95` | decaimento 0,95 | todos | ≈ 76 |
| `decay=0.9` | decaimento 0,9 | todos | ≈ 37 |
| `alpha=0.3` | α = 0,3 | todos | ≈ 389 |
| `alpha=0.05` | α = 0,05 (acrescentada depois) | todos | ≈ 389 |
| `lambda=0.7` / `0.9` / `0.99` | λ (traço de substituição) | SARSA(λ) | ≈ 389 |
| `lambda=…,traco=acumulativo` | λ = 0,7 / 0,9 / 0,99, traço acumulativo | SARSA(λ) | ≈ 389 |
| `N=20` | N = 20 | Dyna-Q | ≈ 389 |
| `modelo=ultimo` | modelo "último desfecho" | Dyna-Q | ≈ 389 |

**Matrizes com o Q inicial** (todos os algoritmos; ✓ = já existia, **nova** = acrescentada):

| | Q inicial = 0 | Q inicial = −100 | Q inicial = −300 |
|---|---|---|---|
| γ = 1, ε 0,5, decaimento 0,99 (padrão) | ✓ | ✓ | ✓ |
| γ = 0,9, ε 0,5, decaimento 0,99 | ✓ | **nova** | **nova** |
| γ = 1, ε 0,1, decaimento 0,99 | ✓ | **nova** | **nova** |
| γ = 1, ε 0,5, decaimento 0,95 | ✓ | **nova** | **nova** |
| γ = 1, ε 0,1, decaimento 0,95 | ✓ | **nova** | **nova** |

## 5. Notas para a justificativa (reescrever com minhas palavras)

**Discretizações**
- 3 uniformes → efeito de *quantas* células.
- 2 não uniformes com ≈100 células → efeito de *onde* estão.
- Baixa resolução: muitas visitas por célula vs aliasing forte (célula = 10% da pista).
- Alta resolução: pouco aliasing vs poucas visitas por célula em ≤100 mil passos.
- `pos_detalhada`: detalhe onde a posição decide (parede/inversão, vale/início, empurrar até o
  topo). Hipótese: precisão de posição importa.
- `vel_detalhada`: a política boa é "empurrar no sentido da velocidade". Perto de v≈0 está a
  inversão; separar "quase parado" de "andando". Menos auto-laços quando o carro está lento.
- Combinada: as duas juntas valem mais que cada uma?

**Hiperparâmetros**
- Q-learning como baseline: Dyna-Q = Q-learning + modelo + planejamento → a diferença isola o
  efeito do planejamento.
- Q inicial 0 é otimista (valores reais entre ≈ −200 e 0) → exploração sistemática.
  - Hipótese: −100 piora e −300 (pessimista) piora mais, sobretudo com ε baixo.
- ε 0,5 → 0,01: explorar no começo, quase guloso no fim.
  - Hipótese: com Q otimista, ε importa menos que Q inicial.
  - Decaimentos 0,95 e 0,9 testam explorar por menos tempo.
- λ = 0 é a referência (SARSA de 1 passo).
  - Hipótese: λ maior acelera, porque a recompensa só vem depois de um balanço longo.
  - 0,99 ≈ Monte Carlo → mais variância.
  - Hipótese: o traço acumulativo pode "estourar" com as revisitas do carro lento.
- N = 10 vs 20: retorno decrescente? Mais planejamento amplifica o erro do modelo sob aliasing?
- Modelo amostral vs último: a discretização torna as transições aparentemente estocásticas.
  - Hipótese: o "último" sofre com auto-laços, principalmente na grade de baixa resolução.
- γ = 1 no padrão: recompensa esparsa (−1 por passo até o objetivo), então não descontar
  preserva o sinal do objetivo ao longo de um balanço longo; Q = −(passos esperados até o objetivo).
- γ = 0,9: sem hipótese prévia; análise exploratória.
- α (0,05 / 0,1 / 0,3): hipótese: α é mais sensível que os outros parâmetros. α é o tamanho do
  passo: grande aprende rápido mas sobrescreve o que aprendeu (pior com aliasing); pequeno é
  estável mas lento.
- Interação observada a registrar: com γ = 1, Q inicial não altera o comportamento antes do
  1º sucesso; ver `NOTAS_ANALISE.md`.

## 6. Histórico de mudanças no plano (material para o diário de bordo)

1. Padrão inicial com decaimento 0,999 → trocado para 0,99. Com 0,999, ε ainda estaria ≈ 0,30 no
   episódio 500, e as curvas finais misturariam a política aprendida com ação aleatória.
2. Q inicial −1 → trocado por −100 e −300. −1 ainda é otimista (≈ igual a 0), então não testaria
   inicialização não otimista.
3. λ padrão 0,1 → 0. Para medir o efeito dos traços, a referência certa é λ = 0.
   - λ testados: 0,7 / 0,9 / 0,99, com os dois tipos de traço.
   - Removida a variação "traço acumulativo com λ = 0": é idêntica ao padrão, porque com λ = 0 os
     traços zeram a cada passo.
4. Removidas as variações com decaimento 0,999 (com ε 0,5 e com ε 0,1). Não levam ε para perto de
   zero dentro do orçamento, ao contrário do que a literatura espera (explorar e convergir para a
   política gulosa).
5. Acrescentados decaimento 0,9, a discretização 20×20, as duas não uniformes e a combinação delas.
6. Depois da primeira rodada: hipótese do Q inicial refutada para γ = 1 → acrescentadas as matrizes
   Q inicial × γ (γ = 0,9 com −100 e −300) e Q inicial × exploração (ε 0,1 / decaimento 0,95 /
   os dois, com −100 e −300). 48 execuções novas (total 130).
7. Acrescentado α = 0,05, para testar a sensibilidade a α (total 136).
8. Planejada a fase 2: combinação dos melhores valores de cada fator (supondo independência),
   executada nas 6 discretizações, para testar a independência e escolher as grades do notebook.
9. Regra da fase 2: incluir todos os empates gerava 236 combinações (≈ 24 h), porque o teste com
   10 seeds tem pouco poder. Trocada pela regra condicionada à evidência (só muda um fator se o
   padrão for significativamente pior): 8 combinações, 48 execuções.
10. Rodada 2 (busca coordenada): melhor mudança isolada vira a nova base e os outros fatores são
    variados de novo; extensões α = 0,02 e N = 40 (borda da faixa). O que conta como resultado
    novo é decidido depois de ver os resultados. Última etapa por causa do prazo.

## 7. Como reproduzir

A busca é executada pelo script `busca.py`, que reaproveita o código do notebook. Os comandos
abaixo listam as 136 execuções e rodam a busca dividida em duas metades, que podem ser executadas
em paralelo em dois processos:

```
python busca.py --listar
python busca.py --parte 0 --de 2
python busca.py --parte 1 --de 2
```

Os resultados são gravados em `../resultados/`, a mesma pasta de cache usada pelo notebook.
Execuções já concluídas são carregadas do disco em vez de treinadas de novo, de modo que, se a
configuração final escolhida for uma das testadas aqui, o notebook não precisa treiná-la outra
vez. Depois da busca, `python resumo.py` gera a tabela-resumo e os gráficos de comparação.
