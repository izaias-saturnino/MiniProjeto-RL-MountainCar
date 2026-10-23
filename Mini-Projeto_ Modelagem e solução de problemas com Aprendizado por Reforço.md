# **Mini-Projeto: Modelagem e solução de problemas com Aprendizado por Reforço**

## **1\. Contextualização**

Neste projeto, vocês irão aplicar os conhecimentos obtidos até então em aula, além de usar a criatividade para conceber uma solução viável para um problema de aprendizado por reforço.  O objetivo é avaliar a eficiência amostral (*sample efficiency*) de diferentes abordagens.

O uso de assistentes/agentes de IA é permitido e encorajado, porém apenas para tarefas “braçais” (e.g. implementar os algoritmos que você já entendeu ou o codigo de plotar gráficos). Dessa forma, a avaliação irá valorizar o seu projeto de experimentos e sua capacidade de análise de resultados.

## **2\. O Ambiente**

O ambiente será o `MountainCar-v0` da biblioteca Gymnasium.

* O objetivo é fazer o carro subir a montanha à direita.  
* O motor é fraco, exigindo que o agente aprenda a dar ré para ganhar embalo.  
* O espaço de ações é discreto (3 ações), mas o **espaço de estados é contínuo** (Posição e Velocidade).  
* A recompensa é de \-1 a cada passo de tempo (*step*), o que caracteriza um problema de recompensa esparsa.

## **3\. Objetivos e Tarefas**

A ideia neste projeto é usar os algoritmos já vistos em aula. Portanto, a modelagem deve viabilizar o uso de aprendizado tabular.

### **Tarefa A: Modelagem do Espaço de Estados**

Vocês deverão definam e implementem pelo menos **duas formas ou resoluções de discretização diferentes**. Por exemplo, uma grade mais "grossa" (menos estados discretos; representação mais grosseira) e uma grade mais "fina" (mais  estados discretos; representação mais precisa).

### **Tarefa B: Implementação dos Agentes**

Implementem e treinem três agentes tabulares distintos:

1. **Agente Baseline (TD de 1 passo):** Q-Learning ou SARSA(0) padrão.  
2. **Agente com Propagação Acelerada:** SARSA(λ) usando Traços de Elegibilidade.  
3. **Agente Baseado em Modelos:** Dyna-Q, realizando N passos de planejamento a cada iteração com o ambiente (você poderá testar diferentes valores de N).

### **Tarefa C: Avaliação Experimental e Reprodutibilidade**

Encontre a melhor parametrização para cada algoritmo.

Considere que o aprendizado por reforço possui alta variância. Para lidar com isso, execute o treinamento de cada algoritmo pelo menos **10 vezes** (utilizando sementes/seeds aleatórias diferentes). Registre a recompensa do episódio e a quantidade acumulada de passos (total de *steps* de interação com o ambiente desde o início do treinamento). Registre também o custo computacional, ou seja, quantas atualizações de valores Q foram feitas por cada algoritmo.

## **4\. Entregáveis e Estrutura do Relatório**

A entrega é o arquivo correspondente ao Colab de base preenchido preenchido com o código executável e as células de texto compondo as seções abaixo. O Colab de base está no link: [https\://drive.google.com/file/d/1oA8EDn\_o851IkrhcJo26X7F-ykIbrznD/view?usp=sharing](https://drive.google.com/file/d/1oA8EDn_o851IkrhcJo26X7F-ykIbrznD/view?usp=sharing).

### **4.1. Gráficos de Curva de Aprendizado (Recompensa vs. Passos no Ambiente)**

Gerem conjuntos de gráficos para cada técnica de discretização experimentada.

* No primeiro conjunto de gráficos, plote as curvas dos 3 algoritmos (Baseline, SARSA(λ) e Dyna-Q), com a contagem de episódios no eixo x e o número de passos até o sucesso no eixo y. Para cada algoritmo, a linha principal deve ser a **média** das 10 repetições. Adicione uma **área sombreada** ao redor da linha representando o Intervalo de Confiança de 95%.  
* No segundo conjunto de gráficos, plote as curvas dos 3 algoritmos (Baseline, SARSA(λ) e Dyna-Q), com a contagem de episódios no eixo x e o número de atualizações de valor-Q no eixo y. Para cada algoritmo, a linha principal deve ser a **média** das 10 repetições. Adicione uma **área sombreada** ao redor da linha representando o Intervalo de Confiança de 95%.

### **4.2. Análise de Resultados e Aliasing de Estados**

Compare o desempenho dos algoritmos e das discretizações:

* Qual o melhor algoritmo em termos de eficiência amostral?  
* Qual o melhor algoritmo em termos de eficiência computacional?  
* Como a modelagem de estados afetou especificamente o modelo construído pelo Dyna-Q em comparação com o SARSA(λ)? Pense na relação entre as transições reais do ambiente vs nos estados discretizados.  
* Gere vídeos no colab dos agentes que valem a pena ser vistos (e.g. melhor desempenho, política inusitada, etc.)

### **4.3. Diário de Bordo e Registro de Falhas**

Relatem de forma transparente as **tentativas que não deram certo** e/ou que resultaram em desempenho inferior.

* O que notadamente falhou nas escolhas de hiperparâmetros (alpha, decaimento de epsilon, λ, N) falharam? Por que?  
* Houve alguma dificuldade ou erros na implementação de alguma etapa do projeto? Como foi resolvido?

## **5\. Rubrica de Avaliação**

| Critério | Descrição | Peso |
| :---- | :---- | :---: |
| **Implementação: Discretização** | Criação correta das funções de discretização e definição clara de duas estratégias distintas (Tarefa A). | 15% |
| **Implementação: Agentes** | Código correto e funcional para Q-Learning/SARSA, SARSA(λ) e Dyna-Q (Tarefa B). | 25% |
| **Rigor Experimental** | Estrutura de repetição com pelo menos 10 *seeds* aleatórias e coleta correta das métricas por *steps* (Tarefa C). | 10% |
| **Visualização de Dados (Gráficos)** | Gráficos gerados corretamente, eixos corretos, linhas de média e sombreamento presentes e visíveis. | 15% |
| **Análise e Discussão** | Profundidade da comparação entre os algoritmos e a compreensão demonstrada sobre as diferentes representações dos Estados. | 20% |
| **Diário de Bordo** | Riqueza de detalhes sobre as tentativas, justificativas técnicas para as falhas. | 15% |

