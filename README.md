# Mini-projeto: aprendizado por reforço tabular no MountainCar

Comparação de três agentes tabulares (Q-learning, SARSA(λ) e Dyna-Q) no `MountainCar-v0`
discretizado, avaliando eficiência amostral e computacional. O enunciado está em
`Mini-Projeto_ Modelagem e solução de problemas com Aprendizado por Reforço.md`.

## Estrutura

- `Mini-Projeto_ ... - Colab.ipynb`: notebook entregue (código, gráficos, análise e diário de
  bordo), com a configuração final na célula CONFIG.
- `mini_projeto_busca/`: busca de hiperparâmetros feita fora do notebook.
  - `busca.py`: fase 1, variações de um fator por vez em torno da configuração padrão.
  - `fase2.py`: combinações dos melhores valores de cada fator.
  - `rodada2.py`: busca coordenada a partir da melhor mudança isolada.
  - `resumo.py`: tabelas-resumo e gráficos da busca.
  - `PLANO_EXPERIMENTOS.md`: plano, configurações testadas e histórico de mudanças.
  - `NOTAS_ANALISE.md`: notas sobre os resultados.
- `resultados/`: cache das execuções (um `.pkl` e um `.csv` por configuração, cada uma com 10
  seeds × 500 episódios). O notebook e os scripts carregam daqui em vez de treinar de novo.

## Como reproduzir

O notebook roda com "Run all"; com os resultados em `resultados/`, nada é treinado de novo.
A busca é refeita com (cada comando pula o que já estiver no cache):

```
cd mini_projeto_busca
python busca.py --parte 0 --de 2      # e --parte 1 --de 2 em outro processo
python fase2.py --parte 0 --de 2      # e --parte 1 --de 2
python rodada2.py --parte 0 --de 2    # e --parte 1 --de 2
python resumo.py
python rodada2.py --resultados
```

Figuras, vídeos, tabelas-resumo e logs também estão versionados, mas são recriados por esses
comandos.
