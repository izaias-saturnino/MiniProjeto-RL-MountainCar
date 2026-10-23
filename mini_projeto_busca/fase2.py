"""Fase 2: combinação dos melhores valores de cada fator (supondo independência).

Regra (definida antes de ver os resultados da fase 1):
  * para cada algoritmo e cada fator, o melhor valor é o de menor média de desempenho final
    (passos médios nos últimos 50 episódios) entre as variações de UM fator por vez;
  * empate = diferença não significativa num teste t de Welch a 5% (10 seeds de cada lado);
  * regra condicionada à evidência: se o valor padrão está empatado com o melhor, o fator fica no
    padrão; senão, entram o melhor e todos os valores empatados com ele;
  * as combinações são todas as misturas dos valores selecionados de cada fator;
  * Dyna-Q: valores escolhidos a partir da baixa_resolucao e da alta_resolucao (união das
    combinações); Q-learning e SARSA(λ): só a partir da baixa_resolucao (na alta todos ≈ 200);
  * cada combinação roda nas 6 discretizações.

Uso:
    python fase2.py --listar             # mostra os empates, as combinações e o nº de execuções
    python fase2.py --parte 0 --de 2     # roda metade das execuções
    python fase2.py --parte 1 --de 2
"""
import argparse
import contextlib
import io
import itertools
import time

import numpy as np
from scipy import stats

import busca
import resumo

ALFA_TESTE = 0.05
ULTIMOS = resumo.ULTIMOS

# fator -> {valor: rótulo da variação de um fator por vez (fase 1)}
FATORES_COMUNS = {
    'gamma': {1.0: 'padrao', 0.9: 'gamma=0.9'},
    'q_inicial': {0.0: 'padrao', -100.0: 'q_inicial=-100', -300.0: 'q_inicial=-300'},
    'epsilon_inicial': {0.5: 'padrao', 0.1: 'eps=0.1'},
    'decaimento_epsilon': {0.99: 'padrao', 0.95: 'decay=0.95', 0.9: 'decay=0.9'},
    'alpha': {0.1: 'padrao', 0.3: 'alpha=0.3', 0.05: 'alpha=0.05'},
}
FATORES_ESPECIFICOS = {
    'Q-learning': {},
    # λ e tipo de traço formam uma grade completa na fase 1, então entram como um fator conjunto
    'SARSA(λ)': {
        ('lambda_', 'tipo_traco'): {
            (0.0, 'substituicao'): 'padrao',
            (0.7, 'substituicao'): 'lambda=0.7',
            (0.9, 'substituicao'): 'lambda=0.9',
            (0.99, 'substituicao'): 'lambda=0.99',
            (0.7, 'acumulativo'): 'lambda=0.7,traco=acumulativo',
            (0.9, 'acumulativo'): 'lambda=0.9,traco=acumulativo',
            (0.99, 'acumulativo'): 'lambda=0.99,traco=acumulativo',
        },
    },
    'Dyna-Q': {
        'n_planejamento': {10: 'padrao', 20: 'N=20'},
        'tipo_modelo': {'amostral': 'padrao', 'ultimo': 'modelo=ultimo'},
    },
}
GRADES_ORIGEM = {
    'Q-learning': ['baixa_resolucao'],
    'SARSA(λ)': ['baixa_resolucao'],
    'Dyna-Q': ['baixa_resolucao', 'alta_resolucao'],
}
TODAS_AS_GRADES = ['baixa_resolucao', 'media_resolucao', 'alta_resolucao',
                   'pos_detalhada', 'vel_detalhada', 'pos_vel_detalhada']


def valores_por_seed(met, disc, alg, rotulo):
    g = met[(met.discretizacao == disc) & (met.algoritmo == alg) & (met.rotulo == rotulo)]
    return g.sort_values('seed')[f'passos_medios_ult{ULTIMOS}'].to_numpy()


def _p_welch(x, y):
    if np.std(x) == 0 and np.std(y) == 0:
        return 1.0 if np.mean(x) == np.mean(y) else 0.0   # sem variância: compara as médias
    return stats.ttest_ind(x, y, equal_var=False).pvalue


def empatados(met, disc, alg, opcoes):
    """Valores selecionados do fator (regra "condicionada à evidência").

    Se o valor padrão está empatado com o melhor (Welch, 5%), o fator fica no padrão: não há
    evidência para mudá-lo. Caso contrário, entram o melhor e todos os valores empatados com ele.
    """
    medias = {v: valores_por_seed(met, disc, alg, r).mean() for v, r in opcoes.items()}
    melhor = min(medias, key=medias.get)
    padrao = next(v for v, r in opcoes.items() if r == 'padrao')
    x_melhor = valores_por_seed(met, disc, alg, opcoes[melhor])
    p_vs_melhor = {v: (1.0 if v == melhor else _p_welch(valores_por_seed(met, disc, alg, r), x_melhor))
                   for v, r in opcoes.items()}
    if p_vs_melhor[padrao] >= ALFA_TESTE:
        selecionados = [padrao]
    else:
        selecionados = [v for v in opcoes if p_vs_melhor[v] >= ALFA_TESTE]
    detalhes = [(v, medias[v], p_vs_melhor[v], v in selecionados) for v in opcoes]
    return selecionados, detalhes


def montar_combinacoes(met, verbose=True):
    combinacoes = {}   # (alg, rótulo) -> params
    for alg in busca.PADRAO_ALG:
        fatores = {**FATORES_COMUNS, **FATORES_ESPECIFICOS[alg]}
        for disc in GRADES_ORIGEM[alg]:
            if verbose:
                print(f'\n=== {alg} | fatores escolhidos a partir de "{disc}"')
            escolhas = {}
            for fator, opcoes in fatores.items():
                sel, det = empatados(met, disc, alg, opcoes)
                escolhas[fator] = sel
                if verbose:
                    nome = '×'.join(fator) if isinstance(fator, tuple) else fator
                    txt = '  '.join(f"{'*' if e else ' '}{v}: {m:.1f} (p={p:.2f})" for v, m, p, e in det)
                    print(f'  {nome:28s} {txt}')
            for mistura in itertools.product(*escolhas.values()):
                params = dict(busca.PADRAO, **busca.PADRAO_ALG[alg])
                partes = []
                for fator, valor in zip(escolhas, mistura):
                    chaves = fator if isinstance(fator, tuple) else (fator,)
                    vals = valor if isinstance(fator, tuple) else (valor,)
                    for k, v in zip(chaves, vals):
                        if params[k] != v:
                            partes.append(f'{k}={v}')
                        params[k] = v
                rotulo = 'F2[' + ','.join(partes) + ']' if partes else 'F2[padrao]'
                combinacoes[(alg, rotulo)] = params
    return combinacoes


def planejar_fase2(ns, df):
    with contextlib.redirect_stdout(io.StringIO()):
        met = ns['metricas_por_execucao'](df, ULTIMOS)
    combinacoes = montar_combinacoes(met)
    tarefas = [(disc, alg, rotulo, params)
               for (alg, rotulo), params in combinacoes.items() for disc in TODAS_AS_GRADES]
    return combinacoes, tarefas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parte', type=int, default=0)
    ap.add_argument('--de', type=int, default=1)
    ap.add_argument('--listar', action='store_true')
    args = ap.parse_args()

    ns, df = resumo.carregar()
    combinacoes, tarefas = planejar_fase2(ns, df)
    print(f'\n{len(combinacoes)} combinações × {len(TODAS_AS_GRADES)} grades = {len(tarefas)} execuções')
    if args.listar:
        for alg, rotulo in combinacoes:
            print(f'  {alg:11s} {rotulo}')
        return

    busca.prioridade_baixa()
    minhas = [t for i, t in enumerate(tarefas) if i % args.de == args.parte]
    print(f'parte {args.parte}/{args.de}: {len(minhas)} execuções', flush=True)
    t0 = time.time()
    for j, (nome_disc, nome_alg, rotulo, params) in enumerate(minhas, 1):
        t = time.time()
        ns['executar_varias_seeds'](nome_alg, nome_disc, params, pasta=busca.PASTA, reutilizar=True,
                                    rotulo=rotulo, verbose=False)
        print(f'[{j}/{len(minhas)}] {nome_disc:17s} {nome_alg:11s} {rotulo:40s} '
              f'{(time.time() - t) / 60:5.1f} min  (total {(time.time() - t0) / 60:.1f} min)', flush=True)
    print('FIM', flush=True)


if __name__ == '__main__':
    main()
