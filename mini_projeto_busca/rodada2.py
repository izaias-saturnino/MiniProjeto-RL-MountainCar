"""Rodada 2: busca coordenada a partir da melhor mudança isolada.

Para cada algoritmo, a melhor mudança isolada da fase 1 (na grade de origem) vira a nova base, e
todos os outros fatores são variados de novo, um por vez, em torno dela. Para cada variação são
relatados a média, o IC 95% e o valor-p do teste t de Welch (5%) contra a nova base, apenas como
informação: o que conta como resultado novo é decidido depois de ver os resultados.

Extensões de faixa: quando o melhor valor ficou na borda da faixa testada, testa-se um valor além
dela (α = 0,02 para o Dyna-Q na baixa resolução; N = 40 para o Dyna-Q na alta resolução).

Uso:
    python rodada2.py --listar           # lista as execuções
    python rodada2.py --parte 0 --de 2   # roda metade
    python rodada2.py --parte 1 --de 2
    python rodada2.py --resultados       # compara cada variação com a nova base
"""
import argparse
import contextlib
import io
import os
import time

import numpy as np

import busca
import fase2
import resumo

AQUI = os.path.dirname(os.path.abspath(__file__))

# (algoritmo, grade de origem) -> mudanças da nova base em relação ao padrão
BASES = {
    ('Q-learning', 'baixa_resolucao'): dict(q_inicial=-100.0),
    ('SARSA(λ)', 'baixa_resolucao'): dict(lambda_=0.9, tipo_traco='acumulativo'),
    ('Dyna-Q', 'baixa_resolucao'): dict(alpha=0.05),
    ('Dyna-Q', 'alta_resolucao'): dict(n_planejamento=20),
}
# valores além da borda da faixa testada
EXTENSOES = {
    ('Dyna-Q', 'baixa_resolucao'): {'alpha': [0.02]},
    ('Dyna-Q', 'alta_resolucao'): {'n_planejamento': [40]},
}


def _rotulo(mudancas):
    return 'R2[' + ','.join(f'{k}={v}' for k, v in mudancas.items()) + ']' if mudancas else 'R2[base]'


def planejar():
    """Lista de (grade, algoritmo, rótulo, params, mudança em relação à base)."""
    tarefas = []
    for (alg, disc), mud_base in BASES.items():
        base = dict(busca.PADRAO, **busca.PADRAO_ALG[alg])
        base.update(mud_base)
        tarefas.append((disc, alg, _rotulo(mud_base), base, {}))
        fatores = {**fase2.FATORES_COMUNS, **fase2.FATORES_ESPECIFICOS[alg]}
        valores = {f: list(op) for f, op in fatores.items()}
        for f, extras in EXTENSOES.get((alg, disc), {}).items():
            valores[f] = valores[f] + extras
        for fator, opcoes in valores.items():
            chaves = fator if isinstance(fator, tuple) else (fator,)
            atual = tuple(base[k] for k in chaves)
            for valor in opcoes:
                vals = valor if isinstance(fator, tuple) else (valor,)
                if tuple(vals) == atual:
                    continue
                params = dict(base)
                mud = {}
                for k, v in zip(chaves, vals):
                    params[k] = v
                    mud[k] = v
                tarefas.append((disc, alg, _rotulo({**mud_base, **mud}), params, mud))
    return tarefas


def carregar_ns():
    with contextlib.redirect_stdout(io.StringIO()):
        return busca.registrar_discretizacoes(busca.carregar_codigo_notebook())


def resultados(ns, tarefas):
    col = f'passos_medios_ult{resumo.ULTIMOS}'
    dfs = []
    for disc, alg, rotulo, params, _ in tarefas:
        with contextlib.redirect_stdout(io.StringIO()):
            p = ns['executar_varias_seeds'](alg, disc, params, pasta=busca.PASTA, reutilizar=True,
                                            rotulo=rotulo, verbose=False)
        d = p['df'].copy()
        d['rotulo'] = rotulo
        dfs.append(d)
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    met = ns['metricas_por_execucao'](df, resumo.ULTIMOS)
    tab = resumo.resumir(ns, df, arquivo='resumo_rodada2.csv')
    achados = []
    for (alg, disc), mud_base in BASES.items():
        rot_base = _rotulo(mud_base)
        x_base = fase2.valores_por_seed(met, disc, alg, rot_base)
        print(f'\n=== {alg} | {disc} | nova base {rot_base}: {x_base.mean():.1f}')
        linhas = []
        for d, a, rotulo, _, mud in tarefas:
            if (a, d) != (alg, disc) or not mud:
                continue
            x = fase2.valores_por_seed(met, disc, alg, rotulo)
            p = fase2._p_welch(x, x_base)
            melhor = x.mean() < x_base.mean() and p < fase2.ALFA_TESTE
            linhas.append((x.mean(), rotulo, p, melhor))
            if melhor:
                achados.append((alg, disc, rotulo, x.mean(), x_base.mean(), p))
        for m, rotulo, p, melhor in sorted(linhas):
            r = tab[(tab.algoritmo == alg) & (tab.discretizacao == disc) & (tab.rotulo == rotulo)].iloc[0]
            print(f"  {'**' if melhor else '  '} {rotulo:55s} {m:6.1f} ±{r[col + '_ic95']:5.1f}  p={p:.3f}")
    print('\nSignificativamente melhores que a nova base (Welch 5%; informativo):')
    for a in achados or [('nenhum',)]:
        print('  ', a if len(a) == 1 else f'{a[0]} | {a[1]} | {a[2]}: {a[3]:.1f} vs base {a[4]:.1f} (p={a[5]:.3f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parte', type=int, default=0)
    ap.add_argument('--de', type=int, default=1)
    ap.add_argument('--listar', action='store_true')
    ap.add_argument('--resultados', action='store_true')
    args = ap.parse_args()

    ns = carregar_ns()
    tarefas = planejar()
    if args.listar:
        novas = 0
        for disc, alg, rotulo, params, _ in tarefas:
            existe = resumo._existe(ns, alg, disc, params)
            novas += not existe
            print(f"  {'cache' if existe else 'NOVA '}  {disc:16s} {alg:11s} {rotulo}")
        print(f'{len(tarefas)} execuções, {novas} novas')
        return
    if args.resultados:
        resultados(ns, tarefas)
        return

    busca.prioridade_baixa()
    minhas = [t for i, t in enumerate(tarefas) if i % args.de == args.parte]
    print(f'parte {args.parte}/{args.de}: {len(minhas)} execuções', flush=True)
    t0 = time.time()
    for j, (disc, alg, rotulo, params, _) in enumerate(minhas, 1):
        t = time.time()
        ns['executar_varias_seeds'](alg, disc, params, pasta=busca.PASTA, reutilizar=True,
                                    rotulo=rotulo, verbose=False)
        print(f'[{j}/{len(minhas)}] {disc:16s} {alg:11s} {rotulo:55s} '
              f'{(time.time() - t) / 60:5.1f} min  (total {(time.time() - t0) / 60:.1f} min)', flush=True)
    print('FIM', flush=True)


if __name__ == '__main__':
    main()
