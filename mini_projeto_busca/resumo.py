"""Resumo da busca de hiperparâmetros (rodar depois de busca.py).

Carrega do cache (../resultados/) todas as execuções planejadas em busca.py e gera:
  resumo_busca.csv          métricas por configuração (média entre as 10 seeds ± IC 95%)
  desempenho_final_<disc>.png   desempenho final de cada variação, por algoritmo
  discretizacoes_padrao.png     configuração padrão nas 6 discretizações

Métrica de escolha: desempenho final = média de passos nos últimos 50 episódios (menor = melhor).
Métricas complementares: episódio do 1º sucesso, média de passos em todos os episódios
(área sob a curva), atualizações de Q.
"""
import contextlib
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import busca

AQUI = os.path.dirname(os.path.abspath(__file__))
ULTIMOS = 50


def carregar():
    with contextlib.redirect_stdout(io.StringIO()):
        ns = busca.registrar_discretizacoes(busca.carregar_codigo_notebook())
    tarefas = busca.planejar(list(busca.DISCRETIZACOES_BASE))
    dfs, faltando = [], []
    for nome_disc, nome_alg, rotulo, params in tarefas:
        with contextlib.redirect_stdout(io.StringIO()):
            pacote = ns['executar_varias_seeds'](nome_alg, nome_disc, params, pasta=busca.PASTA,
                                                 reutilizar=True, rotulo=rotulo, verbose=False) \
                if _existe(ns, nome_alg, nome_disc, params) else None
        if pacote is None:
            faltando.append((nome_disc, nome_alg, rotulo))
            continue
        df = pacote['df'].copy()
        df['rotulo'] = rotulo
        dfs.append(df)
    if faltando:
        print(f'ATENÇÃO: {len(faltando)} execuções ainda não estão no cache (busca não terminou?):')
        for f in faltando:
            print('   ', *f)
    return ns, pd.concat(dfs, ignore_index=True)


def _existe(ns, nome_alg, nome_disc, params):
    """True se a execução já está no cache (para não treinar nada aqui)."""
    chave, _ = ns['_chave_config'](nome_alg, ns['DISCRETIZADORES'][nome_disc], params, ns['SEEDS'],
                                   ns['N_EPISODIOS'], ns['MAX_PASSOS_TOTAL'], ns['MAX_PASSOS_EPISODIO'])
    return any(chave in f for f in os.listdir(busca.PASTA) if f.endswith('.pkl'))


def resumir(ns, df, arquivo='resumo_busca.csv'):
    """Tabela-resumo (média ± IC 95% entre seeds) + nº de seeds com pelo menos um sucesso."""
    tab = ns['tabela_resumo'](df, ultimos=ULTIMOS)
    # o 1º sucesso é NaN quando a seed nunca chegou ao objetivo
    met = ns['metricas_por_execucao'](df, ULTIMOS)
    com_sucesso = (met.assign(ok=met.primeiro_sucesso_ep.notna())
                   .groupby(['discretizacao', 'algoritmo', 'rotulo'], sort=False).ok.sum()
                   .rename('seeds_com_sucesso').reset_index())
    tab = tab.merge(com_sucesso, on=['discretizacao', 'algoritmo', 'rotulo'])
    tab = tab.sort_values(['discretizacao', 'algoritmo', f'passos_medios_ult{ULTIMOS}'])
    tab.to_csv(os.path.join(AQUI, arquivo), index=False)
    return tab


def imprimir(tab):
    col = f'passos_medios_ult{ULTIMOS}'
    for (disc, alg), g in tab.groupby(['discretizacao', 'algoritmo'], sort=False):
        print(f'\n=== {disc} | {alg}  (ordenado pelo desempenho final; menor = melhor)')
        for _, r in g.iterrows():
            print(f"  {r.rotulo:30s} final {r[col]:6.1f} ±{r[col + '_ic95']:5.1f} | "
                  f"1º sucesso ep {r.primeiro_sucesso_ep:6.1f} ({int(r.seeds_com_sucesso)}/10 seeds) | "
                  f"média todos {r.passos_medios_todos:6.1f} | "
                  f"atualiz. Q {r.atualizacoes_q_total / 1e6:6.2f} M")


def grafico_variacoes(tab, disc):
    col = f'passos_medios_ult{ULTIMOS}'
    g_disc = tab[tab.discretizacao == disc]
    algs = list(dict.fromkeys(g_disc.algoritmo))
    fig, axes = plt.subplots(1, len(algs), figsize=(5.2 * len(algs), 6), sharex=True)
    for ax, alg in zip(np.atleast_1d(axes), algs):
        g = g_disc[g_disc.algoritmo == alg].sort_values(col, ascending=False)
        cores = ['tab:orange' if r == 'padrao' else 'tab:blue' for r in g.rotulo]
        ax.barh(g.rotulo, g[col], xerr=g[col + '_ic95'], color=cores, alpha=0.8, capsize=3)
        ax.set_title(alg)
        ax.set_xlabel(f'passos médios nos últimos {ULTIMOS} episódios (IC 95%)')
        ax.axvline(200, color='gray', ls='--', lw=1)
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle(f'Desempenho final por variação, discretização "{disc}" (menor = melhor; laranja = padrão)')
    fig.tight_layout()
    caminho = os.path.join(AQUI, f'desempenho_final_{disc}.png')
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_discretizacoes(tab):
    col = f'passos_medios_ult{ULTIMOS}'
    g = tab[tab.rotulo == 'padrao']
    ordem = ['baixa_resolucao', 'media_resolucao', 'alta_resolucao', 'pos_detalhada', 'vel_detalhada', 'pos_vel_detalhada']
    algs = list(dict.fromkeys(g.algoritmo))
    x = np.arange(len(ordem))
    larg = 0.8 / len(algs)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, alg in enumerate(algs):
        ga = g[g.algoritmo == alg].set_index('discretizacao').reindex(ordem)
        ax.bar(x + i * larg, ga[col], larg, yerr=ga[col + '_ic95'], capsize=3, label=alg)
    ax.set_xticks(x + larg * (len(algs) - 1) / 2, ordem)
    ax.set_ylabel(f'passos médios nos últimos {ULTIMOS} episódios (IC 95%)')
    ax.axhline(200, color='gray', ls='--', lw=1)
    ax.set_title('Configuração padrão nas 6 discretizações (menor = melhor)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    caminho = os.path.join(AQUI, 'discretizacoes_padrao.png')
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


ORDEM_GRADES = ['baixa_resolucao', 'media_resolucao', 'alta_resolucao',
                'pos_detalhada', 'vel_detalhada', 'pos_vel_detalhada']


def carregar_fase2(ns, df):
    """Carrega do cache as execuções da fase 2 (combinações × 6 grades) que já existirem."""
    import fase2
    with contextlib.redirect_stdout(io.StringIO()):
        _, tarefas = fase2.planejar_fase2(ns, df)
    dfs, faltando = [], 0
    for nome_disc, nome_alg, rotulo, params in tarefas:
        if not _existe(ns, nome_alg, nome_disc, params):
            faltando += 1
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            pacote = ns['executar_varias_seeds'](nome_alg, nome_disc, params, pasta=busca.PASTA,
                                                 reutilizar=True, rotulo=rotulo, verbose=False)
        d = pacote['df'].copy()
        d['rotulo'] = rotulo
        dfs.append(d)
    if faltando:
        print(f'ATENÇÃO: {faltando} execuções da fase 2 ainda não estão no cache.')
    return pd.concat(dfs, ignore_index=True) if dfs else None


def grafico_fase2(tab2):
    """Desempenho final de cada combinação da fase 2 nas 6 grades, um painel por algoritmo."""
    col = f'passos_medios_ult{ULTIMOS}'
    algs = list(dict.fromkeys(tab2.algoritmo))
    fig, axes = plt.subplots(len(algs), 1, figsize=(12, 4.2 * len(algs)), sharex=True)
    x = np.arange(len(ORDEM_GRADES))
    for ax, alg in zip(np.atleast_1d(axes), algs):
        g = tab2[tab2.algoritmo == alg]
        rotulos = list(dict.fromkeys(g.rotulo))
        larg = 0.8 / len(rotulos)
        for i, r in enumerate(rotulos):
            gr = g[g.rotulo == r].set_index('discretizacao').reindex(ORDEM_GRADES)
            ax.bar(x + i * larg, gr[col], larg, yerr=gr[col + '_ic95'], capsize=2, label=r)
        ax.axhline(200, color='gray', ls='--', lw=1)
        ax.set_title(alg)
        ax.set_ylabel(f'passos médios, últimos {ULTIMOS} ep.')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(axis='y', alpha=0.3)
    np.atleast_1d(axes)[-1].set_xticks(x + 0.4 - 0.4 / max(1, len(rotulos)), ORDEM_GRADES)
    fig.suptitle('Fase 2: combinações dos melhores valores nas 6 discretizações (menor = melhor)')
    fig.tight_layout()
    caminho = os.path.join(AQUI, 'fase2_grades.png')
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


if __name__ == '__main__':
    ns, df = carregar()
    tab = resumir(ns, df)
    imprimir(tab)
    for disc in ('baixa_resolucao', 'alta_resolucao'):
        print('figura:', grafico_variacoes(tab, disc))
    print('figura:', grafico_discretizacoes(tab))
    print('tabela:', os.path.join(AQUI, 'resumo_busca.csv'))

    df2 = carregar_fase2(ns, df)
    if df2 is not None:
        tab2 = resumir(ns, df2, arquivo='resumo_fase2.csv')
        print('\n######## FASE 2 ########')
        imprimir(tab2)
        print('figura:', grafico_fase2(tab2))
        print('tabela:', os.path.join(AQUI, 'resumo_fase2.csv'))
