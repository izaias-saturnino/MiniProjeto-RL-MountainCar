"""Busca de hiperparâmetros do mini-projeto (MountainCar-v0, agentes tabulares).

Reaproveita o código do notebook do mini-projeto (células de imports, CONFIG,
discretização, agentes e executor) e roda as variações "um fator por vez" em
relação aos valores padrão abaixo, com 10 seeds e 500 episódios por execução.

Os resultados são salvos na pasta `resultados/` ao lado do notebook (mesmo cache
do notebook): se a configuração final escolhida for uma das testadas aqui, o
notebook apenas carrega os resultados em vez de treinar de novo.

Uso:
    python busca.py --listar             # mostra as execuções planejadas
    python busca.py --parte 0 --de 2     # roda a metade 0 (para rodar em 2 processos)
    python busca.py --parte 1 --de 2
    python resumo.py                     # tabela-resumo + gráficos da busca
"""
import argparse
import glob
import os
import sys
import time

import nbformat
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
NOTEBOOK = glob.glob(os.path.join(RAIZ, 'Mini-Projeto_*Colab.ipynb'))[0]
PASTA = os.path.join(RAIZ, 'resultados')

# ---------------------------------------------------------------------------
# Valores padrão e variações (um fator por vez)
# ---------------------------------------------------------------------------
PADRAO = dict(alpha=0.1, gamma=1.0, epsilon_inicial=0.5, epsilon_final=0.01,
              decaimento_epsilon=0.99, tipo_decaimento='exponencial', q_inicial=0.0)
PADRAO_ALG = {
    'Q-learning': {},
    'SARSA(λ)': dict(lambda_=0.0, tipo_traco='substituicao'),   # λ=0: SARSA de 1 passo (referência)
    'Dyna-Q': dict(n_planejamento=10, tipo_modelo='amostral'),
}
TODOS = list(PADRAO_ALG)

# (rótulo, mudanças em relação ao padrão, algoritmos afetados)
VARIACOES = [
    ('padrao', {}, TODOS),
    ('gamma=0.9', dict(gamma=0.9), TODOS),
    ('q_inicial=-100', dict(q_inicial=-100.0), TODOS),
    ('q_inicial=-300', dict(q_inicial=-300.0), TODOS),
    ('eps=0.1', dict(epsilon_inicial=0.1), TODOS),
    ('eps=0.1,decay=0.95', dict(epsilon_inicial=0.1, decaimento_epsilon=0.95), TODOS),
    ('decay=0.95', dict(decaimento_epsilon=0.95), TODOS),
    ('alpha=0.3', dict(alpha=0.3), TODOS),
    ('lambda=0.7', dict(lambda_=0.7), ['SARSA(λ)']),
    ('lambda=0.9', dict(lambda_=0.9), ['SARSA(λ)']),
    ('lambda=0.99', dict(lambda_=0.99), ['SARSA(λ)']),
    ('N=20', dict(n_planejamento=20), ['Dyna-Q']),
    ('modelo=ultimo', dict(tipo_modelo='ultimo'), ['Dyna-Q']),
    # acrescentadas depois da primeira rodada
    ('decay=0.9', dict(decaimento_epsilon=0.9), TODOS),
    ('lambda=0.7,traco=acumulativo', dict(lambda_=0.7, tipo_traco='acumulativo'), ['SARSA(λ)']),
    ('lambda=0.9,traco=acumulativo', dict(lambda_=0.9, tipo_traco='acumulativo'), ['SARSA(λ)']),
    ('lambda=0.99,traco=acumulativo', dict(lambda_=0.99, tipo_traco='acumulativo'), ['SARSA(λ)']),
    # matriz Q inicial × γ e Q inicial × exploração (as células com Q inicial = 0 já existem acima)
    ('gamma=0.9,q_inicial=-100', dict(gamma=0.9, q_inicial=-100.0), TODOS),
    ('gamma=0.9,q_inicial=-300', dict(gamma=0.9, q_inicial=-300.0), TODOS),
    ('eps=0.1,q_inicial=-100', dict(epsilon_inicial=0.1, q_inicial=-100.0), TODOS),
    ('eps=0.1,q_inicial=-300', dict(epsilon_inicial=0.1, q_inicial=-300.0), TODOS),
    ('decay=0.95,q_inicial=-100', dict(decaimento_epsilon=0.95, q_inicial=-100.0), TODOS),
    ('decay=0.95,q_inicial=-300', dict(decaimento_epsilon=0.95, q_inicial=-300.0), TODOS),
    ('eps=0.1,decay=0.95,q_inicial=-100',
     dict(epsilon_inicial=0.1, decaimento_epsilon=0.95, q_inicial=-100.0), TODOS),
    ('eps=0.1,decay=0.95,q_inicial=-300',
     dict(epsilon_inicial=0.1, decaimento_epsilon=0.95, q_inicial=-300.0), TODOS),
    ('alpha=0.05', dict(alpha=0.05), TODOS),
]


def _uniformes(low, high, n):
    """Bordas internas de n intervalos iguais em [low, high]."""
    return [round(float(v), 6) for v in np.linspace(low, high, n + 1)[1:-1]]


# Discretizações da busca. Ficam definidas aqui (e não lidas da célula CONFIG do notebook) para a
# busca continuar reproduzível mesmo depois que a CONFIG passou a ter as grades finais.
# Posição em [-1.2, 0.6] (o objetivo é atingido em 0.5); velocidade em [-0.07, 0.07].
# Nas duas grades-base rodam todas as variações; nas extras, só a configuração padrão.
DISCRETIZACOES_BASE = {
    'baixa_resolucao': {'bins': (10, 10)},
    'alta_resolucao': {'bins': (40, 40)},
}
# As duas não uniformes têm ~100 células, como a 10x10, para comparar *onde* ficam as
# células e não só quantas são.
DISCRETIZACOES_EXTRAS = {
    'media_resolucao': {'bins': (20, 20)},
    # posição detalhada perto da parede da esquerda, do fundo do vale (~-0.52) e do
    # objetivo; velocidade em baixa resolução
    'pos_detalhada': {'bins': (18, 6), 'bordas': [
        [-1.15, -1.1, -1.0, -0.85, -0.7, -0.6, -0.55, -0.5, -0.45, -0.4,
         -0.25, -0.05, 0.15, 0.3, 0.4, 0.45, 0.5],
        _uniformes(-0.07, 0.07, 6)]},
    # posição em baixa resolução; velocidade detalhada perto de 0 (onde o carro inverte o sentido)
    'vel_detalhada': {'bins': (6, 16), 'bordas': [
        _uniformes(-1.2, 0.6, 6),
        [-0.05, -0.035, -0.02, -0.012, -0.007, -0.004, -0.002, 0.0,
         0.002, 0.004, 0.007, 0.012, 0.02, 0.035, 0.05]]},
}
# Combinação das duas: posição detalhada + velocidade detalhada (18 x 16 = 288 células)
DISCRETIZACOES_EXTRAS['pos_vel_detalhada'] = {
    'bins': (18, 16),
    'bordas': [DISCRETIZACOES_EXTRAS['pos_detalhada']['bordas'][0],
               DISCRETIZACOES_EXTRAS['vel_detalhada']['bordas'][1]]}
# Nas discretizações extras roda só a configuração padrão (comparação de discretizações).
VARIACOES_EXTRAS = ['padrao']


def planejar(discretizacoes):
    """Lista de execuções (discretização, algoritmo, rótulo, params)."""
    tarefas = []
    for rotulo, mudancas, algs in VARIACOES:
        for nome_alg in algs:
            for nome_disc in discretizacoes:
                params = dict(PADRAO, **PADRAO_ALG[nome_alg])
                params.update(mudancas)
                tarefas.append((nome_disc, nome_alg, rotulo, params))
    for rotulo, mudancas, algs in VARIACOES:
        if rotulo not in VARIACOES_EXTRAS:
            continue
        for nome_alg in algs:
            for nome_disc in DISCRETIZACOES_EXTRAS:
                params = dict(PADRAO, **PADRAO_ALG[nome_alg])
                params.update(mudancas)
                tarefas.append((nome_disc, nome_alg, rotulo, params))
    return tarefas


def parte_de(i, n):
    """Divide as execuções entre n processos alternando as discretizações (carga equilibrada)."""
    return (i // 2 + i % 2) % n


def prioridade_baixa():
    """Roda abaixo do normal para não atrapalhar o resto da máquina."""
    if sys.platform == 'win32':
        import ctypes
        k = ctypes.windll.kernel32
        # sem restype/argtypes o ctypes trunca o handle no Python 64-bit
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    else:
        os.nice(10)


def registrar_discretizacoes(ns):
    """Registra no namespace do notebook todas as discretizações da busca (base + extras)."""
    for nome, cfg in {**DISCRETIZACOES_BASE, **DISCRETIZACOES_EXTRAS}.items():
        ns['DISCRETIZACOES'][nome] = cfg
        ns['DISCRETIZADORES'][nome] = ns['criar_discretizador'](cfg['bins'], nome=nome,
                                                                bordas=cfg.get('bordas'))
    return ns


def carregar_codigo_notebook():
    """Executa as células de código do notebook (até as funções da Tarefa C)."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns = {'__name__': 'notebook_mini_projeto'}
    os.chdir(RAIZ)
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        if cell.source.lstrip().startswith('# ====') and 'TAREFA C: Execução dos experimentos' in cell.source:
            break   # para antes de rodar os experimentos do próprio notebook
        exec(compile(cell.source, NOTEBOOK, 'exec'), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parte', type=int, default=0)
    ap.add_argument('--de', type=int, default=1)
    ap.add_argument('--listar', action='store_true')
    ap.add_argument('--rotulo', default=None, help='roda só as execuções com este rótulo')
    args = ap.parse_args()

    ns = registrar_discretizacoes(carregar_codigo_notebook())
    for nome in {**DISCRETIZACOES_BASE, **DISCRETIZACOES_EXTRAS}:
        print(ns['DISCRETIZADORES'][nome].descricao())
    tarefas = planejar(list(DISCRETIZACOES_BASE))
    minhas = [t for i, t in enumerate(tarefas) if parte_de(i, args.de) == args.parte]
    if args.rotulo:
        minhas = [t for t in tarefas if t[2] == args.rotulo]

    if args.listar:
        for i, (d, a, r, _) in enumerate(tarefas):
            print(f'{i:3d}  parte {parte_de(i, args.de)}  {d:7s} {a:11s} {r}')
        print(f'{len(tarefas)} execuções (cada uma: {len(ns["SEEDS"])} seeds x {ns["N_EPISODIOS"]} episódios)')
        return

    prioridade_baixa()
    print(f'parte {args.parte}/{args.de}: {len(minhas)} de {len(tarefas)} execuções -> {PASTA}', flush=True)
    t0 = time.time()
    for j, (nome_disc, nome_alg, rotulo, params) in enumerate(minhas, 1):
        t = time.time()
        ns['executar_varias_seeds'](nome_alg, nome_disc, params, pasta=PASTA, reutilizar=True,
                                    rotulo=rotulo, verbose=False)
        print(f'[{j}/{len(minhas)}] {nome_disc:7s} {nome_alg:11s} {rotulo:22s} '
              f'{(time.time() - t) / 60:5.1f} min  (total {(time.time() - t0) / 60:.1f} min)', flush=True)
    print('FIM', flush=True)


if __name__ == '__main__':
    main()
