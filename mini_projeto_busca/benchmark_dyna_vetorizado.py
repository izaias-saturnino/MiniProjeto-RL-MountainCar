"""Dyna-Q com planejamento vetorizado (numpy) vs. o Dyna-Q original (laço em Python).

Objetivo: verificar se a diferença de tempo de relógio entre SARSA(λ) e Dyna-Q vem da
implementação. O SARSA(λ) atualiza todos os traços ativos numa única operação numpy, enquanto
o planejamento do Dyna-Q faz N atualizações num laço em Python.

Diferença de algoritmo (importante): no original, as N atualizações de planejamento são
sequenciais (cada uma vê os valores Q deixados pela anterior). Na versão vetorizada, os N alvos
são calculados com a MESMA tabela Q e aplicados juntos (atualização síncrona). Além disso, a
amostragem do planejamento passa a usar o mesmo gerador aleatório do agente. Por isso o
desempenho também é comparado, e não só o tempo.

O modelo continua o mesmo: contagens de todos os desfechos (r, s', terminou) de cada par (s,a),
com amostragem proporcional às contagens ('amostral') ou o último desfecho ('ultimo'). O
dicionário original do modelo é mantido, para `estatisticas_modelo()` e para o custo de atualizar
o modelo ser igual nas duas versões.

Uso:
    python benchmark_dyna_vetorizado.py      # roda as 4 configurações finais do notebook (10 seeds)
"""
import contextlib
import io
import os
import sys

import numpy as np

import busca

NOME = 'Dyna-Q (vetorizado)'


def criar_classe(ns):
    base = ns['AgenteDynaQ']

    class AgenteDynaQVetorizado(base):
        nome = NOME

        def __init__(self, n_estados, n_acoes, **kwargs):
            super().__init__(n_estados, n_acoes, **kwargs)
            n_pares_max = n_estados * n_acoes
            self._K = 8                                            # desfechos por par (cresce)
            self._par_id = np.full(n_pares_max, -1, dtype=np.int64)
            self._par_s = np.zeros(n_pares_max, dtype=np.int64)
            self._par_a = np.zeros(n_pares_max, dtype=np.int64)
            self._n_pares = 0
            self._d_s2 = np.zeros((n_pares_max, self._K), dtype=np.int64)
            self._d_r = np.zeros((n_pares_max, self._K))
            self._d_term = np.zeros((n_pares_max, self._K), dtype=bool)
            self._d_cont = np.zeros((n_pares_max, self._K))
            self._n_desf = np.zeros(n_pares_max, dtype=np.int64)
            self._ultimo_k = np.zeros(n_pares_max, dtype=np.int64)
            self._indice_k = {}                                    # (par, desfecho) -> coluna

        def _crescer(self):
            extra = self._K
            self._d_s2 = np.pad(self._d_s2, ((0, 0), (0, extra)))
            self._d_r = np.pad(self._d_r, ((0, 0), (0, extra)))
            self._d_term = np.pad(self._d_term, ((0, 0), (0, extra)))
            self._d_cont = np.pad(self._d_cont, ((0, 0), (0, extra)))
            self._K += extra

        def _atualizar_modelo(self, s, a, r, s2, terminou):
            super()._atualizar_modelo(s, a, r, s2, terminou)       # dicionário original (estatísticas)
            plano = s * self.n_acoes + a
            p = self._par_id[plano]
            if p < 0:
                p = self._n_pares
                self._par_id[plano] = p
                self._par_s[p], self._par_a[p] = s, a
                self._n_pares += 1
            chave = (int(p), float(r), int(s2), bool(terminou))
            k = self._indice_k.get(chave)
            if k is None:
                k = int(self._n_desf[p])
                if k == self._K:
                    self._crescer()
                self._d_s2[p, k], self._d_r[p, k], self._d_term[p, k] = s2, r, terminou
                self._n_desf[p] += 1
                self._indice_k[chave] = k
            self._d_cont[p, k] += 1
            self._ultimo_k[p] = k

        def planejar(self, n):
            pares = self.rng.integers(self._n_pares, size=n)
            if self.tipo_modelo == 'amostral':
                acum = np.cumsum(self._d_cont[pares], axis=1)
                u = self.rng.random(n) * acum[:, -1]
                k = (acum > u[:, None]).argmax(axis=1)
            else:
                k = self._ultimo_k[pares]
            s2, r, term = self._d_s2[pares, k], self._d_r[pares, k], self._d_term[pares, k]
            S, A = self._par_s[pares], self._par_a[pares]
            alvo = r + self.gamma * np.where(term, 0.0, self.q_values[s2].max(axis=1))
            delta = alvo - self.q_values[S, A]
            np.add.at(self.q_values, (S, A), self.alpha * delta)   # aplica as N atualizações juntas
            self.n_atualizacoes += n
            self.n_atualizacoes_planejamento += n

    return AgenteDynaQVetorizado


def main():
    busca.prioridade_baixa()
    with contextlib.redirect_stdout(io.StringIO()):
        ns = busca.carregar_codigo_notebook()                     # CONFIG final do notebook (4 grades)
    ns['ALGORITMOS'][NOME] = criar_classe(ns)
    col = 'passos_medios_ult50'
    print(f"{'grade':17s} {'versão':20s} {'final':>13s} {'média todos':>13s} {'atualiz. Q':>10s} {'tempo/seed':>10s}")
    for disc in ns['DISCRETIZACOES']:
        params = ns['HIPERPARAMETROS'][disc]['Dyna-Q']
        for nome in ('Dyna-Q', NOME):
            pac = ns['executar_varias_seeds'](nome, disc, params, pasta=busca.PASTA, reutilizar=True,
                                              rotulo=nome, verbose=False)
            met = ns['metricas_por_execucao'](pac['df'], 50)
            m, h = ns['media_ic'](met[col])
            mt, ht = ns['media_ic'](met['passos_medios_todos'])
            print(f"{disc:17s} {nome:20s} {m:6.1f} ±{h:4.1f}  {mt:6.1f} ±{ht:4.1f} "
                  f"{met['atualizacoes_q_total'].mean() / 1e6:8.2f} M {met['tempo_s'].mean():8.1f} s", flush=True)


if __name__ == '__main__':
    sys.exit(main())
