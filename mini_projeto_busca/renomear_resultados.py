"""Migra os resultados salvos para os novos nomes das discretizações.

grossa -> baixa_resolucao, media -> media_resolucao, fina -> alta_resolucao.
Renomeia os arquivos .pkl/.csv em ../resultados/ e corrige a coluna 'discretizacao' dentro deles.
Rodar uma vez, com a busca parada.
"""
import glob
import os
import pickle

import pandas as pd

PASTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resultados')
NOVOS = {'grossa': 'baixa_resolucao', 'media': 'media_resolucao', 'fina': 'alta_resolucao'}

n = 0
for antigo, novo in NOVOS.items():
    for caminho in glob.glob(os.path.join(PASTA, f'{antigo}__*')):
        destino = os.path.join(PASTA, novo + os.path.basename(caminho)[len(antigo):])
        if caminho.endswith('.pkl'):
            with open(caminho, 'rb') as f:
                pacote = pickle.load(f)
            pacote['df']['discretizacao'] = pacote['df']['discretizacao'].replace(NOVOS)
            if 'arquivo' in pacote:
                pacote['arquivo'] = os.path.splitext(destino)[0]
            with open(destino, 'wb') as f:
                pickle.dump(pacote, f)
        elif caminho.endswith('.csv'):
            df = pd.read_csv(caminho)
            if 'discretizacao' in df.columns:
                df['discretizacao'] = df['discretizacao'].replace(NOVOS)
            df.to_csv(destino, index=False)
        else:
            continue
        os.remove(caminho)
        n += 1
print(f'{n} arquivos migrados em {PASTA}')
