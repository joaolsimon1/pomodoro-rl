"""Consolida concluido.json de uma grade: tabela por configuração (média e IC95% entre sementes)
e gráficos. Compara só métricas independentes da recompensa (minutos, minutos efetivos,
frequência de cansaço alto); a recompensa de variantes diferentes NÃO é comparável.

  python src/analise.py --pasta experiments/grade
  python src/analise.py --pasta experiments/grade --politica qlearning_eficaz_p10_s11
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICAS = ['minutos_por_episodio', 'minutos_efetivos_por_episodio', 'freq_cansaco_alto']


def carregar(pasta):
    linhas = []
    for f in sorted(Path(pasta).glob('*/concluido.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if 'metricas' not in r:
            continue
        aprende = r['algoritmo'] in ('qlearning', 'bandit')   # baselines não dependem da recompensa
        linha = {'nome': r['nome'], 'algoritmo': r['algoritmo'], 'variante': r.get('variante') if aprende else '-',
                 'penalidade': r['penalidade'] if aprende else np.nan, 'semente': r['semente']}
        linha.update({k: r['metricas'][k] for k in METRICAS + ['recompensa_por_episodio']})
        for u, d in r['metricas']['por_urgencia'].items():
            linha[f'alto_{u}'] = d['freq_cansaco_alto']
            linha[f'min_ef_{u}'] = d['minutos_efetivos_por_ciclo']
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela(df):
    chaves = ['algoritmo', 'variante', 'penalidade']
    g = df.groupby(chaves, dropna=False)
    saida = g.size().rename('n_sementes').to_frame()
    for m in METRICAS:
        media, desvio = g[m].mean(), g[m].std(ddof=1).fillna(0)
        saida[m] = media
        saida[m + '_ic95'] = 1.96 * desvio / np.sqrt(saida['n_sementes'])
    return saida.reset_index().sort_values('minutos_efetivos_por_episodio', ascending=False)


def graficos(pasta, df, tab):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in tab.iterrows():
        rot = r['algoritmo'] if r['variante'] == '-' else f"{r['algoritmo']}/{r['variante']}/p{r['penalidade']:g}"
        ax.errorbar(r['freq_cansaco_alto'], r['minutos_efetivos_por_episodio'],
                    xerr=r['freq_cansaco_alto_ic95'], yerr=r['minutos_efetivos_por_episodio_ic95'], fmt='o', ms=4)
        ax.annotate(rot, (r['freq_cansaco_alto'], r['minutos_efetivos_por_episodio']), fontsize=6)
    ax.set_xlabel('Frequência de cansaço alto (fração de ciclos)')
    ax.set_ylabel('Minutos efetivos por episódio')
    ax.set_title('Rendimento x exaustão (média ± IC95% entre sementes)')
    fig.tight_layout(); fig.savefig(Path(pasta) / 'pareto.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for f in sorted(Path(pasta).glob('qlearning_*/concluido.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r['curva']:
            ax.plot([c['episodio'] for c in r['curva']], [c['recompensa_media'] for c in r['curva']],
                    alpha=.4, lw=.8)
    ax.set_xlabel('Episódio de treino'); ax.set_ylabel('Recompensa média por episódio (de cada execução)')
    ax.set_title('Curvas de aprendizado do Q-learning')
    fig.tight_layout(); fig.savefig(Path(pasta) / 'curvas.png', dpi=180); plt.close(fig)


def mostrar_politica(pasta, nome):
    r = json.loads((Path(pasta) / nome / 'concluido.json').read_text(encoding='utf-8'))
    if 'politica' not in r:
        print(f'{nome} não aprende (sem tabela de política).')
        return
    print(f'Política gulosa de {nome} (linhas: cansaço 1-5; colunas: longe | semana | vespera)')
    for c, linha in enumerate(r['politica'], 1):
        print(f'  cansaço {c}: ' + ' | '.join(f'{x:6s}' for x in linha))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pasta', required=True)
    p.add_argument('--politica', default=None, help='nome da pasta de uma execução para imprimir a política')
    p.add_argument('--sem-graficos', action='store_true')
    a = p.parse_args(argv)
    if a.politica:
        mostrar_politica(a.pasta, a.politica)
        return 0
    df = carregar(a.pasta)
    if df.empty:
        print('Nenhum concluido.json encontrado em', a.pasta)
        return 1
    tab = tabela(df)
    pd.set_option('display.width', 200, 'display.max_columns', 20)
    print(tab.round(3).to_string(index=False))
    tab.to_csv(Path(a.pasta) / 'resumo.csv', index=False)
    df.to_csv(Path(a.pasta) / 'execucoes.csv', index=False)
    if not a.sem_graficos:
        graficos(a.pasta, df, tab)
        print('gráficos: pareto.png, curvas.png em', a.pasta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
