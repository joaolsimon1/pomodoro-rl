"""Gera o dataset simulado de transições (s, a, r, s') com uma política de comportamento.

Cada linha é um ciclo. Por padrão a política é `aleatorio` (exploração uniforme), que dá
cobertura de todos os pares (estado, ação) e permite RL offline depois. As colunas
`fadiga_latente*` existem só porque o dado é simulado; um dataset real não as teria.
Progresso: {"avancou": k} por bloco de episódios (ver paralelo.py).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agentes import SEM_APRENDIZADO, criar_agente
from config import ACOES, URGENCIAS, montar
from simulador import SimuladorPomodoro
from treino import Progresso, emitir_stdout


def gerar(cfg, politica, n_episodios, semente, emitir=emitir_stdout, ep0=0):
    env = SimuladorPomodoro(cfg, np.random.default_rng([semente, 10]))
    agente = criar_agente(politica, cfg, np.random.default_rng([semente, 11]))
    prog = Progresso(n_episodios, cfg['treino']['blocos_progresso'], emitir)
    linhas = []
    for ep in range(n_episodios):
        obs, feito, ciclo = env.reset(), False, 0
        while not feito:
            F0 = env.F
            a = agente.escolher(obs)
            obs2, r, feito, i = env.step(a)
            linhas.append((ep0 + ep, ciclo, URGENCIAS[obs[1]], obs[0], ACOES[a], i['minutos'],
                           round(i['minutos_efetivos'], 3), round(r, 3), obs2[0], bool(i['cansaco_alto']),
                           bool(feito), round(F0, 4), round(i['fadiga'], 4)))
            obs, ciclo = obs2, ciclo + 1
        prog.add()
    prog.flush()
    colunas = ['episodio', 'ciclo', 'urgencia', 'cansaco', 'acao', 'minutos_estudo', 'minutos_efetivos',
               'recompensa', 'cansaco_seguinte', 'cansaco_alto_seguinte', 'terminal',
               'fadiga_latente', 'fadiga_latente_seguinte']
    return pd.DataFrame(linhas, columns=colunas)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'config/padrao.json'))
    p.add_argument('--out', required=True)
    p.add_argument('--politica', choices=SEM_APRENDIZADO, default='aleatorio')
    p.add_argument('--episodios', type=int, default=10000)
    p.add_argument('--semente', type=int, default=11)
    p.add_argument('--ep0', type=int, default=0, help='numeração inicial dos episódios (para shards)')
    p.add_argument('--variante', default=None)
    p.add_argument('--penalidade', type=float, default=None)
    p.add_argument('--set', dest='sets', action='append', default=[], metavar='CHAVE=VALOR')
    a = p.parse_args(argv)
    out = Path(a.out)
    if (out / 'concluido.json').exists():
        print('já concluído:', a.out)
        return 0
    cfg = montar(a.config, a.variante, a.penalidade, a.sets)
    out.mkdir(parents=True, exist_ok=True)
    df = gerar(cfg, a.politica, a.episodios, a.semente, ep0=a.ep0)
    df.to_csv(out / 'dataset.csv', index=False)
    (out / 'concluido.tmp').write_text(json.dumps({'linhas': len(df), 'politica': a.politica, 'semente': a.semente,
                                                   'episodios': a.episodios, 'config': cfg}), encoding='utf-8')
    os.replace(out / 'concluido.tmp', out / 'concluido.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
