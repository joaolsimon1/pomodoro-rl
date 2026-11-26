"""Uma execução independente: treina (se o agente aprende), avalia e grava concluido.json.

Progresso: linhas JSON {"avancou": k} na saída padrão (k = episódios concluídos desde a
última linha). paralelo.py soma esses valores na barra. Total = unidades(cfg, algoritmo).

Sementes: ambiente de treino, ambiente de avaliação e agente usam geradores separados
derivados de --semente; a avaliação usa o mesmo fluxo de sorteio para todos os algoritmos
com a mesma semente (comparação mais justa).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from agentes import APRENDEM, TODOS, criar_agente
from config import ACOES, URGENCIAS, montar
from simulador import SimuladorPomodoro


def unidades(cfg, algoritmo):
    t = cfg['treino']
    return t['n_episodios_avaliacao'] + (t['n_episodios'] if algoritmo in APRENDEM else 0)


def emitir_stdout(k):
    print(json.dumps({'avancou': int(k)}), flush=True)


class Progresso:
    """Agrega episódios e emite no máximo `blocos` linhas; flush() garante o total exato."""

    def __init__(self, total, blocos, emitir):
        self.passo, self.emitir, self.pend = max(1, total // max(1, blocos)), emitir, 0

    def add(self, k=1):
        self.pend += k
        if self.pend >= self.passo:
            self.flush()

    def flush(self):
        if self.pend:
            self.emitir(self.pend)
            self.pend = 0


def rodar_episodio(env, agente, eps, aprender):
    obs, ret, passos = env.reset(), 0.0, []
    feito = False
    while not feito:
        a = agente.escolher(obs, eps)
        obs2, r, feito, info = env.step(a)
        if aprender:
            agente.atualizar(obs, a, r, obs2, feito)
        passos.append((a, r, info))
        ret += r
        obs = obs2
    return ret, passos


def treinar(cfg, agente, env, semente, emitir):
    t, a = cfg['treino'], cfg['agente']
    n = t['n_episodios']
    prog = Progresso(n, t['blocos_progresso'], emitir)
    curva, acum, cont = [], 0.0, 0
    passo_bloco = max(1, n // t['blocos_progresso'])
    for ep in range(n):
        frac = min(1.0, ep / max(1.0, a['eps_decai_frac'] * n))
        eps = a['eps_inicio'] + (a['eps_fim'] - a['eps_inicio']) * frac
        ret, _ = rodar_episodio(env, agente, eps, True)
        acum, cont = acum + ret, cont + 1
        if cont >= passo_bloco:
            curva.append({'episodio': ep + 1, 'recompensa_media': acum / cont, 'eps': eps})
            acum, cont = 0.0, 0
        prog.add()
    prog.flush()
    return curva


def avaliar(cfg, agente, env, emitir):
    """Política gulosa. Métricas independentes da variante de recompensa (comparáveis)."""
    n = cfg['treino']['n_episodios_avaliacao']
    prog = Progresso(n, cfg['treino']['blocos_progresso'], emitir)
    A, MIN, EF, ALTO, URG, RET, R = [], [], [], [], [], [], []
    for _ in range(n):
        ret, passos = rodar_episodio(env, agente, 0.0, False)
        RET.append(ret)
        for a, r, i in passos:
            A.append(a); MIN.append(i['minutos']); EF.append(i['minutos_efetivos'])
            ALTO.append(i['cansaco_alto']); URG.append(i['urgencia']); R.append(r)
        prog.add()
    prog.flush()
    A, MIN, EF, ALTO, URG = map(np.asarray, (A, MIN, EF, ALTO, URG))
    metricas = {
        'minutos_por_episodio': float(MIN.sum() / n),
        'minutos_efetivos_por_episodio': float(EF.sum() / n),
        'freq_cansaco_alto': float(ALTO.mean()),
        'recompensa_por_episodio': float(np.mean(RET)),   # na recompensa DESTA execução; não comparar entre variantes
        'uso_acoes': {nm: float((A == j).mean()) for j, nm in enumerate(ACOES)},
        'por_urgencia': {},
    }
    for u, nome in enumerate(URGENCIAS):
        m = URG == u
        if m.any():
            metricas['por_urgencia'][nome] = {
                'minutos_por_ciclo': float(MIN[m].mean()),
                'minutos_efetivos_por_ciclo': float(EF[m].mean()),
                'freq_cansaco_alto': float(ALTO[m].mean()),
                'uso_acoes': {nm: float((A[m] == j).mean()) for j, nm in enumerate(ACOES)},
                'n_ciclos': int(m.sum())}
    return metricas


def executar(cfg, algoritmo, semente, out, emitir=emitir_stdout):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env_tr = SimuladorPomodoro(cfg, np.random.default_rng([semente, 0]))
    env_ev = SimuladorPomodoro(cfg, np.random.default_rng([semente, 1]))
    agente = criar_agente(algoritmo, cfg, np.random.default_rng([semente, 2]))
    curva = treinar(cfg, agente, env_tr, semente, emitir) if algoritmo in APRENDEM else []
    metricas = avaliar(cfg, agente, env_ev, emitir)
    res = {'nome': out.name, 'algoritmo': algoritmo, 'semente': semente,
           'variante': cfg['recompensa'].get('variante'), 'penalidade': cfg['recompensa']['penalidade'],
           'config': cfg, 'metricas': metricas, 'curva': curva, 'tempo_s': round(time.time() - t0, 2)}
    if algoritmo in APRENDEM:
        res['politica'] = agente.tabela_politica()      # linhas: cansaço 1..5; colunas: urgência
        res['Q'] = agente.Q.tolist()
    tmp = out / 'concluido.tmp'
    tmp.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, out / 'concluido.json')           # gravação atômica: só existe se terminou
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'config/padrao.json'))
    p.add_argument('--out', required=True)
    p.add_argument('--algoritmo', choices=TODOS, default='qlearning')
    p.add_argument('--variante', default=None)
    p.add_argument('--penalidade', type=float, default=None)
    p.add_argument('--semente', type=int, default=11)
    p.add_argument('--set', dest='sets', action='append', default=[], metavar='CHAVE=VALOR',
                   help='sobrescreve parâmetro, ex.: --set simulador.ruido_autorrelato=0')
    p.add_argument('--rapido', action='store_true', help='poucos episódios (teste de fumaça)')
    a = p.parse_args(argv)
    cfg = montar(a.config, a.variante, a.penalidade, a.sets, a.rapido)
    if (Path(a.out) / 'concluido.json').exists():
        print('já concluído:', a.out)
        return 0
    executar(cfg, a.algoritmo, a.semente, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
