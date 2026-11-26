"""Executa a grade de experimentos ou gera o dataset simulado, em paralelo e com barra.

Exemplos (a partir da raiz do projeto):
  python src/executar_grade.py --modo grade --cpus 4
  python src/executar_grade.py --modo grade --cpus 8 --rapido --sementes 11 22
  python src/executar_grade.py --modo grade --cpus 4 --set simulador.ruido_autorrelato=0
  python src/executar_grade.py --modo grade --dry-run          # só conta execuções e episódios
  python src/executar_grade.py --modo dataset --cpus 4 --shards 8 --episodios 80000

--cpus = número de execuções simultâneas (1 CPU cada). A grade vem de config/padrao.json
(chave "grade") e pode ser sobrescrita pelas opções --algoritmos/--variantes/--penalidades/--sementes.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from config import montar
from paralelo import job_from_command, run_jobs
from treino import unidades

RAIZ = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent


def _fmt(x):
    return f'{x:g}'


def jobs_grade(config, saida, algoritmos=None, variantes=None, penalidades=None, sementes=None,
               baselines=None, sets=(), rapido=False):
    """Lista de Jobs pendentes (concluídos são pulados). Retorna (jobs, total_de_execucoes)."""
    base = montar(config, sets=sets, rapido=rapido)
    g = base['grade']
    algoritmos = g['algoritmos'] if algoritmos is None else algoritmos
    variantes = g['variantes'] if variantes is None else variantes
    penalidades = g['penalidades'] if penalidades is None else penalidades
    sementes = g['sementes'] if sementes is None else sementes
    baselines = g['baselines'] if baselines is None else baselines
    saida = Path(saida)
    comuns = ['--config', str(config), '--rapido'] if rapido else ['--config', str(config)]
    for s in sets:
        comuns += ['--set', s]
    execucoes = []
    for alg in algoritmos:
        for var in variantes:
            for pen in penalidades:
                for sem in sementes:
                    execucoes.append((alg, var, pen, sem, saida / f'{alg}_{var}_p{_fmt(pen)}_s{sem}'))
    for alg in baselines:
        for sem in sementes:
            execucoes.append((alg, None, None, sem, saida / f'{alg}_s{sem}'))
    jobs = []
    for alg, var, pen, sem, pasta in execucoes:
        cmd = [sys.executable, str(SRC / 'treino.py'), '--algoritmo', alg, '--semente', sem, '--out', pasta] + comuns
        if var:
            cmd += ['--variante', var]
        if pen is not None:
            cmd += ['--penalidade', pen]
        cfg_job = montar(config, var, pen, sets, rapido)
        jobs.append(job_from_command(cmd, unidades(cfg_job, alg)))
    return [j for j in jobs if j], len(execucoes)


def jobs_dataset(config, saida, shards, episodios, politica='aleatorio', semente=11, sets=(), rapido=False):
    cfg = montar(config, sets=sets, rapido=rapido)
    saida = Path(saida)
    por_shard = max(1, episodios // shards)
    jobs = []
    for k in range(shards):
        pasta = saida / f'shard_{k:03d}'
        cmd = [sys.executable, str(SRC / 'dataset.py'), '--config', str(config), '--out', pasta,
               '--politica', politica, '--episodios', por_shard, '--semente', semente + k,
               '--ep0', k * por_shard]
        for s in sets:
            cmd += ['--set', s]
        jobs.append(job_from_command(cmd, por_shard))
    return [j for j in jobs if j], shards


def consolidar_dataset(saida):
    partes = sorted(Path(saida).glob('shard_*/dataset.csv'))
    if not partes:
        return None
    df = pd.concat([pd.read_csv(p) for p in partes], ignore_index=True)
    destino = Path(saida) / 'dataset.csv'
    df.to_csv(destino, index=False)
    return destino, len(df)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modo', choices=['grade', 'dataset'], default='grade')
    p.add_argument('--config', default=str(RAIZ / 'config/padrao.json'))
    p.add_argument('--saida', default=None, help='padrão: experiments/grade ou data/dataset_simulado')
    p.add_argument('--cpus', type=int, default=1, help='execuções simultâneas (limitado ao nº de CPUs)')
    p.add_argument('--rapido', action='store_true', help='poucos episódios (teste de fumaça)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--set', dest='sets', action='append', default=[], metavar='CHAVE=VALOR')
    p.add_argument('--algoritmos', nargs='+'); p.add_argument('--variantes', nargs='+')
    p.add_argument('--penalidades', nargs='+', type=float); p.add_argument('--sementes', nargs='+', type=int)
    p.add_argument('--shards', type=int, default=4); p.add_argument('--episodios', type=int, default=20000)
    p.add_argument('--politica', default='aleatorio')
    a = p.parse_args(argv)
    total_cpus = os.cpu_count() or 1
    cpus = max(1, min(a.cpus, total_cpus))
    print(f'CPUs disponíveis: {total_cpus} | usando: {cpus}')
    if a.modo == 'grade':
        saida = a.saida or str(RAIZ / 'experiments/grade')
        jobs, total = jobs_grade(a.config, saida, a.algoritmos, a.variantes, a.penalidades, a.sementes,
                                 sets=a.sets, rapido=a.rapido)
        desc = 'Grade'
    else:
        saida = a.saida or str(RAIZ / 'data/dataset_simulado')
        jobs, total = jobs_dataset(a.config, saida, a.shards, a.episodios, a.politica,
                                   sets=a.sets, rapido=a.rapido)
        desc = 'Dataset'
    print(f'{len(jobs)} de {total} execuções pendentes, {sum(j.units for j in jobs)} episódios, {cpus} em paralelo')
    if a.dry_run:
        return 0
    run_jobs(jobs, workers=cpus, desc=desc)
    if a.modo == 'dataset':
        r = consolidar_dataset(saida)
        print('dataset consolidado:', r)
    else:
        print(f'Concluído. Analise com: python src/analise.py --pasta {saida}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
