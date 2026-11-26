"""Execução paralela de comandos independentes (treino.py, dataset.py) com barra de progresso.

Cada execução grava na sua própria pasta e só cria `concluido.json` ao terminar, então
rodá-las em processos simultâneos não altera nenhum resultado, e pastas já concluídas são
puladas na retomada. Os processos filhos usam 1 thread (OMP/MKL/OPENBLAS = 1): `workers`
equivale ao número de CPUs usadas.

A barra avança por episódio, lido da saída dos filhos (linhas JSON {"avancou": k}).
"""
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

PROGRESSO = re.compile(r'^\{"avancou":\s*(\d+)')


@dataclass
class Job:
    name: str
    steps: list                 # comandos (listas de str) executados em sequência
    units: int                  # unidades de progresso (episódios) que o job produzirá
    folder: Path = None         # onde gravar stdout.log (opcional)
    tail: list = field(default_factory=list)


def arg(cmd, flag):
    """Valor de `flag` em uma lista de argumentos (ex.: '--out')."""
    return cmd[cmd.index(flag) + 1]


def job_from_command(cmd, units, name=None):
    """Job a partir de um comando com --out. Retorna None se concluido.json já existe."""
    folder = Path(arg(cmd, '--out'))
    if (folder / 'concluido.json').exists():
        return None
    return Job(name or folder.name, [cmd], units, folder)


class _Bar:
    """tqdm se existir; caso contrário, uma linha de texto simples."""

    def __init__(self, total, desc, initial=0):
        self.total, self.n, self.t0 = total, initial, time.time()
        self.desc, self.postfix, self._last = desc, '', 0.
        try:
            from tqdm.auto import tqdm
            self.bar = tqdm(total=total, initial=initial, desc=desc, unit='ep', dynamic_ncols=True)
        except ImportError:
            self.bar = None
            self._print(force=True)

    def _print(self, force=False):
        now = time.time()
        if not force and now - self._last < 1.:
            return
        self._last = now
        pct = 100 * self.n / self.total if self.total else 100
        print(f'\r{self.desc}: {self.n}/{self.total} ({pct:.0f}%) {self.postfix}   ', end='', flush=True)

    def update(self, k=1):
        self.n += k
        if self.bar:
            self.bar.update(k)
        else:
            self._print()

    def set_postfix(self, text):
        self.postfix = text
        if self.bar:
            self.bar.set_postfix_str(text, refresh=False)

    def write(self, text):
        if self.bar:
            self.bar.write(text)
        else:
            print('\r' + text + ' ' * 20)

    def close(self):
        if self.bar:
            self.bar.close()
        else:
            self._print(force=True)
            print()


def run_jobs(jobs, workers=1, desc='Execução', raise_on_error=True):
    """Executa `jobs` com até `workers` processos simultâneos.

    Uma falha não interrompe os demais; ao final as falhas são listadas (com o fim da saída
    de cada uma) e, se raise_on_error, levanta RuntimeError. Ctrl+C encerra os filhos.
    """
    jobs = [j for j in jobs if j is not None]
    if not jobs:
        print('Nada a executar: todas as execuções já estão concluídas.')
        return {'ok': 0, 'failed': []}
    workers = max(1, min(int(workers), len(jobs)))
    total = sum(j.units for j in jobs)
    bar = _Bar(total, desc)
    lock, stop, live = threading.Lock(), threading.Event(), set()
    state = {'done': 0, 'failed': []}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1')

    def refresh():
        bar.set_postfix(f"jobs {state['done']}/{len(jobs)} | ativos {len(live)} | falhas {len(state['failed'])}")

    def work(job):
        log = None
        if job.folder is not None:
            job.folder.mkdir(parents=True, exist_ok=True)
            log = open(job.folder / 'stdout.log', 'a', encoding='utf-8')
        try:
            for step in job.steps:
                if stop.is_set():
                    return job, -1
                if log:
                    log.write('$ ' + ' '.join(map(str, step)) + '\n'); log.flush()
                proc = subprocess.Popen([str(x) for x in step], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env, encoding='utf-8', errors='replace')
                with lock:
                    live.add(proc); refresh()
                for line in proc.stdout:
                    if log:
                        log.write(line); log.flush()
                    job.tail.append(line.rstrip()); del job.tail[:-25]
                    m = PROGRESSO.match(line)
                    if m:
                        with lock:
                            bar.update(int(m.group(1)))
                proc.stdout.close(); rc = proc.wait()
                with lock:
                    live.discard(proc); refresh()
                if rc != 0:
                    return job, rc
            return job, 0
        finally:
            if log:
                log.close()

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(work, j) for j in jobs]
    try:
        for fut in as_completed(futures):
            job, rc = fut.result()
            with lock:
                if rc == 0:
                    state['done'] += 1
                else:
                    state['failed'].append((job.name, rc, list(job.tail)))
                    bar.write(f'[falhou rc={rc}] {job.name}')
                refresh()
    except BaseException:
        stop.set()
        with lock:
            for proc in list(live):
                proc.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        bar.close()
        raise
    pool.shutdown()
    bar.close()
    for name, rc, tail in state['failed']:
        print(f'\n=== {name} (código {rc}) — fim da saída ===')
        print('\n'.join(tail[-15:]))
    summary = {'ok': state['done'], 'failed': [n for n, _, _ in state['failed']]}
    if state['failed'] and raise_on_error:
        raise RuntimeError(f"{len(state['failed'])} execução(ões) falharam: {summary['failed']}")
    return summary
