"""Análise de sensibilidade da dinâmica de fadiga."""
import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean

from executar_grade import RAIZ, jobs_grade
from paralelo import run_jobs

SEMENTES = [11, 22, 33, 44, 55]
CENARIOS = {
    "referencia": [],
    "fadiga_lenta": ["simulador.fadiga_por_min=0.0045"],
    "fadiga_rapida": ["simulador.fadiga_por_min=0.0075"],
    "recuperacao_lenta": ["simulador.recuperacao_por_min_pausa=0.02625"],
    "recuperacao_rapida": ["simulador.recuperacao_por_min_pausa=0.04375"],
}


def resultados(pasta):
    linhas = []
    for arquivo in sorted(pasta.glob("*/*/concluido.json")):
        dado = json.loads(arquivo.read_text(encoding="utf-8"))
        m = dado["metricas"]
        ciclos = dado["config"]["simulador"]["ciclos"]
        duracao = sum(fracao * sum(ciclos[a]) for a, fracao in m["uso_acoes"].items())
        tempo = duracao * dado["config"]["simulador"]["n_ciclos_por_episodio"]
        linhas.append(dict(cenario=arquivo.parents[1].name, algoritmo=dado["algoritmo"],
                           semente=dado["semente"], efetivos_por_hora=60*m["minutos_efetivos_por_episodio"]/tempo,
                           fracao_cansaco_alto=m["freq_cansaco_alto"]))
    with (pasta / "execucoes.csv").open("w", newline="", encoding="utf-8") as saida:
        escritor = csv.DictWriter(saida, fieldnames=linhas[0].keys())
        escritor.writeheader()
        escritor.writerows(linhas)
    import matplotlib.pyplot as plt
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for eixo, metrica, titulo in zip(eixos, ["efetivos_por_hora", "fracao_cansaco_alto"],
                                    ["Minutos efetivos por hora", "Fração de ciclos com cansaço alto"]):
        for alg in sorted({r["algoritmo"] for r in linhas}):
            valores = [[r[metrica] for r in linhas if r["algoritmo"] == alg and r["cenario"] == c]
                       for c in CENARIOS]
            eixo.plot(list(CENARIOS), [mean(v) for v in valores], "o-", label=alg)
        eixo.set_ylabel(titulo)
        eixo.tick_params(axis="x", rotation=30)
    eixos[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pasta / "sensibilidade.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpus", type=int, default=28)
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    pasta = RAIZ / "experiments" / ("sensibilidade_rapida" if a.rapido else "sensibilidade")
    jobs = []
    total = 0
    for nome, ajustes in CENARIOS.items():
        pendentes, quantidade = jobs_grade(RAIZ / "config/padrao.json", pasta / nome,
            variantes=["eficaz"], penalidades=[10], sementes=SEMENTES, sets=ajustes, rapido=a.rapido)
        jobs.extend(pendentes)
        total += quantidade
    print(f"{len(jobs)} pendentes de {total} execuções")
    if a.dry_run:
        return
    for chave in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
        os.environ[chave] = "1"
    run_jobs(jobs, workers=max(1, min(a.cpus, os.cpu_count() or 1)), desc="Sensibilidade")
    resultados(pasta)


if __name__ == "__main__":
    main()
