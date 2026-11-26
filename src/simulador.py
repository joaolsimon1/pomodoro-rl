"""Simulador de sessões de estudo (ambiente do MDP).

Fadiga LATENTE F em [0, 1] (não observada). O aluno só reporta um autorrelato ruidoso
de 1 a 5, então o problema é parcialmente observável (observação = (cansaço, urgência)).

Dinâmica de um ciclo (estudo E min, pausa P min):
  * Estudo: a cada minuto t, F sobe  fadiga_por_min * mult_urgencia * (1 + crescimento * t/25).
    A taxa cresce com o tempo contínuo no bloco, então blocos longos custam mais por minuto.
  * Pausa: F cai recuperacao_por_min_pausa * P (piso 0).
  * Eficiência por minuto: 1 - (1 - eficiencia_min) * F**expoente (estudar cansado rende menos).
  * Autorrelato: round(1 + 4F + ruído), limitado a [1, 5].

Recompensa: +pontos_por_min por minuto (opcional: minutos ponderados por eficiência) e
penalidade se o autorrelato seguinte for >= limiar_cansaco_alto (opcional: depende da urgência).

Todos os números são hipóteses editáveis em config/padrao.json, não medições de alunos reais.
"""
import numpy as np

from config import ACOES


def indice_estado(obs):
    """(cansaço 1..5, urgência 0..2) -> índice 0..14."""
    return (obs[0] - 1) * 3 + obs[1]


class SimuladorPomodoro:
    N_ESTADOS = 15
    N_ACOES = len(ACOES)

    def __init__(self, cfg, rng):
        self.s, self.r, self.rng = cfg['simulador'], cfg['recompensa'], rng
        self.estudo = [self.s['ciclos'][a][0] for a in ACOES]
        self.pausa = [self.s['ciclos'][a][1] for a in ACOES]
        self._t = [np.arange(e, dtype=float) for e in self.estudo]
        self.F = self.urgencia = self.ciclo = self.cansaco = 0

    def reportar(self, F):
        ruido = self.rng.normal(0.0, self.s['ruido_autorrelato']) if self.s['ruido_autorrelato'] > 0 else 0.0
        return int(np.clip(np.rint(1 + 4 * F + ruido), 1, 5))

    def reset(self, urgencia=None, fadiga=None):
        self.urgencia = int(self.rng.choice(3, p=self.s['urgencia_probs'])) if urgencia is None else int(urgencia)
        self.F = float(self.rng.uniform(*self.s['fadiga_inicial'])) if fadiga is None else float(fadiga)
        self.ciclo = 0
        self.cansaco = self.reportar(self.F)
        return (self.cansaco, self.urgencia)

    def step(self, acao):
        s, r = self.s, self.r
        E, P = self.estudo[acao], self.pausa[acao]
        taxa = s['fadiga_por_min'] * s['multiplicador_urgencia'][self.urgencia] \
            * (1 + s['crescimento_tempo_continuo'] * self._t[acao] / 25.0)
        caminho = np.minimum(self.F + np.cumsum(taxa), 1.0)
        eficiencia = 1 - (1 - s['eficiencia_min']) * caminho ** s['expoente_eficiencia']
        min_efetivos = float(eficiencia.sum())
        F_antes = self.F
        self.F = max(0.0, float(caminho[-1]) - s['recuperacao_por_min_pausa'] * P)
        self.cansaco = self.reportar(self.F)
        alto = self.cansaco >= s['limiar_cansaco_alto']
        pontos = r['pontos_por_min'] * (min_efetivos if r['ponderar_por_eficiencia'] else E)
        penal = r['penalidade'] * r['penalidade_por_urgencia'][self.urgencia] if alto else 0.0
        urg = self.urgencia
        self.ciclo += 1
        feito = self.ciclo >= s['n_ciclos_por_episodio']
        info = {'minutos': E, 'minutos_efetivos': min_efetivos, 'cansaco_alto': bool(alto),
                'penalidade': penal, 'fadiga_antes': F_antes, 'fadiga': self.F, 'urgencia': urg}
        return (self.cansaco, self.urgencia), pontos - penal, feito, info
