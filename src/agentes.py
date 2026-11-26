"""Agentes tabulares e políticas de referência.

* qlearning: Q-learning tabular (15 estados x 3 ações), com desconto gamma.
* bandit:    bandit contextual = mesmo código com gamma=0 (só a recompensa imediata).
             Serve de baseline para saber se o MDP (efeito de longo prazo) realmente ajuda.
* fixo_*:    sempre o mesmo ciclo (curto/padrao/foco).
* regra:     heurística simples por cansaço relatado.
* aleatorio: ação uniforme (também usada para gerar o dataset simulado).
"""
import numpy as np

from config import ACOES
from simulador import indice_estado, SimuladorPomodoro

APRENDEM = ('qlearning', 'bandit')
SEM_APRENDIZADO = ('fixo_curto', 'fixo_padrao', 'fixo_foco', 'regra', 'aleatorio')
TODOS = APRENDEM + SEM_APRENDIZADO


class AgenteTabular:
    aprende = True

    def __init__(self, cfg, rng, gamma=None):
        a = cfg['agente']
        self.alpha = a['alpha']
        self.gamma = a['gamma'] if gamma is None else gamma
        self.rng = rng
        self.Q = np.full((SimuladorPomodoro.N_ESTADOS, SimuladorPomodoro.N_ACOES), float(a['q_inicial']))

    def gulosa(self, s):
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))   # desempate aleatório

    def escolher(self, obs, eps=0.0):
        s = indice_estado(obs)
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(self.Q.shape[1]))
        return self.gulosa(s)

    def atualizar(self, obs, a, r, obs2, feito):
        s, s2 = indice_estado(obs), indice_estado(obs2)
        alvo = r if feito else r + self.gamma * self.Q[s2].max()   # terminal: sem bootstrap
        self.Q[s, a] += self.alpha * (alvo - self.Q[s, a])

    def tabela_politica(self):
        """Ação gulosa para cada (cansaço 1..5) x (urgência) -- sem desempate aleatório."""
        return [[ACOES[int(np.argmax(self.Q[(c - 1) * 3 + u]))] for u in range(3)] for c in range(1, 6)]


class PoliticaFixa:
    aprende = False

    def __init__(self, acao):
        self.acao = ACOES.index(acao)

    def escolher(self, obs, eps=0.0):
        return self.acao


class Regra:
    aprende = False

    def __init__(self, cfg):
        self.alto, self.baixo = cfg['regra']['curto_a_partir_de'], cfg['regra']['foco_ate']

    def escolher(self, obs, eps=0.0):
        if obs[0] >= self.alto:
            return ACOES.index('curto')
        return ACOES.index('foco') if obs[0] <= self.baixo else ACOES.index('padrao')


class Aleatorio:
    aprende = False

    def __init__(self, rng):
        self.rng = rng

    def escolher(self, obs, eps=0.0):
        return int(self.rng.integers(len(ACOES)))


def criar_agente(nome, cfg, rng):
    if nome == 'qlearning':
        return AgenteTabular(cfg, rng)
    if nome == 'bandit':
        return AgenteTabular(cfg, rng, gamma=0.0)
    if nome.startswith('fixo_') and nome[5:] in ACOES:
        return PoliticaFixa(nome[5:])
    if nome == 'regra':
        return Regra(cfg)
    if nome == 'aleatorio':
        return Aleatorio(rng)
    raise ValueError(f'Algoritmo desconhecido: {nome}. Opções: {TODOS}')
