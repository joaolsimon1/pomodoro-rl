import sys, unittest
from pathlib import Path
import numpy as np
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))
from config import montar
from agentes import criar_agente, AgenteTabular
from simulador import indice_estado

CFG = RAIZ / 'config/padrao.json'

class AgenteTests(unittest.TestCase):
    def setUp(self): self.cfg = montar(CFG)

    def test_atualizacao_q_e_terminal(self):
        ag = AgenteTabular(self.cfg, np.random.default_rng(0)); ag.alpha = 1.0
        s2 = indice_estado((3, 1)); ag.Q[s2] = [1., 5., 2.]
        ag.atualizar((1, 0), 0, 2.0, (3, 1), False)
        self.assertAlmostEqual(ag.Q[indice_estado((1, 0)), 0], 2.0 + 0.95 * 5.0)
        ag.atualizar((1, 0), 1, 2.0, (3, 1), True)               # terminal: sem bootstrap
        self.assertAlmostEqual(ag.Q[indice_estado((1, 0)), 1], 2.0)

    def test_bandit_ignora_futuro(self):
        ag = criar_agente('bandit', self.cfg, np.random.default_rng(0)); ag.alpha = 1.0
        ag.Q[indice_estado((3, 1))] = [9., 9., 9.]
        ag.atualizar((1, 0), 0, 2.0, (3, 1), False)
        self.assertAlmostEqual(ag.Q[indice_estado((1, 0)), 0], 2.0)

    def test_aprende_bandit_trivial(self):
        ag = AgenteTabular(self.cfg, np.random.default_rng(1)); rng = np.random.default_rng(2)
        for _ in range(3000):
            a = ag.escolher((2, 0), eps=0.5); r = [0., 1., 0.5][a] + rng.normal(0, .1)
            ag.atualizar((2, 0), a, r, (2, 0), True)
        self.assertEqual(ag.gulosa(indice_estado((2, 0))), 1)
        self.assertEqual(ag.tabela_politica()[1][0], 'padrao')

    def test_baselines(self):
        rng = np.random.default_rng(0)
        self.assertEqual(criar_agente('fixo_foco', self.cfg, rng).escolher((1, 0)), 2)
        regra = criar_agente('regra', self.cfg, rng)
        self.assertEqual([regra.escolher((c, 0)) for c in (1, 3, 5)], [2, 1, 0])
        with self.assertRaises(ValueError): criar_agente('xyz', self.cfg, rng)

if __name__ == '__main__': unittest.main()
