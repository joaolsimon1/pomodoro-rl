import sys, unittest
from pathlib import Path
import numpy as np
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))
from config import montar, definir, ACOES
from simulador import SimuladorPomodoro, indice_estado

CFG = RAIZ / 'config/padrao.json'

def env(sets=(), variante=None, seed=0):
    return SimuladorPomodoro(montar(CFG, variante=variante, sets=sets), np.random.default_rng(seed))

class SimuladorTests(unittest.TestCase):
    def test_limites_e_termino(self):
        e = env(); obs = e.reset(); n = 0; feito = False
        while not feito:
            obs, r, feito, info = e.step(n % 3); n += 1
            self.assertTrue(1 <= obs[0] <= 5 and 0 <= obs[1] <= 2 and 0 <= info['fadiga'] <= 1)
            self.assertTrue(0 <= indice_estado(obs) < 15)
        self.assertEqual(n, 8)

    def test_deterministico_por_semente(self):
        def traj(seed):
            e = env(seed=seed); e.reset(); return [e.step(a % 3)[1] for a in range(8)]
        self.assertEqual(traj(5), traj(5)); self.assertNotEqual(traj(5), traj(6))

    def test_foco_cansa_mais_que_curto(self):
        for a_menos, a_mais in [(0, 1), (1, 2)]:
            f = []
            for a in (a_menos, a_mais):
                e = env(['simulador.ruido_autorrelato=0']); e.reset(urgencia=0, fadiga=0.2); f.append(e.step(a)[3]['fadiga'])
            self.assertLess(f[0], f[1])

    def test_urgencia_aumenta_fadiga(self):
        f = []
        for u in (0, 2):
            e = env(); e.reset(urgencia=u, fadiga=0.2); f.append(e.step(1)[3]['fadiga'])
        self.assertLess(f[0], f[1])

    def test_pausa_recupera(self):
        e = env(['simulador.ruido_autorrelato=0', 'simulador.fadiga_por_min=0']); e.reset(urgencia=0, fadiga=0.6)
        self.assertAlmostEqual(e.step(2)[3]['fadiga'], 0.6 - 0.035 * 10)

    def test_penalidade_quando_cansaco_alto(self):
        e = env(['simulador.ruido_autorrelato=0', 'simulador.fadiga_por_min=0', 'simulador.recuperacao_por_min_pausa=0'])
        e.reset(urgencia=0, fadiga=0.9)                    # relato 1+3.6 -> 5 (>= limiar 4)
        _, r, _, i = e.step(1)
        self.assertTrue(i['cansaco_alto']); self.assertAlmostEqual(r, 25 - 10)
        e.reset(urgencia=0, fadiga=0.0); _, r, _, i = e.step(1)
        self.assertFalse(i['cansaco_alto']); self.assertAlmostEqual(r, 25)

    def test_recompensa_basica_e_minutos(self):
        e = env(['simulador.ruido_autorrelato=0', 'simulador.fadiga_por_min=0']); e.reset(urgencia=0, fadiga=0)
        for a, m in enumerate([15, 25, 50]):
            e.reset(urgencia=0, fadiga=0); self.assertAlmostEqual(e.step(a)[1], m)

    def test_eficiencia_reduz_recompensa_e_urgencia_muda_penalidade(self):
        base = env(['simulador.ruido_autorrelato=0'], 'basica'); efic = env(['simulador.ruido_autorrelato=0'], 'eficaz')
        base.reset(urgencia=0, fadiga=0.3); efic.reset(urgencia=0, fadiga=0.3)
        self.assertLess(efic.step(2)[1], base.step(2)[1])
        pens = []
        for u in (0, 2):
            e = env(['simulador.ruido_autorrelato=0', 'simulador.fadiga_por_min=0', 'simulador.recuperacao_por_min_pausa=0'], 'urgencia')
            e.reset(urgencia=u, fadiga=0.95); pens.append(e.step(0)[3]['penalidade'])
        self.assertEqual(pens, [15.0, 5.0])

    def test_config_invalida_e_chave_inexistente(self):
        with self.assertRaises(KeyError): montar(CFG, sets=['simulador.nao_existe=1'])
        with self.assertRaises(ValueError): montar(CFG, sets=['simulador.urgencia_probs=[0.5,0.5]'])

if __name__ == '__main__': unittest.main()
