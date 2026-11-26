import json, sys, tempfile, unittest
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))
from config import montar
from treino import executar, unidades
from dataset import gerar
import executar_grade

CFG = RAIZ / 'config/padrao.json'

class TreinoTests(unittest.TestCase):
    def test_progresso_soma_total_e_arquivos(self):
        cfg = montar(CFG, 'eficaz', rapido=True)
        for alg in ['qlearning', 'fixo_padrao']:
            with tempfile.TemporaryDirectory() as d:
                vistos = []; res = executar(cfg, alg, 7, Path(d) / alg, emitir=vistos.append)
                self.assertEqual(sum(vistos), unidades(cfg, alg))
                self.assertTrue((Path(d) / alg / 'concluido.json').exists())
                m = res['metricas']; self.assertTrue(0 <= m['freq_cansaco_alto'] <= 1)
                self.assertAlmostEqual(sum(m['uso_acoes'].values()), 1.0)
                self.assertEqual(('politica' in res), alg == 'qlearning')

    def test_reprodutivel_e_avaliacao_comum(self):
        cfg = montar(CFG, 'basica', rapido=True)
        with tempfile.TemporaryDirectory() as d:
            a = executar(cfg, 'qlearning', 3, Path(d) / 'a', emitir=lambda k: None)['metricas']
            b = executar(cfg, 'qlearning', 3, Path(d) / 'b', emitir=lambda k: None)['metricas']
            self.assertEqual(a, b)

    def test_dataset(self):
        cfg = montar(CFG, rapido=True); vistos = []
        df = gerar(cfg, 'aleatorio', 50, 1, emitir=vistos.append, ep0=100)
        self.assertEqual(len(df), 50 * 8); self.assertEqual(sum(vistos), 50)
        self.assertEqual(df.episodio.min(), 100); self.assertEqual(set(df.acao), {'curto', 'padrao', 'foco'})
        self.assertEqual(int(df.terminal.sum()), 50)

    def test_montagem_da_grade_e_retomada(self):
        with tempfile.TemporaryDirectory() as d:
            jobs, total = executar_grade.jobs_grade(CFG, d)
            self.assertEqual(total, 2 * 3 * 3 * 5 + 5 * 5)          # 90 aprendizes + 25 baselines
            jobs, total = executar_grade.jobs_grade(CFG, d, algoritmos=['qlearning'], variantes=['eficaz'],
                                                    penalidades=[5], sementes=[1], baselines=['fixo_foco'], rapido=True)
            self.assertEqual((len(jobs), total), (2, 2))
            self.assertEqual(sum(j.units for j in jobs), 1500 + 300 + 300)
            (jobs[0].folder).mkdir(parents=True); (jobs[0].folder / 'concluido.json').write_text('{}')
            self.assertEqual(len(executar_grade.jobs_grade(CFG, d, algoritmos=['qlearning'], variantes=['eficaz'],
                             penalidades=[5], sementes=[1], baselines=['fixo_foco'], rapido=True)[0]), 1)

if __name__ == '__main__': unittest.main()
