import json, sys, tempfile, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import paralelo

FAKE = '''
import sys, json, time, pathlib
out, passos, falha = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1'
p = pathlib.Path(out); p.mkdir(parents=True, exist_ok=True)
(p / 'span.txt').write_text(str(time.time()) + ' ')
for e in range(1, passos + 1):
    time.sleep(.3); print(json.dumps({'avancou': 1}), flush=True)
    if falha and e == 1: sys.exit(3)
with open(p / 'span.txt', 'a') as f: f.write(str(time.time()))
(p / 'concluido.json').write_text('{}')
'''

class Recorder:
    instances = []
    def __init__(self, total, desc, initial=0): self.total, self.n = total, 0; Recorder.instances.append(self)
    def update(self, k=1): self.n += k
    def set_postfix(self, t): self.postfix = t
    def write(self, t): pass
    def close(self): pass

class ParaleloTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.script = self.root / 'fake.py'; self.script.write_text(FAKE)
        self._bar = paralelo._Bar; paralelo._Bar = Recorder; Recorder.instances.clear()
    def tearDown(self):
        paralelo._Bar = self._bar; self.tmp.cleanup()

    def cmd(self, nome, passos=3, falha=0):
        return [sys.executable, str(self.script), str(self.root / nome), str(passos), str(falha)]
    def spans(self, nomes):
        return [list(map(float, (self.root / n / 'span.txt').read_text().split())) for n in nomes]

    def test_paralelo_e_progresso_completo(self):
        nomes = ['a', 'b', 'c', 'd']
        jobs = [paralelo.Job(n, [self.cmd(n)], 3, self.root / n) for n in nomes]
        t = time.time(); paralelo.run_jobs(jobs, workers=4); par = time.time() - t
        bar = Recorder.instances[-1]; self.assertEqual((bar.n, bar.total), (12, 12))
        s = self.spans(nomes)
        self.assertTrue(all(x[0] < min(y[1] for y in s) for x in s), 'execuções deveriam se sobrepor')
        self.assertLess(par, 3 * .3 * 4 * .75)

    def test_workers_limita_concorrencia(self):
        nomes = ['a', 'b', 'c']
        jobs = [paralelo.Job(n, [self.cmd(n, 2)], 2, self.root / n) for n in nomes]
        paralelo.run_jobs(jobs, workers=1)
        s = sorted(self.spans(nomes))
        for prev, nxt in zip(s, s[1:]):
            self.assertLessEqual(prev[1], nxt[0] + 1e-3, 'com 1 worker não pode haver sobreposição')

    def test_falha_nao_interrompe_outros(self):
        jobs = [paralelo.Job('ruim', [self.cmd('ruim', 3, 1)], 3, self.root / 'ruim'),
                paralelo.Job('bom', [self.cmd('bom', 3)], 3, self.root / 'bom')]
        with self.assertRaises(RuntimeError) as ctx: paralelo.run_jobs(jobs, workers=2)
        self.assertIn('ruim', str(ctx.exception))
        self.assertTrue((self.root / 'bom' / 'concluido.json').exists())
        s = paralelo.run_jobs([paralelo.Job('ruim2', [self.cmd('ruim2', 3, 1)], 3)], raise_on_error=False)
        self.assertEqual(s['failed'], ['ruim2'])

    def test_job_from_command_pula_concluidos(self):
        c = self.cmd('r') + ['--out', str(self.root / 'r')]
        self.assertEqual(paralelo.job_from_command(c, 5).units, 5)
        (self.root / 'r').mkdir(); (self.root / 'r' / 'concluido.json').write_text('{}')
        self.assertIsNone(paralelo.job_from_command(c, 5))

if __name__ == '__main__': unittest.main()
