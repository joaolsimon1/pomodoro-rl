"""Leitura da configuração JSON, variantes de recompensa e sobrescritas (--set chave=valor)."""
import json
from pathlib import Path

ACOES = ('curto', 'padrao', 'foco')
URGENCIAS = ('longe', 'semana', 'vespera')


def carregar(caminho):
    return json.loads(Path(caminho).read_text(encoding='utf-8'))


def _converter(texto):
    try:
        return json.loads(texto)
    except ValueError:
        return texto


def definir(cfg, chave, valor):
    """Atribui cfg['a']['b'] = valor para a chave 'a.b'; a chave precisa já existir (evita typos)."""
    partes = chave.split('.')
    d = cfg
    for p in partes[:-1]:
        if not isinstance(d.get(p), dict):
            raise KeyError(f'Chave inexistente: {chave}')
        d = d[p]
    if partes[-1] not in d:
        raise KeyError(f'Chave inexistente: {chave}')
    d[partes[-1]] = valor


def validar(cfg):
    s, r = cfg['simulador'], cfg['recompensa']
    if set(s['ciclos']) != set(ACOES):
        raise ValueError(f'ciclos deve ter exatamente as chaves {ACOES}')
    for a in ACOES:
        e, p = s['ciclos'][a]
        if e < 1 or p < 0:
            raise ValueError(f'ciclo {a}: estudo>=1 e pausa>=0')
    if len(s['urgencia_probs']) != 3 or abs(sum(s['urgencia_probs']) - 1) > 1e-6:
        raise ValueError('urgencia_probs precisa de 3 valores somando 1')
    if len(s['multiplicador_urgencia']) != 3 or len(r['penalidade_por_urgencia']) != 3:
        raise ValueError('multiplicador_urgencia e penalidade_por_urgencia precisam de 3 valores')
    lo, hi = s['fadiga_inicial']
    if not 0 <= lo <= hi <= 1:
        raise ValueError('fadiga_inicial: 0 <= min <= max <= 1')
    if not 1 <= s['limiar_cansaco_alto'] <= 5:
        raise ValueError('limiar_cansaco_alto entre 1 e 5')
    if s['n_ciclos_por_episodio'] < 1 or r['penalidade'] < 0:
        raise ValueError('n_ciclos_por_episodio>=1 e penalidade>=0')
    t = cfg['treino']
    if t['n_episodios'] < 1 or t['n_episodios_avaliacao'] < 1 or t['blocos_progresso'] < 1:
        raise ValueError('treino: valores devem ser >= 1')
    return cfg


def montar(caminho, variante=None, penalidade=None, sets=(), rapido=False):
    """Config final: arquivo -> variante -> penalidade -> rapido -> --set (o último vence)."""
    cfg = carregar(caminho)
    if variante:
        cfg['recompensa'].update(cfg['variantes_recompensa'][variante])
        cfg['recompensa']['variante'] = variante
    if penalidade is not None:
        cfg['recompensa']['penalidade'] = float(penalidade)
    if rapido:
        cfg['treino'].update(n_episodios=1500, n_episodios_avaliacao=300, blocos_progresso=5)
    for item in sets:
        chave, valor = item.split('=', 1)
        definir(cfg, chave.strip(), _converter(valor))
    return validar(cfg)
