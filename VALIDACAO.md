# Validação desta versão

Ambiente de teste: Linux, Python 3.12, numpy 2.x, pandas 3.x, matplotlib 3.x, **1 CPU**. Os números abaixo são verificações de software, não resultados científicos.

## O que foi verificado

`python -m unittest discover -s tests -v`: 21 testes passaram.

- **Simulador (9):** limites de fadiga e relato, término em 8 ciclos, determinismo por semente, foco cansa mais que curto, urgência aumenta a fadiga, pausa recupera exatamente `0,035 × minutos`, penalidade só com relato ≥ limiar, recompensa básica igual aos minutos, eficiência reduz a recompensa, penalidade por urgência (15 / 5 com fator 1,5 / 0,5), erro em chave inexistente e em probabilidades inválidas.
- **Agentes (4):** atualização de Q com e sem estado terminal, bandit ignora o futuro, aprendizado em um bandit trivial, baselines.
- **Treino, dataset e grade (4):** soma dos avisos de progresso igual ao total de episódios previsto, arquivos gravados, métricas coerentes, reprodutibilidade (mesma semente, mesmas métricas), dataset com 8 linhas por episódio e numeração de shards, montagem da grade (90 + 25 execuções) e retomada pulando pastas concluídas.
- **Paralelismo (4)**, com script falso: sobreposição real de processos com `workers=4`, limite de concorrência com `workers=1`, progresso completo (12/12), falha isolada sem interromper as demais, pular execuções concluídas.

Integração em modo `--rapido`: grade completa (115 execuções, 169.500 episódios), análise (tabela, `pareto.png`, `curvas.png`, política) e dataset em 4 shards (16.000 transições, todas as combinações cansaço × ação visitadas) rodaram sem erro. Uma execução padrão (20.000 episódios de treino + 2.000 de avaliação) leva cerca de 5–6 s.

## Observação do ajuste do simulador

Os parâmetros padrão foram escolhidos numa busca pequena (8 combinações, semente 11) para que a dinâmica não fosse trivial: com recuperação baixa (0,02) a penalidade era inevitável em cansaço alto e o agente escolhia foco em todos os estados. Esse ajuste foi feito por plausibilidade da dinâmica e **não** para favorecer o RL. Em uma execução de fumaça (semente 11, 20.000 episódios), observou-se: recompensa `basica` degenera em foco (Q-learning usa foco em 69% dos ciclos, 49% de ciclos com cansaço alto; o bandit é idêntico a `fixo_foco`); com `eficaz`, o Q-learning ficou próximo da melhor política fixa em minutos efetivos (123 contra 125 de `fixo_padrao`) com cansaço alto em 7% dos ciclos contra 21%. É uma única semente, sem intervalo de confiança: não use como conclusão.

## Limites desta validação

- Só havia 1 CPU: o ganho de tempo com `--cpus > 1` em execuções reais de `treino.py` não foi medido. A sobreposição foi verificada pelos testes com script falso.
- `tqdm` não estava instalado: só o modo de barra em texto simples foi exercitado. A barra `tqdm.auto` (usada nos notebooks) não foi testada.
- O notebook foi validado executando suas células com um harness (sem Jupyter); a célula de imagens (`IPython.display.Image`) não foi executada em um kernel real. Rode a primeira célula (que executa os testes) e depois em ordem, com `RAPIDO = True`, antes de ativar a grade completa.
- A grade completa (`RAPIDO = False`) não foi executada. Não há estudo de sensibilidade a parâmetros nem resultados com várias sementes.
- Nenhum dado real de estudantes foi usado.
