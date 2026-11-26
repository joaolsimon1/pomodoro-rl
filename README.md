# Pomodoro adaptativo com aprendizado por reforço

Base de código para o projeto final de Aprendizado por Reforço: um agente que escolhe o próximo ciclo de estudo (curto, padrão ou foco) conforme o cansaço relatado e a urgência da prova, para maximizar o rendimento sem levar o aluno à exaustão. Esta entrega contém o **simulador**, o **gerador de dataset simulado**, os **agentes** (Q-learning, bandit contextual e baselines), a **execução paralela com barra de progresso** e a **análise**. Não há resultados científicos nesta entrega: os números de `VALIDACAO.md` são verificações de software.

Artigo de base: Vinesh et al., *Smart Break Recommendation System Based on Student Mental Fatigue and Task Type*, ICSFT 2026 (https://ieeexplore.ieee.org/document/11507606). O projeto **adapta** a ideia (pausas adaptativas com RL) a um problema mais simples, com autorrelato no lugar de sinais de teclado/mouse; não é uma reprodução. Pelo enunciado, o escopo precisa ser validado com o professor antes de avançar.

## Modelagem (MDP)

| Elemento | Definição |
|---|---|
| Estado (observação) | (cansaço autorrelatado 1–5, urgência: longe / semana / véspera) = 15 estados |
| Ações | `curto` 15/5, `padrao` 25/5, `foco` 50/10 (estudo/pausa em minutos) |
| Recompensa | +1 por minuto de estudo; −10 se o cansaço relatado no ciclo seguinte for ≥ 4 |
| Episódio | uma sessão de 8 ciclos; urgência fixa por episódio, fadiga inicial sorteada |
| Agentes | Q-learning tabular (γ = 0,95) e bandit contextual (mesmo código, γ = 0) |

### Decisões de modelagem (e por quê)

- **Espaço pequeno, método tabular.** 15 estados × 3 ações: Q-learning tabular é suficiente e interpretável (a política aprendida é uma tabela 5×3). Não usa A2C.
- **Bandit como baseline.** Se o bandit (só recompensa imediata) igualar o Q-learning, o horizonte longo do MDP não está ajudando.
- **Recompensa potencialmente degenerada.** Com +1/min, o ciclo foco rende mais minutos em quase qualquer situação, e só a penalidade o contém. Por isso a grade varre a **razão +1/penalidade** e três **variantes de recompensa**:
  - `basica`: exatamente a proposta (+1/min, −10 fixo);
  - `eficaz`: minutos ponderados pela eficiência (estudar muito cansado rende menos);
  - `urgencia`: eficiência + penalidade dependente da urgência (tolera mais cansaço na véspera: 0,5×; longe da prova: 1,5×).
- **Observação ruidosa.** O cansaço real (fadiga latente em [0, 1]) é oculto; o agente só vê o autorrelato com ruído (`ruido_autorrelato`). Use `--set simulador.ruido_autorrelato=0` para ver o efeito da observabilidade parcial.
- **Avaliação independente da recompensa.** Variantes diferentes têm recompensas diferentes, então a comparação usa **minutos de estudo**, **minutos efetivos** e **frequência de cansaço alto** (fração de ciclos com autorrelato ≥ limiar), todas contra as políticas fixas (sempre curto / padrão / foco), a regra por cansaço e a aleatória.

### Limitações do simulador

Todos os parâmetros são **hipóteses plausíveis**, não medidos em alunos; foram ajustados para a dinâmica ser não trivial (ciclos longos cansam mais, pausas recuperam). Resultados valem para este simulador, não para estudantes reais. O estado não inclui o índice do ciclo dentro da sessão (horizonte finito tratado como se fosse infinito), e a fadiga é um único escalar (sem componente crônica). Esses pontos podem ser extensões ou limitações no artigo.

## Estrutura

```
config/padrao.json         # TODOS os parâmetros editáveis
src/config.py              # leitura, variantes de recompensa, --set chave=valor
src/simulador.py           # ambiente (fadiga latente, autorrelato, recompensa)
src/agentes.py             # Q-learning, bandit, fixos, regra, aleatório
src/treino.py              # UMA execução: treina, avalia, grava concluido.json
src/dataset.py             # dataset simulado de transições (CSV)
src/paralelo.py            # processos paralelos + barra de progresso (tqdm ou texto)
src/executar_grade.py      # grade completa ou dataset em shards, com --cpus
src/analise.py             # tabela (média ± IC95% entre sementes), gráficos, política
notebooks/01_simulacao_pomodoro.ipynb
tests/                     # 21 testes (simulador, agentes, treino/dataset, paralelismo)
```

## Executar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v

# teste de fumaça (poucos episódios)
python src/executar_grade.py --modo grade --cpus 4 --rapido --sementes 11 22

# grade completa: 90 execuções de agentes + 25 de baselines
python src/executar_grade.py --modo grade --cpus 4
python src/analise.py --pasta experiments/grade
python src/analise.py --pasta experiments/grade --politica qlearning_eficaz_p10_s11

# dataset simulado (política aleatória), em 8 shards
python src/executar_grade.py --modo dataset --cpus 4 --shards 8 --episodios 80000
```

`--cpus N` define quantas execuções rodam simultaneamente (1 CPU cada; limitado ao número de CPUs da máquina). A barra conta **episódios** concluídos em todas as execuções e mostra jobs concluídos, ativos e falhos. Execuções já concluídas são puladas (basta repetir o comando para retomar); uma falha não interrompe as demais. `--dry-run` só conta execuções e episódios. Uma execução individual:

```bash
python src/treino.py --out experiments/teste --algoritmo qlearning --variante eficaz --penalidade 10 --semente 11
```

## Parâmetros editáveis

Edite `config/padrao.json` ou sobrescreva sem editar: `--set simulador.fadiga_por_min=0.008` (repetível; vale para `treino.py`, `dataset.py` e `executar_grade.py`; chaves inexistentes dão erro).

| Chave | Padrão | Significado |
|---|---|---|
| `simulador.n_ciclos_por_episodio` | 8 | ciclos por sessão |
| `simulador.ciclos.*` | 15/5, 25/5, 50/10 | estudo/pausa em minutos por ação |
| `simulador.urgencia_probs` | 0,4 / 0,4 / 0,2 | probabilidade de longe / semana / véspera por episódio |
| `simulador.fadiga_inicial` | [0, 0,3] | faixa da fadiga latente inicial |
| `simulador.fadiga_por_min` | 0,006 | aumento de fadiga por minuto de estudo |
| `simulador.crescimento_tempo_continuo` | 1,0 | quanto a taxa cresce com o tempo contínuo no bloco (blocos longos custam mais) |
| `simulador.multiplicador_urgencia` | 1, 1,1, 1,25 | multiplica a taxa de fadiga por urgência |
| `simulador.recuperacao_por_min_pausa` | 0,035 | queda de fadiga por minuto de pausa |
| `simulador.ruido_autorrelato` | 0,35 | desvio-padrão do ruído do relato (0 = observação perfeita) |
| `simulador.limiar_cansaco_alto` | 4 | relato a partir do qual há penalidade |
| `simulador.eficiencia_min`, `expoente_eficiencia` | 0, 1,5 | eficiência por minuto = 1 − (1 − min)·F^expoente |
| `recompensa.pontos_por_min` / `penalidade` | 1 / 10 | +1 por minuto; −10 por cansaço alto |
| `recompensa.ponderar_por_eficiencia` | false | minutos ponderados por eficiência |
| `recompensa.penalidade_por_urgencia` | 1, 1, 1 | fator da penalidade por urgência |
| `variantes_recompensa.*` | basica, eficaz, urgencia | pacotes das duas últimas opções |
| `agente.alpha`, `gamma`, `q_inicial` | 0,1, 0,95, 0 | taxa de aprendizado, desconto, Q inicial |
| `agente.eps_inicio`, `eps_fim`, `eps_decai_frac` | 1, 0,05, 0,7 | exploração ε linear até 70% do treino |
| `regra.*` | 4, 2 | baseline: curto se cansaço ≥ 4; foco se ≤ 2; senão padrão |
| `treino.n_episodios`, `n_episodios_avaliacao` | 20000, 2000 | episódios de treino e de avaliação gulosa |
| `grade.*` | ver arquivo | algoritmos, variantes, penalidades, sementes e baselines da grade |

## Dataset simulado

`dataset.py` gera uma linha por ciclo: `episodio, ciclo, urgencia, cansaco, acao, minutos_estudo, minutos_efetivos, recompensa, cansaco_seguinte, cansaco_alto_seguinte, terminal, fadiga_latente, fadiga_latente_seguinte`. Com política aleatória, todos os pares (estado, ação) são visitados. As colunas `fadiga_latente*` só existem por ser simulado; **remova-as** ao treinar RL offline para imitar um dataset real. Um dataset real exigiria coletar exatamente as tuplas (s, a, s'), por exemplo com um app que sorteia o ciclo e pergunta o cansaço (1–5) ao final.

## Hipóteses e plano para o artigo (sugestão a validar com o professor)

- **H1.** O Q-learning obtém mais minutos efetivos que as políticas fixas com frequência de cansaço alto menor ou igual.
- **H2.** O Q-learning supera o bandit (o horizonte longo importa); se não, refutar H2 é um resultado.
- **H3.** A política aprendida é sensível à razão +1/penalidade e à variante de recompensa; a variante `basica` tende a degenerar em foco.
- **H4.** O desempenho cai com ruído maior no autorrelato (observabilidade parcial).

Mapeamento ao enunciado: resumo, introdução (fadiga digital, Pomodoro), trabalhos relacionados (artigo de base, datasets como SWELL-KW, StudentLife, EdNet para calibrar), metodologia (MDP e simulador acima), resultados (`analise.py`: tabela, `pareto.png`, `curvas.png`, tabela de política), discussão (degeneração da recompensa, limitações do simulador) e conclusão (hipóteses validadas ou refutadas). Próximos passos naturais: calibrar o simulador com dados da literatura, RL offline no dataset, coleta com voluntários reais, incluir o ciclo da sessão no estado.
## Experimento principal do seminário

Em quais dinâmicas de fadiga o Q-learning supera um bandit contextual ou uma regra simples? Execute a partir da raiz:

```bash
python src/sensibilidade.py --dry-run
python src/sensibilidade.py --cpus 28
```

São cinco cenários: referência e variações de ±25% em fadiga por minuto ou recuperação por minuto de pausa, um fator por vez. As amplitudes são perturbações hipotéticas locais. A recompensa é sempre `eficaz`, com penalidade 10. Cada cenário usa as mesmas cinco sementes e sete políticas. Q-learning e bandit treinam de novo em cada cenário. A avaliação usa episódios independentes do treino. Os 175 trabalhos são distribuídos entre processos em CPU; as tabelas de 15 × 3 dispensam GPU.

Em `experiments/sensibilidade/`, consulte `execucoes.csv` para comparar políticas dentro de cada par cenário/semente e `sensibilidade.png` para as médias. Minutos efetivos por hora = 60 × minutos efetivos totais / tempo total de estudo e pausas. O horizonte continua em oito decisões. A segunda métrica é a fração de ciclos com relato ≥ 4 após a pausa. Observe se ganhos de rendimento vêm acompanhados de mais cansaço e se o sentido das diferenças se repete entre sementes. As conclusões se aplicam aos cenários simulados.

O comando retoma trabalhos concluídos. Para alterar o protocolo, arquive a pasta de resultados antes de iniciar outra grade. Use `--rapido` para testar o fluxo numa pasta separada.
