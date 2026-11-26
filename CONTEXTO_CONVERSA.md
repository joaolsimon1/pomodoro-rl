# Contexto da conversa (para transferir a outro LLM)

> Como usar: cole este arquivo no início de uma nova conversa e anexe `pomodoro-rl.zip`. Peça, por exemplo: "Leia o contexto e o README do zip; depois me ajude com [próximo passo]". Idioma de trabalho: português do Brasil. Data de referência: 24–25/11/2026.

## 1. Quem e o quê

Estudante cursando a disciplina de Aprendizado por Reforço, fazendo o **projeto final** (o enunciado fala em grupos). Ideia do trabalho: um **Pomodoro para estudo com RL**, no qual o agente aprende o ritmo de estudo/pausa que maximiza o rendimento sem levar o aluno à exaustão. O trabalho foi proposto com base em um artigo e **ainda precisa ser validado com o professor** (exigência do enunciado).

## 2. Enunciado do trabalho (resumo fiel)

- Projeto final: aplicar e/ou modificar um sistema baseado em RL em um benchmark ou problema relevante; conversar com o professor e validar o escopo (sem validação há risco de projeto inadequado).
- Formas válidas: (a) usar a abordagem de um artigo em outro problema/benchmark; (b) modificar a abordagem de um artigo no mesmo problema; (c) criar uma aplicação que use RL (site, jogo, app, serviço), com mais valor se houver contato com o mundo real (deploy, teste com usuários).
- Tem que "pensar como cientista": hipótese, cenários e métricas de avaliação, conclusão validando ou refutando a hipótese; comentar limitações, desafios e barreiras.
- Entrega: artigo científico com resumo, introdução, trabalhos relacionados, abordagem/metodologia, resultados, discussão e conclusão (conteúdo adaptável à modalidade; numa aplicação, trabalhos relacionados podem ser aplicações concorrentes). Template sugerido: arXiv/Prime AI style (Overleaf).

## 3. Artigo de base

Vinesh, Rakshitha G, Skandapriya V, Bhoomika G M, Hemalatha J D. *Smart Break Recommendation System Based on Student Mental Fatigue and Task Type*. ICSFT 2026 (Bangalore, 2–3 jan. 2026), IEEE. DOI 10.1109/ICSFT66733.2026.11507606. Link do usuário: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11507606

Resumo do artigo:
- Sistema local (Python) que monitora passivamente teclado (pynput), mouse e janela ativa, agrega snapshots a cada 60 s e usa: Random Forest para classificar a tarefa (6 classes; acurácia de teste 89,7%), XGBoost para fadiga (escore [0,1] → Baixo/Médio/Alto; acurácia 82,4%, ROC-AUC 0,91) e um agente **A2C** que escolhe o tipo/momento da pausa (micro, alongamento, respiração, caminhada, hidratação).
- Estado do RL: nível de fadiga, tipo de tarefa, tempo desde a pausa, duração do foco. Recompensa: 0,5·Δfadiga + 0,3·Δprodutividade + 0,2·conformidade; γ = 0,95; lr 0,0003; hidden 64; exploração 0,3 → 0,05.
- Piloto: 5 participantes, 2 semanas; aceitação 62% (alegam 2,2× sobre intervalo fixo); CPU < 2%, ~150 MB RAM; privacidade por processamento local.
- Inconsistências notadas: faixas de acurácia no resumo vs. valores pontuais no corpo; 84,2% na conclusão vs. 82,4% na Tabela II; CPU "< 2%" vs. "< 2,5%"; XGBoost descrito como regressor e classificador; o título fala em "estudantes" mas o piloto não especifica; amostra minúscula.
- Limitações declaradas: piloto pequeno, adaptação de 1–2 semanas (cold start), só 5 tipos de pausa.

## 4. Proposta do grupo (MDP)

Pomodoro com RL em vez da regra fixa 25/5.

- **Estado** (antes de iniciar um bloco): nível de cansaço 1–5 informado pelo aluno ao fim do ciclo anterior; urgência: "Longe da Prova", "Semana da Prova", "Véspera da Prova". (15 estados.)
- **Ação:** Ciclo Curto 15/5 (cansaço alto), Ciclo Padrão 25/5, Ciclo Foco 50/10 (energia alta).
- **Recompensa:** +1 por minuto de estudo concluído; −10 se a ação levar o aluno a relatar cansaço alto no ciclo seguinte.

## 5. Datasets (resposta dada; links NÃO verificados)

Não existe dataset público conhecido com o MDP exato (cansaço → tipo de ciclo → cansaço seguinte). Úteis como proxy ou para calibrar um simulador: **SWELL-KW** (trabalho de conhecimento, teclado/mouse, pressão de tempo, carga mental), **StudentLife/Dartmouth** (estresse, sono e prazos de estudantes), **OULAD** (cliques e datas de avaliação; sem cansaço), **EdNet / ASSISTments / Junyi** (desempenho × tempo de sessão, pausas). Também vale buscar "pomodoro" e "study sessions" no Kaggle/Zenodo. Caminhos sugeridos: (1) simulador calibrado, (2) coleta própria (app que sorteia o ciclo e pergunta o cansaço 1–5, gerando tuplas s, a, s'), (3) híbrido (pré-treino no simulador, ajuste com dados reais).

## 6. Observações de modelagem feitas e adotadas no código

1. Espaço pequeno (5×3 estados, 3 ações): Q-learning tabular ou bandit contextual bastam; A2C é exagero. Manter o MDP, mas comparar com bandit.
2. Recompensa possivelmente degenerada: com +1/min, foco domina e só a penalidade contém; sensível à razão +1/−10. Alternativas: varrer a razão, ponderar minutos por eficácia (minutos em baixo cansaço valem mais).
3. Urgência deveria alterar o trade-off (tolerar mais cansaço na véspera): penalidade dependente da urgência.
4. Cansaço autorrelatado é ruidoso/subjetivo: tratar como parcialmente observável ou discutir como limitação.
5. Avaliação: comparar políticas no simulador (minutos totais, minutos efetivos, frequência de cansaço alto) contra as fixas; deixar claro que depende das premissas do simulador.

## 7. Entregáveis desta etapa

O usuário enviou um zip modelo (`reli-h100-paralelo.zip`, projeto "reli-seminario": `src/` com `paralelo.py` (processos filhos + barra tqdm lendo linhas JSON), `estudo.py`, etc.; `tests/`, `notebooks/`, `README.md`, `VALIDACAO.md`, `requirements.txt`, `.gitignore`) e pediu: este `.md` e um zip semelhante, com **desenvolvimento simples**, base para dataset simulado e agente, **parâmetros editáveis**, **paralelismo com escolha do número de CPUs**, **barra de progresso**, seguindo as observações da seção 6.

Foi entregue `pomodoro-rl.zip`:

```
config/padrao.json    parâmetros (JSON); sobrescrita com --set chave=valor
src/config.py         leitura/validação, variantes de recompensa
src/simulador.py      ambiente: fadiga latente [0,1], autorrelato 1–5 ruidoso, recompensa
src/agentes.py        qlearning, bandit (γ=0), fixo_{curto,padrao,foco}, regra, aleatorio
src/treino.py         uma execução (treino+avaliação) -> concluido.json; progresso em JSON
src/dataset.py        dataset simulado de transições (CSV), política de comportamento aleatória
src/paralelo.py       run_jobs(jobs, workers) + barra (adaptado do modelo)
src/executar_grade.py --modo grade|dataset --cpus N --rapido --dry-run
src/analise.py        tabela média ± IC95% entre sementes, pareto.png, curvas.png, política
notebooks/01_simulacao_pomodoro.ipynb   N_CPUS, OVERRIDES, RAPIDO
tests/                21 testes; README.md; VALIDACAO.md
```

Dinâmica do simulador (hipóteses editáveis, não calibradas com dados reais): cada minuto de estudo aumenta a fadiga `0,006 × mult_urgência × (1 + 1,0·t/25)`; cada minuto de pausa reduz `0,035`; eficiência por minuto `1 − F^1,5`; relato = round(1 + 4F + N(0, 0,35)) em [1,5]; "cansaço alto" = relato ≥ 4; episódio = 8 ciclos com urgência fixa (probabilidades 0,4/0,4/0,2). Variantes de recompensa: `basica` (+1/min, −10), `eficaz` (minutos ponderados por eficiência), `urgencia` (eficiência + penalidade 1,5×/1×/0,5× por urgência). Grade padrão: {qlearning, bandit} × 3 variantes × penalidades {5,10,20} × 5 sementes + 5 baselines × 5 sementes = 115 execuções. Avaliação só por métricas independentes da recompensa.

## 8. Estado e limitações

- 21 testes passam; grade e dataset em modo rápido rodaram de ponta a ponta. Ambiente de teste com 1 CPU: o ganho real de `--cpus > 1` não foi medido; só o modo texto da barra foi testado (tqdm ausente); o notebook não foi executado em Jupyter real.
- Fumaça (semente 11, sem IC, não é conclusão): `basica` degenera em foco; com `eficaz` o Q-learning ficou próximo do melhor fixo em minutos efetivos com bem menos ciclos de cansaço alto. Parâmetros do simulador foram ajustados por plausibilidade numa busca pequena (a recuperação inicial de 0,02 tornava a penalidade inevitável).
- Simplificações: horizonte finito sem o índice do ciclo no estado; fadiga escalar única (sem componente crônica); sem dados reais.
- Nenhuma grade completa executada ainda.

## 9. Hipóteses sugeridas (a validar com o professor)

H1: Q-learning dá mais minutos efetivos que as políticas fixas com frequência de cansaço alto ≤. H2: Q-learning supera o bandit (se não, é resultado). H3: a política é sensível à razão +1/penalidade e à variante; `basica` tende a degenerar. H4: ruído maior no autorrelato piora o desempenho.

## 10. Próximos passos possíveis

Validar escopo com o professor; rodar a grade completa e analisar; estudos de sensibilidade (ruído, limiar, parâmetros da fadiga); RL offline sobre o dataset (removendo colunas latentes); calibrar o simulador com literatura/datasets; incluir o ciclo da sessão no estado; coleta com voluntários reais (aplicação com teste de usuários, que o enunciado valoriza); redigir o artigo no template sugerido.
