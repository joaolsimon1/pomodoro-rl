# Análise do experimento de sensibilidade e generalização

## Objetivo e conclusão

O experimento investiga como políticas de escolha de ciclos Pomodoro se comportam quando as hipóteses sobre fadiga e recuperação mudam. Nos cinco cenários simulados, o Q-learning apresentou maior rendimento e menor frequência de cansaço alto do que o bandit contextual, tanto com treino em cada cenário quanto na avaliação das políticas treinadas na referência. Os intervalos de confiança de 95% das diferenças pareadas não incluem zero nessas comparações.

O ciclo curto fixo, porém, superou o Q-learning nas duas métricas em todos os cenários e nos dois protocolos. Assim, os resultados favorecem o Q-learning frente ao bandit testado, mas não demonstram que ele seja a melhor política entre as alternativas avaliadas. As conclusões se restringem às condições simuladas e às métricas adotadas.

## Desenho experimental

O estado observado combina cansaço autorrelatado de 1 a 5 e três níveis de urgência. A fadiga real é latente. As ações são ciclos curto (15 minutos de estudo e 5 de pausa), padrão (25/5) e foco (50/10). Cada episódio tem oito ciclos. A fadiga cresce durante o estudo, diminui na pausa e afeta a eficiência do estudo. O autorrelato contém ruído.

Foram comparados Q-learning, bandit contextual, três políticas fixas, uma regra por cansaço e uma política aleatória. Na sensibilidade, Q-learning e bandit são treinados novamente em cada cenário. Na generalização, suas tabelas Q treinadas na referência são avaliadas sem atualização em todos os cenários. Para cada cenário são usadas 30 sementes e, para os agentes que aprendem, 20.000 episódios de treino e 2.000 de avaliação por semente. As políticas de referência também são avaliadas em 2.000 episódios por semente. O notebook registrou 1.050 trabalhos concluídos na grade de sensibilidade.

| Cenário | Fadiga por minuto | Recuperação por minuto de pausa |
|---|---:|---:|
| Referência | 0,006 | 0,035 |
| Fadiga lenta | 0,0045 | 0,035 |
| Fadiga rápida | 0,0075 | 0,035 |
| Recuperação lenta | 0,006 | 0,02625 |
| Recuperação rápida | 0,006 | 0,04375 |

As variações são de 25% em um parâmetro por vez. O custo adicional do tempo contínuo e o ruído de observação permanecem fixos. A recompensa de treino usa minutos efetivos e penalidade 10 para relato de cansaço alto.

As métricas são **minutos efetivos por hora**, considerando estudo e pausas, e **percentual de ciclos com cansaço alto**, medido pelo autorrelato após a pausa. Como cada episódio termina após oito ciclos, políticas diferentes podem ocupar durações totais diferentes. A normalização por hora facilita a comparação, mas não substitui uma avaliação com mesmo orçamento fixo de tempo.

## Sensibilidade com treino em cada cenário

A tabela apresenta diferenças médias pareadas por semente e IC de 95%. Para rendimento, valores positivos favorecem Q-learning. Para cansaço, valores negativos o favorecem. A unidade de cansaço é ponto percentual (p.p.).

| Cenário | Q-learning − bandit: rendimento, IC 95% | Q-learning − bandit: cansaço, IC 95% | Q-learning − curto: rendimento, IC 95% | Q-learning − curto: cansaço, IC 95% |
|---|---:|---:|---:|---:|
| Referência | 14,04 [12,52; 15,57] | −27,28 [−31,16; −23,40] | −8,35 [−9,46; −7,24] | 5,73 [3,45; 8,00] |
| Fadiga lenta | 11,47 [10,15; 12,79] | −22,78 [−25,47; −20,09] | −7,23 [−8,23; −6,23] | 1,84 [1,14; 2,54] |
| Fadiga rápida | 21,16 [19,66; 22,67] | −44,63 [−48,21; −41,05] | −6,11 [−7,38; −4,84] | 3,27 [1,59; 4,95] |
| Recuperação lenta | 21,59 [20,51; 22,67] | −64,69 [−68,02; −61,36] | −6,25 [−7,11; −5,40] | 5,58 [3,68; 7,49] |
| Recuperação rápida | 18,47 [17,46; 19,49] | −17,99 [−18,76; −17,22] | −6,96 [−7,95; −5,98] | 1,11 [0,61; 1,61] |

Na referência, o Q-learning obteve em média 34,63 minutos efetivos por hora (IC 95% 33,53 a 35,74) e 5,73% de ciclos com cansaço alto (IC 3,45% a 8,00%). A vantagem sobre o bandit muda de magnitude com a dinâmica. O ciclo curto fixo permanece superior ao Q-learning nas duas métricas, o que merece destaque ao apresentar o resultado.

## Generalização a partir da referência

Neste protocolo, a avaliação usa políticas aprendidas na referência e outro fluxo aleatório de avaliação. Os valores abaixo são diferenças médias pareadas por semente, com IC de 95%, em minutos efetivos por hora.

| Cenário de avaliação | Q-learning − bandit | Q-learning − curto |
|---|---:|---:|
| Referência | 14,09 [12,56; 15,62] | −8,32 [−9,45; −7,20] |
| Fadiga lenta | 11,99 [10,57; 13,42] | −3,54 [−4,43; −2,66] |
| Fadiga rápida | 12,75 [11,20; 14,31] | −12,06 [−13,37; −10,74] |
| Recuperação lenta | 14,30 [12,71; 15,89] | −10,07 [−11,35; −8,80] |
| Recuperação rápida | 13,31 [11,75; 14,87] | −5,97 [−7,01; −4,92] |

Em cansaço, Q-learning ficou abaixo do bandit entre 9,46 e 46,07 p.p. nas médias dos cenários, mas acima do ciclo curto entre 0,75 e 17,18 p.p. Os respectivos intervalos também não cruzam zero. Nos cenários de fadiga rápida e recuperação lenta, a diferença de cansaço frente ao ciclo curto foi de 17,18 p.p. e 16,76 p.p., respectivamente. Isso indica que a política treinada na referência pode ser sensível a mudanças desfavoráveis, apesar da vantagem sobre o bandit.

A comparação entre os dois protocolos é descritiva: eles usam fluxos de avaliação distintos e o notebook não calcula um intervalo pareado diretamente para o efeito de treinar novamente. Não se deve interpretar a diferença entre as tabelas, isoladamente, como estimativa formal desse efeito.

## Interpretação estatística e limites

Os intervalos são calculados com a distribuição t de Student e 29 graus de liberdade, usando a variabilidade entre as 30 diferenças por semente. São intervalos pontuais para cada comparação, sem ajuste para múltiplas comparações. A mesma semente identifica pares de execuções, mas políticas com trajetórias diferentes não necessariamente vivenciam condições idênticas em cada passo. Ausência de zero no intervalo não mede a relevância prática do efeito nem garante validade fora dos cenários testados.

O bandit e o Q-learning compartilham a representação tabular, mas diferem no desconto. Sua comparação é compatível com investigar o papel do horizonte futuro sob esta implementação e orçamento de treino. Ela não isola todos os possíveis efeitos de convergência, exploração ou ajuste de hiperparâmetros.

A recompensa otimizada é acumulada ao longo de oito decisões e inclui penalidade de cansaço. O rendimento por hora é uma métrica de avaliação diferente. Esse desalinhamento é uma explicação possível para a vantagem do ciclo curto fixo, e requer investigação antes de atribuir o resultado a uma causa específica. Os valores de 0,00% de cansaço para o ciclo curto são arredondamentos observados nas simulações, não prova de probabilidade zero.

A dinâmica e seus parâmetros são hipóteses, não medições de alunos. O estudo cobre perturbações locais de um parâmetro por vez, com a mesma estrutura de simulador, ruído e horizonte. Ele não estabelece eficácia em pessoas, robustez a outras dinâmicas, nem generalização para combinações não testadas. Comparações futuras podem avaliar orçamento fixo de tempo, cenários combinados e ruído de autorrelato, além de examinar as ações aprendidas e o alinhamento da recompensa com a métrica principal.

## Origem dos resultados

Os números vêm das saídas salvas de [02_sensibilidade.ipynb](notebooks/02_sensibilidade.ipynb) e da grade em `experiments/sensibilidade`. A saída de generalização foi mantida no notebook; não há CSV de resultados individuais de generalização disponível no projeto no momento desta análise. As tabelas do notebook exibem ICs e detalhes adicionais.
