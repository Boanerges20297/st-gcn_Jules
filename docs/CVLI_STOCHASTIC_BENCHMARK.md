# Benchmark CVLI Estocástico

## Objetivo operacional

Este benchmark foi desenhado para o alvo correto de produção:

- prever **o próximo bairro com CVLI**
- ranquear bairros por probabilidade/sinal operacional no horizonte
- não otimizar apenas volume agregado de mortes

Em termos práticos, cada dia de validação vira um problema de ranking entre bairros de Fortaleza.

## Recorte temporal oficial

- treino: `2022-01-01` até `2024-12-31`
- validação principal: `2025-01-01` até `2025-12-31`
- horizonte: `14` dias
- cutoff máximo do bruto enriquecido para benchmark: `2025-12-31`

## Base e sinal estocástico

Arquivo bruto analisado:

- `data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv`

Resumo do sinal CVLI até `2025-12-31`:

| métrica | valor |
|---|---:|
| linhas CVLI | 11.704 |
| localizações únicas | 296 |
| período | 2022-01-01 a 2025-12-31 |
| média diária | 8,018 |
| variância diária | 10,444 |
| Fano factor | 1,303 |
| taxa de dias zerados | 0,07% |
| autocorrelação lag 1 | 0,089 |
| autocorrelação lag 7 | 0,122 |
| share de dias positivos com mais de 1 caso | 99,73% |
| p95 dos dias positivos | 14 |
| máximo diário | 20 |

Leitura objetiva:

- o processo não é puramente Poisson i.i.d.; há sobredispersão moderada
- existe memória temporal curta, especialmente semanal
- o sinal é esparso por bairro, mas não no agregado diário
- isso favorece modelos count-aware com boas features de recência e momentum

## Métricas do benchmark

- `hit1_event`: frequência em que o top-1 previsto caiu entre os bairros com CVLI no horizonte
- `p5_event`: precisão média do top-5
- `p10_event`: precisão média do top-10
- `p20_event`: precisão média do top-20
- `recall10_event`: cobertura média dos bairros positivos dentro do top-10
- `recall20_event`: cobertura média dos bairros positivos dentro do top-20
- `overlap10_rank`: sobreposição do top-10 previsto com o top-10 real
- `mrr_event`: quão cedo aparece o primeiro bairro positivo no ranking

## Grade avaliada

Modelos clássicos:

- `zero_baseline`
- `lag1_baseline`
- `roll7_baseline`
- `poisson_regressor`
- `histgb_classifier`
- `logit_classifier`
- `hurdle_logit_poisson`

Modelos deep incluídos na suíte:

- `ShallowGAT`
- `DeepSTGAT_64`
- `PureSTGCN_64`
- `FortalezaHeteroSTGAT`

## Resultado principal em Fortaleza

Resultado clássico consolidado em 2025:

| modelo | hit1 | p10 | mrr |
|---|---:|---:|---:|
| `zero_baseline` | 0,474 | 0,159 | 0,578 |
| `lag1_baseline` | 0,426 | 0,200 | 0,592 |
| `roll7_baseline` | 0,577 | 0,382 | 0,729 |
| `poisson_regressor` | 0,696 | 0,528 | 0,826 |
| `histgb_classifier` | 0,719 | 0,514 | 0,817 |
| `logit_classifier` | 0,690 | 0,518 | 0,826 |
| `hurdle_logit_poisson` | 0,659 | 0,538 | 0,810 |

## Leitura por percentil

Percentil relativo dentro dos 7 modelos clássicos:

| modelo | percentil hit1 | percentil p10 | percentil mrr |
|---|---:|---:|---:|
| `histgb_classifier` | 100 | 50 | 67 |
| `poisson_regressor` | 83 | 83 | 100 |
| `logit_classifier` | 67 | 67 | 100 |
| `hurdle_logit_poisson` | 50 | 100 | 50 |
| `roll7_baseline` | 33 | 33 | 33 |
| `zero_baseline` | 17 | 17 | 17 |
| `lag1_baseline` | 0 | 0 | 0 |

Interpretação direta:

- `histgb_classifier` foi o melhor em top-1 puro
- `hurdle_logit_poisson` foi o melhor em precisão top-10
- `poisson_regressor` foi o modelo mais equilibrado entre acerto imediato, cobertura útil e ordenação
- `logit_classifier` empatou o melhor `mrr`, mas ficou atrás do `poisson_regressor` em hit1 e p10

## Validação de estabilidade mensal

Playoff mensal já rodado para os finalistas em `2025-01` a `2025-03`:

Artefato:

- `outputs/benchmarks/cvli_finalists_monthly_fortaleza_20260612_223448.md`

Resumo:

| modelo | hit1 médio | desvio hit1 | p10 médio | desvio p10 | mrr médio |
|---|---:|---:|---:|---:|---:|
| `poisson_regressor` | 0,872 | 0,114 | 0,532 | 0,119 | 0,925 |
| `histgb_classifier` | 0,651 | 0,240 | 0,539 | 0,086 | 0,785 |

Leitura operacional:

- o `histgb_classifier` manteve `p10` parecido, mas oscilou demais em top-1
- o `poisson_regressor` foi substancialmente mais robusto em `hit1` e `mrr`
- a pior queda mensal do `histgb_classifier` foi muito mais severa

Detalhe mensal do `poisson_regressor`:

| mês | hit1 | p10 | mrr |
|---|---:|---:|---:|
| `2025-01` | 1,000 | 0,542 | 1,000 |
| `2025-02` | 0,893 | 0,382 | 0,914 |
| `2025-03` | 0,722 | 0,672 | 0,861 |

Detalhe mensal do `histgb_classifier`:

| mês | hit1 | p10 | mrr |
|---|---:|---:|---:|
| `2025-01` | 0,742 | 0,587 | 0,871 |
| `2025-02` | 0,321 | 0,418 | 0,539 |
| `2025-03` | 0,889 | 0,611 | 0,944 |

## Decisão de promoção

Champion promovido para Fortaleza:

- `poisson_regressor`

Motivos:

1. ficou no percentil 83 de `hit1`, 83 de `p10` e 100 de `mrr`
2. foi o melhor trade-off entre precisão no topo e consistência de ordenação
3. mostrou estabilidade mensal claramente superior no playoff já rodado
4. é mais explicável e mais barato operacionalmente que manter o fluxo deep como champion da capital

## Situação dos modelos deep

Os modelos deep não foram excluídos do benchmark.

Estado atual:

- a suíte deep foi incorporada ao pipeline
- houve execução rápida de sanity check com artefatos em `outputs/benchmarks/cvli_stochastic_suite_*.{json,csv,md}`
- o comparativo deep completo para Fortaleza ainda não é a base desta promoção

Decisão tomada:

- a promoção atual usa evidência consolidada dos modelos clássicos
- os deep permanecem como trilha de challenger/benchmark, não como champion ativo desta troca

## Expansão para RMF e Interior

Benchmark clássico adicional em 2025:

### RMF

| modelo | hit1 | p10 | mrr |
|---|---:|---:|---:|
| `poisson_regressor` | 1,000 | 0,746 | 1,000 |
| `histgb_classifier` | 1,000 | 0,752 | 1,000 |
| `logit_classifier` | 1,000 | 0,778 | 1,000 |
| `hurdle_logit_poisson` | 1,000 | 0,778 | 1,000 |

Leitura:

- RMF ficou muito forte com toda a família clássica
- `poisson_regressor` não foi o melhor `p10`, mas continuou em patamar alto
- a perda para o melhor `p10` foi pequena o bastante para justificar padronização estadual

### Interior

| modelo | hit1 | p10 | mrr |
|---|---:|---:|---:|
| `poisson_regressor` | 0,940 | 0,728 | 0,970 |
| `histgb_classifier` | 0,974 | 0,705 | 0,987 |
| `logit_classifier` | 0,935 | 0,723 | 0,967 |
| `hurdle_logit_poisson` | 0,957 | 0,723 | 0,979 |

Leitura:

- no Interior, o `poisson_regressor` foi o melhor em `p10`
- `histgb_classifier` ganhou em `hit1`, mas com menor `p10`
- para shortlist operacional, o Poisson encaixa melhor

## Decisão arquitetural atual

Backend promovido:

- `fortaleza`: `poisson_regressor`
- `rmf`: `poisson_regressor`
- `interior`: `poisson_regressor`

Motivação:

1. reduzir peso computacional do app
2. padronizar inferência e manutenção
3. manter métricas competitivas nas três regiões
4. preservar explicabilidade e simplicidade de retreino

## Artefatos gerados

- benchmark geral: `outputs/benchmarks/cvli_stochastic_suite_20260612_220443.{json,csv,md}`
- playoff mensal: `outputs/benchmarks/cvli_finalists_monthly_fortaleza_20260612_223448.{json,csv,md}`
- artefato promovido: `models/active/production/poisson/fortaleza_poisson_regressor.pkl`
- metadados do artefato: `models/active/production/poisson/fortaleza_poisson_regressor.json`
- artefatos estaduais: `models/active/production/poisson/fortaleza_poisson_regressor.pkl`, `models/active/production/poisson/rmf_poisson_regressor.pkl`, `models/active/production/poisson/interior_poisson_regressor.pkl`

## Implicação arquitetural

Após a promoção:

- as três regiões passam a usar `Poisson Ranker`
- o app deixa de depender do fluxo deep como champion de produção
- Champion/Challenger legado de Fortaleza fica desativado por padrão nesta configuração
- health, export estático, API de risco e Hermes passam a anunciar a arquitetura híbrida correta

## Retreino recomendado

Script de promoção/retreino estadual:

```powershell
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

Quando rodar:

- retreino tático regular: `1x por semana`
- retreino obrigatório: após atualização relevante do `ENRIQUECIDO`
- retreino extraordinário: quando `VALIDATION_LOG.md` mostrar piora persistente por 2 ou 3 ciclos

Cadência prática recomendada:

1. atualizar base e processados
2. rodar `scripts\promote_statewide_poisson_regressors.py`
3. subir/reiniciar a aplicação
4. conferir `/api/risk`, `VALIDATION_LOG.md` e export Hermes

Regra simples para operação:

- se a ingestão nova for diária, o modelo não precisa ser retreinado diariamente
- o ideal é retreinar semanalmente ou quando houver mudança material de regime
- se Fortaleza, RMF ou Interior começarem a cair no `P@10` observado, antecipar o retreino
