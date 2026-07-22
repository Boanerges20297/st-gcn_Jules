# Greedy Espacial e Busca de Raio

## Hipotese

Depois de filtrar bairros recorrentes, a selecao de microareas por pontos CVLI passados com decaimento temporal pode capturar melhor o local futuro dos CVLIs do que selecionar apenas bairros inteiros.

## Implementacao testada

Script: `scripts/experiments/fortaleza_hybrid_capture_spike.py`

Foram usados:
- filtro de elegibilidade `TOP30`;
- pontos CVLI reais com latitude/longitude;
- celulas espaciais geradas a partir do historico passado ate a data de predicao;
- score decaido por recencia;
- selecao greedy das celulas mais fortes;
- avaliacao por captura dos CVLIs futuros dentro de um raio operacional.

Importante: raio em km nao e area em km2. A area aproximada de cada celula circular e `pi * raio^2`; a area total bruta reportada e uma aproximacao sem descontar sobreposicoes.

## Resultado observado

Horizonte de 30 dias:

Baseline bairro, `HIBRIDO_RECENTE_TOP30`, `K=20`:
- captura media: aproximadamente 67,5%.

Greedy espacial, `TOP30`, 30 celulas, raio 1,5 km:
- captura media: aproximadamente 70,9%;
- area bruta aproximada: 212,1 km2.

Busca em grade:
- 40 celulas, raio 2,0 km: captura aproximadamente 85,7%, mas area bruta aproximada 502,7 km2;
- 30 celulas, raio 2,0 km: captura aproximadamente 83,0%, area bruta aproximada 377,0 km2;
- 20 celulas, raio 2,0 km: captura aproximadamente 77,5%, area bruta aproximada 251,3 km2;
- 30 celulas, raio 1,5 km: captura aproximadamente 70,9%, area bruta aproximada 212,1 km2.

## Interpretacao provisoria

O ganho espacial existe, mas cresce junto com a area coberta. Portanto, a pergunta deixa de ser apenas "qual captura mais CVLI?" e passa a ser uma otimizacao multiobjetivo:

- maximizar captura futura;
- minimizar area coberta;
- reduzir sobreposicao;
- manter legibilidade operacional;
- preservar distribuicao por bairros prioritarios.

## Implicacao para algoritmo genetico

O GA passa a fazer sentido na proxima fase porque o problema agora nao e mais escolher um unico raio, mas equilibrar varios parametros simultaneamente:

- raio;
- numero de celulas;
- penalidade de area;
- penalidade de sobreposicao;
- limite maximo por bairro;
- bonus de contiguidade;
- pesos de recencia e historico.

Antes do GA, a busca em grade simples serviu como baseline honesto para evitar complexidade prematura.

