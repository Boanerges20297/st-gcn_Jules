# Algoritmo Genetico Espacial Multiobjetivo

## Hipotese

O algoritmo genetico pode melhorar a captura espacial futura ao otimizar simultaneamente raio, quantidade de celulas, penalidade de area e penalidade de sobreposicao.

## Implementacao testada

Script: `scripts/experiments/fortaleza_hybrid_capture_spike.py`

Parametros otimizados:
- raio em km;
- numero de celulas selecionadas;
- penalidade por area;
- penalidade de sobreposicao.

O GA foi rodado com populacao pequena e poucas geracoes, usando uma amostra temporal para busca. O melhor gene foi validado em todas as 109 janelas de avaliacao do horizonte de 30 dias.

## Resultado observado

Baseline bairro:
- `HIBRIDO_RECENTE_TOP30`, `K=20`, horizonte 30 dias: captura media aproximada de 67,5%.

Greedy espacial:
- 30 celulas, raio 1,5 km: captura media aproximada de 70,9%;
- area bruta aproximada: 212,1 km2.

GA espacial validado em todas as janelas:
- raio aproximado: 1,85 km;
- celulas selecionadas: 34;
- captura media aproximada: 83,6%;
- area bruta aproximada: 366,7 km2.

## Interpretacao provisoria

O GA encontrou ganho real de captura, mas com aumento relevante da area operacional. Isso confirma que o problema e multiobjetivo: captura isolada nao basta.

Para uso operacional, a fitness final deve explicitar penalidade de area e talvez incluir restricoes mais fortes:
- teto de area total;
- teto por bairro;
- penalidade por sobreposicao real;
- bonus por contiguidade;
- bonus por legibilidade em campo.

## Implicacao para dashboard

A camada espacial otimizada nao deve ser exibida como mapa multicolorido. A melhor representacao inicial e um modo operacional separado com poucas zonas em destaque, borda unica e ranking lateral.

