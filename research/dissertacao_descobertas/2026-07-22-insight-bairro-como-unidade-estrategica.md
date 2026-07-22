# Insight experimental: bairro como unidade estrategica principal

Data: 2026-07-22

## Contexto

Foram testadas alternativas para refinar a indicacao espacial de risco de CVLI em Fortaleza:

- ranking por bairro;
- greedy espacial por pontos historicos;
- colmeia hexagonal dentro dos bairros preditos;
- GA com objetivos de captura, eficiencia por area e objetivo balanceado;
- variacao de raio por tamanho do bairro;
- score local por hexagono.

## Resultado consolidado

| Metodo | Captura futura | Area | Observacao |
|---|---:|---:|---|
| Bairro top 30 | 89,04% | n/a | Melhor captura geral |
| Spatial greedy 40 zonas, raio 2 km | 85,73% | 502,65 km2 | Alta captura, area operacional muito ampla |
| Colmeia GA sem sinal local | 18,70% | 39,21 km2 | Area menor, baixa captura |
| Colmeia GA com sinal local 0,25 | 19,96% | 39,21 km2 | Pequena melhora, ainda insuficiente |
| Colmeia GA com sinal local 0,50 | 19,81% | 38,39 km2 | Melhor eficiencia, baixa captura |

## Interpretacao

Nao houve ganho real ao substituir ou refinar automaticamente a decisao por bairro usando hexagonos. A melhor captura futura continuou no ranking por bairro.

O resultado sugere que, neste momento, a camada mais defensavel para uso estrategico e academico e:

1. indicar bairros de maior risco estimado de CVLI;
2. apresentar evidencias e fatores de risco;
3. deixar a escolha da zona tática dentro do bairro para o gestor e equipes de campo.

## Insight operacional

Para o gestor, a pergunta central nao e "qual ponto exato tera CVLI", mas "quais bairros exigem maior atencao preventiva nos proximos dias".

Assim, o dashboard deve evitar linguagem deterministica. A recomendacao deve ser expressa como:

- bairros de maior risco estimado;
- areas prioritarias para avaliacao operacional;
- apoio a decisao, nao determinacao automatica de policiamento.

## Decisao metodologica provisoria

Manter bairro como unidade estrategica principal. Hexagonos podem permanecer como experimento secundario, mas nao devem ser apresentados como ganho real ate que superem o baseline por bairro ou demonstrem trade-off operacional claramente superior.
