# Resultado negativo util: colmeia GA por bairro preditivo

Data: 2026-07-22

## Experimento

Foi testada uma camada hibrida em duas etapas:

1. Predicao por bairro usando `HIBRIDO_RECENTE_TOP30`.
2. Geracao de colmeia hexagonal fixa dentro dos bairros preditos.
3. Selecao por GA para comparar tres objetivos:
   - maximizar captura bruta;
   - maximizar captura por area;
   - objetivo balanceado entre captura e eficiencia espacial.

## Resultado

Os tres objetivos convergiram para a mesma solucao:

| Objetivo | Captura | Area | Captura/100 km2 | Raio | Hexagonos |
|---|---:|---:|---:|---:|---:|
| capture | 34,37% | 44,82 km2 | 0,767 | 1,5 km | 8 |
| efficiency | 34,37% | 44,82 km2 | 0,767 | 1,5 km | 8 |
| balanced | 34,37% | 44,82 km2 | 0,767 | 1,5 km | 8 |

## Interpretacao

Nao houve ganho real frente ao baseline por bairro. O resultado indica que a colmeia operacional ficou mais limpa visualmente e sem sobreposicao, mas o GA ainda nao encontrou uma solucao superior.

A causa provavel e que os hexagonos ainda herdam essencialmente o score do bairro. Assim, celulas dentro do mesmo bairro ficam pouco diferenciadas, reduzindo o espaco util de otimizacao do GA.

## Proximo teste

Criar score preditivo individual por hexagono, com features locais proprias, mantendo o bairro como camada operacional superior. O historico de CVLI deve ser usado para avaliacao/backtest e nao para posicionar diretamente os hexagonos.
