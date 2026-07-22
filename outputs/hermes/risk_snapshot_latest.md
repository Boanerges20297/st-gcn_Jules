# Artefato Hermes de Risco

- Gerado em: 2026-07-22T19:37:47
- Base de dados ate: 2026-07-17
- Base de dados formatada: 17/07/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20260722_193747.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20260722_193747.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20260722_193747.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 192 registros ate 2026-07-17 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 93.9 | crítico | 95.0% (alta) | 95.6% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 2 | JUAZEIRO DO NORTE | 80.2 | crítico | 92.6% (alta) | 91.2% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-07-17 |
| 3 | SOBRAL | 70.0 | alto | 83.4% (moderada) | 87.5% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 4 | MARACANAU | 67.7 | alto | 93.5% (alta) | 85.5% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 5 | BOA VIAGEM | 47.4 | moderado | 89.1% (alta) | 79.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 6 | CRATO | 39.5 | moderado | 81.0% (moderada) | 76.5% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 7 | AMONTADA | 32.0 | moderado | 74.8% (moderada) | 73.4% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 8 | ITAPAJE | 27.3 | baixo | 79.0% (moderada) | 70.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 9 | IGUATU | 26.6 | baixo | 78.9% (moderada) | 69.1% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 10 | ITAPIPOCA | 25.2 | baixo | 88.3% (alta) | 67.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 11 | BARBALHA | 24.7 | baixo | 73.5% (moderada) | 65.6% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 12 | MORADA NOVA | 17.0 | baixo | 54.9% (muito baixa) | 62.5% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 13 | HORIZONTE | 15.3 | baixo | 65.8% (baixa) | 60.5% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 14 | CANINDE | 14.0 | baixo | 65.5% (baixa) | 58.7% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 15 | RUSSAS | 13.8 | baixo | 54.6% (muito baixa) | 57.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 16 | TIANGUA | 11.6 | baixo | 54.4% (muito baixa) | 55.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-17 |
| 17 | SAO BENEDITO | 9.8 | baixo | 54.1% (muito baixa) | 53.1% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-17 |
| 18 | MARANGUAPE | 9.4 | baixo | 37.3% (muito baixa) | 51.4% (moderada) | Sinal Poisson do ranking operacional; Atividade recente e vizinhança; Tensão territorial | 2026-07-17 |
| 19 | QUIXADA | 8.5 | baixo | 38.7% (muito baixa) | 49.6% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 20 | FORQUILHA | 5.9 | baixo | 38.1% (muito baixa) | 47.5% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 21 | ITAREMA | 5.8 | baixo | 38.1% (muito baixa) | 45.9% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 22 | ARACATI | 5.8 | baixo | 38.1% (muito baixa) | 44.3% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 23 | PACAJUS | 5.0 | baixo | 48.2% (muito baixa) | 42.6% (baixa) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 24 | PENTECOSTE | 4.9 | baixo | 37.8% (muito baixa) | 41.0% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 25 | PACATUBA | 4.6 | baixo | 35.8% (muito baixa) | 39.3% (baixa) | Sinal Poisson do ranking operacional; Atividade recente e vizinhança; Tensão territorial | 2026-07-17 |
| 26 | CASCAVEL | 4.2 | baixo | 48.3% (muito baixa) | 37.7% (baixa) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 27 | GROAIRAS | 3.5 | baixo | 35.2% (muito baixa) | 36.0% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 28 | ITAITINGA | 3.0 | baixo | 48.3% (muito baixa) | 34.3% (baixa) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 29 | TRAIRI | 3.0 | baixo | 48.3% (muito baixa) | 32.7% (baixa) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 30 | SAO GONCALO DO AMARANTE | 2.9 | baixo | 35.8% (muito baixa) | 31.1% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 93.9 | crítico | 95.0% (alta) | 96.3% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 2 | MARACANAU | 67.7 | alto | 93.5% (alta) | 87.7% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 3 | HORIZONTE | 15.3 | baixo | 65.8% (baixa) | 74.0% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 4 | MARANGUAPE | 9.4 | baixo | 37.3% (muito baixa) | 69.5% (moderada) | Sinal Poisson do ranking operacional; Atividade recente e vizinhança; Tensão territorial | 2026-07-17 |
| 5 | PACAJUS | 5.0 | baixo | 48.2% (muito baixa) | 65.3% (moderada) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 6 | PACATUBA | 4.6 | baixo | 35.8% (muito baixa) | 61.8% (moderada) | Sinal Poisson do ranking operacional; Atividade recente e vizinhança; Tensão territorial | 2026-07-17 |
| 7 | CASCAVEL | 4.2 | baixo | 48.3% (muito baixa) | 58.5% (moderada) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 8 | ITAITINGA | 3.0 | baixo | 48.3% (muito baixa) | 54.9% (moderada) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 9 | TRAIRI | 3.0 | baixo | 48.3% (muito baixa) | 51.6% (moderada) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 10 | SAO GONCALO DO AMARANTE | 2.9 | baixo | 35.8% (muito baixa) | 48.2% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 11 | PARACURU | 2.8 | baixo | 35.8% (muito baixa) | 44.8% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 12 | EUSEBIO | 2.6 | baixo | 35.8% (muito baixa) | 41.5% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 13 | BEBERIBE | 2.4 | baixo | 35.8% (muito baixa) | 38.1% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 14 | PARAIPABA | 2.2 | baixo | 34.4% (muito baixa) | 34.7% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 15 | GUAIUBA | 2.1 | baixo | 35.8% (muito baixa) | 31.4% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 16 | AQUIRAZ | 2.1 | baixo | 35.8% (muito baixa) | 28.0% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 17 | PINDORETAMA | 2.0 | baixo | 32.8% (muito baixa) | 24.7% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 18 | SAO LUIS DO CURU | 1.7 | baixo | 31.5% (muito baixa) | 21.3% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |
| 19 | CHOROZINHO | 1.7 | baixo | 32.6% (muito baixa) | 17.9% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-17 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 80.2 | crítico | 92.6% (alta) | 93.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-07-17 |
| 2 | SOBRAL | 70.0 | alto | 83.4% (moderada) | 87.5% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 3 | BOA VIAGEM | 47.4 | moderado | 89.1% (alta) | 79.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 4 | CRATO | 39.5 | moderado | 81.0% (moderada) | 74.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 5 | AMONTADA | 32.0 | moderado | 74.8% (moderada) | 69.2% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 6 | ITAPAJE | 27.3 | baixo | 79.0% (moderada) | 65.0% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 7 | IGUATU | 26.6 | baixo | 78.9% (moderada) | 61.7% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 8 | ITAPIPOCA | 25.2 | baixo | 88.3% (alta) | 58.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 9 | BARBALHA | 24.7 | baixo | 73.5% (moderada) | 55.0% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 10 | MORADA NOVA | 17.0 | baixo | 54.9% (muito baixa) | 50.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 11 | CANINDE | 14.0 | baixo | 65.5% (baixa) | 46.2% (baixa) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 12 | RUSSAS | 13.8 | baixo | 54.6% (muito baixa) | 43.0% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 13 | TIANGUA | 11.6 | baixo | 54.4% (muito baixa) | 39.3% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-17 |
| 14 | SAO BENEDITO | 9.8 | baixo | 54.1% (muito baixa) | 35.8% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-17 |
| 15 | QUIXADA | 8.5 | baixo | 38.7% (muito baixa) | 32.3% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 16 | FORQUILHA | 5.9 | baixo | 38.1% (muito baixa) | 28.6% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 17 | ITAREMA | 5.8 | baixo | 38.1% (muito baixa) | 25.4% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 18 | ARACATI | 5.8 | baixo | 38.1% (muito baixa) | 22.2% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 19 | PENTECOSTE | 4.9 | baixo | 37.8% (muito baixa) | 18.9% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |
| 20 | GROAIRAS | 3.5 | baixo | 35.2% (muito baixa) | 15.4% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-17 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MESSEJANA | 86.7 | crítico | 82.9% (moderada) | 99.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 2 | GRANJA LISBOA | 69.1 | alto | 83.8% (moderada) | 92.8% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 3 | JOSE DE ALENCAR | 47.0 | moderado | 78.6% (moderada) | 85.6% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 4 | BARRA DO CEARA | 44.1 | moderado | 47.0% (muito baixa) | 83.3% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 5 | CONJUNTO PALMEIRAS | 30.9 | baixo | 74.5% (moderada) | 78.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 6 | EDSON QUEIROZ | 30.2 | baixo | 74.3% (moderada) | 76.6% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-17 |
| 7 | JOSE WALTER | 22.3 | baixo | 41.1% (muito baixa) | 73.0% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 8 | PLANALTO AYRTON SENNA | 19.7 | baixo | 40.4% (muito baixa) | 70.8% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 9 | VARJOTA | 15.8 | baixo | 53.8% (muito baixa) | 68.3% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-07-17 |
| 10 | MONDUBIM | 15.8 | baixo | 39.1% (muito baixa) | 66.7% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 11 | PASSARE | 14.7 | baixo | 53.8% (muito baixa) | 64.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-07-17 |
| 12 | LAGOA REDONDA | 14.6 | baixo | 39.7% (muito baixa) | 63.3% (moderada) | Atividade recente e vizinhança; Tensão territorial; Sinal Poisson do ranking operacional | 2026-07-17 |
| 13 | FLORESTA | 12.7 | baixo | 76.2% (moderada) | 61.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 14 | SIQUEIRA | 11.9 | baixo | 37.8% (muito baixa) | 59.6% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 15 | BARROSO | 10.5 | baixo | 37.2% (muito baixa) | 57.6% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-17 |
| 16 | QUINTINO CUNHA | 9.0 | baixo | 37.4% (muito baixa) | 55.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 17 | ANCURI | 8.1 | baixo | 53.3% (muito baixa) | 54.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 18 | CAJAZEIRAS | 8.1 | baixo | 53.3% (muito baixa) | 52.4% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-17 |
| 19 | VICENTE PINZON | 7.6 | baixo | 37.1% (muito baixa) | 50.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 20 | GRANJA PORTUGAL | 7.2 | baixo | 36.9% (muito baixa) | 49.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 21 | PICI | 6.4 | baixo | 36.4% (muito baixa) | 47.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 22 | BONSUCESSO | 5.7 | baixo | 35.2% (muito baixa) | 45.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 23 | BOM JARDIM | 5.4 | baixo | 35.6% (muito baixa) | 44.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 24 | JARDIM DAS OLIVEIRAS | 5.3 | baixo | 35.7% (muito baixa) | 42.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 25 | CARLITO PAMPLONA | 4.9 | baixo | 35.3% (muito baixa) | 40.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 26 | CANINDEZINHO | 4.7 | baixo | 34.7% (muito baixa) | 39.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 27 | CENTRO | 3.9 | baixo | 33.8% (muito baixa) | 37.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 28 | VILA VELHA | 3.8 | baixo | 34.3% (muito baixa) | 35.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 29 | MANOEL SATIRO | 3.8 | baixo | 34.3% (muito baixa) | 34.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
| 30 | ANTONIO BEZERRA | 3.4 | baixo | 33.8% (muito baixa) | 32.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-17 |
