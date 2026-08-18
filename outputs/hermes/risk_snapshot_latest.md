# Artefato Hermes de Risco

- Gerado em: 2026-08-18T15:41:06
- Base de dados ate: 2026-07-31
- Base de dados formatada: 31/07/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20260818_154106.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20260818_154106.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20260818_154106.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 335 registros ate 2026-07-31 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 98.5 | crítico | 99.0% (alta) | 94.4% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 2 | JUAZEIRO DO NORTE | 95.5 | crítico | 99.0% (alta) | 92.3% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 3 | SOBRAL | 70.2 | alto | 88.7% (alta) | 86.0% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-07-31 |
| 4 | MARACANAU | 56.5 | alto | 93.3% (alta) | 81.9% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 5 | CRATO | 54.1 | alto | 90.1% (alta) | 79.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 6 | BOA VIAGEM | 51.5 | alto | 89.5% (alta) | 77.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 7 | AMONTADA | 45.9 | moderado | 83.3% (moderada) | 75.2% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 8 | ITAPAJE | 43.2 | moderado | 87.7% (alta) | 73.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 9 | BARBALHA | 39.4 | moderado | 81.9% (moderada) | 70.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 10 | TIANGUA | 34.3 | moderado | 74.9% (moderada) | 68.3% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 11 | MARANGUAPE | 31.1 | moderado | 66.3% (baixa) | 66.2% (moderada) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-07-31 |
| 12 | PENTECOSTE | 26.0 | baixo | 73.5% (moderada) | 63.7% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 13 | PACATUBA | 20.7 | baixo | 66.2% (baixa) | 61.1% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 14 | CASCAVEL | 18.6 | baixo | 66.0% (baixa) | 59.1% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 15 | AQUIRAZ | 16.1 | baixo | 65.8% (baixa) | 57.1% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 16 | CANINDE | 15.7 | baixo | 70.8% (moderada) | 55.4% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 17 | CHOROZINHO | 15.6 | baixo | 65.8% (baixa) | 53.8% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 18 | SAO BENEDITO | 9.9 | baixo | 53.5% (muito baixa) | 51.2% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 19 | ITAPIPOCA | 9.7 | baixo | 37.3% (muito baixa) | 49.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 20 | IGUATU | 8.3 | baixo | 53.4% (muito baixa) | 47.7% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 21 | MORADA NOVA | 7.9 | baixo | 37.5% (muito baixa) | 46.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 22 | QUIXADA | 6.5 | baixo | 37.5% (muito baixa) | 44.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 23 | RUSSAS | 5.6 | baixo | 37.5% (muito baixa) | 42.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 24 | FORQUILHA | 4.2 | baixo | 37.5% (muito baixa) | 40.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 25 | ARACATI | 3.9 | baixo | 37.4% (muito baixa) | 39.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 26 | ITAREMA | 3.9 | baixo | 37.4% (muito baixa) | 37.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 27 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 35.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 28 | PACAJUS | 1.2 | baixo | 35.3% (muito baixa) | 33.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 29 | SAO GONCALO DO AMARANTE | 1.1 | baixo | 35.3% (muito baixa) | 32.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 30 | PARACURU | 1.0 | baixo | 35.3% (muito baixa) | 30.6% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 98.5 | crítico | 99.0% (alta) | 96.5% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 2 | MARACANAU | 56.5 | alto | 93.3% (alta) | 85.0% (muito alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 3 | MARANGUAPE | 31.1 | moderado | 66.3% (baixa) | 76.7% (alta) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-07-31 |
| 4 | PACATUBA | 20.7 | baixo | 66.2% (baixa) | 71.3% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 5 | CASCAVEL | 18.6 | baixo | 66.0% (baixa) | 67.6% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 6 | AQUIRAZ | 16.1 | baixo | 65.8% (baixa) | 63.7% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 7 | CHOROZINHO | 15.6 | baixo | 65.8% (baixa) | 60.3% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 8 | PACAJUS | 1.2 | baixo | 35.3% (muito baixa) | 54.2% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 9 | SAO GONCALO DO AMARANTE | 1.1 | baixo | 35.3% (muito baixa) | 50.8% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 10 | PARACURU | 1.0 | baixo | 35.3% (muito baixa) | 47.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 11 | EUSEBIO | 0.8 | baixo | 35.2% (muito baixa) | 44.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 12 | HORIZONTE | 0.8 | baixo | 47.7% (muito baixa) | 40.8% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-31 |
| 13 | ITAITINGA | 0.6 | baixo | 35.2% (muito baixa) | 37.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 14 | BEBERIBE | 0.6 | baixo | 35.1% (muito baixa) | 34.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 15 | TRAIRI | 0.5 | baixo | 35.1% (muito baixa) | 30.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 16 | PARAIPABA | 0.4 | baixo | 33.7% (muito baixa) | 27.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 17 | GUAIUBA | 0.3 | baixo | 35.1% (muito baixa) | 24.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 18 | PINDORETAMA | 0.3 | baixo | 32.1% (muito baixa) | 20.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 19 | SAO LUIS DO CURU | 0.0 | baixo | 30.8% (muito baixa) | 17.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 95.5 | crítico | 99.0% (alta) | 92.8% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 2 | SOBRAL | 70.2 | alto | 88.7% (alta) | 84.9% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-07-31 |
| 3 | CRATO | 54.1 | alto | 90.1% (alta) | 78.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 4 | BOA VIAGEM | 51.5 | alto | 89.5% (alta) | 75.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 5 | AMONTADA | 45.9 | moderado | 83.3% (moderada) | 70.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 6 | ITAPAJE | 43.2 | moderado | 87.7% (alta) | 67.3% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 7 | BARBALHA | 39.4 | moderado | 81.9% (moderada) | 63.4% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 8 | TIANGUA | 34.3 | moderado | 74.9% (moderada) | 59.3% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 9 | PENTECOSTE | 26.0 | baixo | 73.5% (moderada) | 54.6% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 10 | CANINDE | 15.7 | baixo | 70.8% (moderada) | 49.5% (baixa) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 11 | SAO BENEDITO | 9.9 | baixo | 53.5% (muito baixa) | 45.3% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 12 | ITAPIPOCA | 9.7 | baixo | 37.3% (muito baixa) | 42.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 13 | IGUATU | 8.3 | baixo | 53.4% (muito baixa) | 38.6% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 14 | MORADA NOVA | 7.9 | baixo | 37.5% (muito baixa) | 35.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 15 | QUIXADA | 6.5 | baixo | 37.5% (muito baixa) | 32.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 16 | RUSSAS | 5.6 | baixo | 37.5% (muito baixa) | 28.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 17 | FORQUILHA | 4.2 | baixo | 37.5% (muito baixa) | 25.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 18 | ARACATI | 3.9 | baixo | 37.4% (muito baixa) | 22.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 19 | ITAREMA | 3.9 | baixo | 37.4% (muito baixa) | 18.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 20 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 15.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MESSEJANA | 97.0 | crítico | 95.0% (alta) | 100.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 2 | JOSE DE ALENCAR | 47.6 | moderado | 88.7% (alta) | 87.3% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-31 |
| 3 | SIQUEIRA | 45.4 | moderado | 77.0% (moderada) | 85.2% (muito alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-31 |
| 4 | BARROSO | 42.2 | moderado | 76.4% (moderada) | 82.9% (alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-31 |
| 5 | BARRA DO CEARA | 41.9 | moderado | 46.4% (muito baixa) | 81.2% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 6 | GRANJA LISBOA | 38.0 | moderado | 56.1% (baixa) | 78.7% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-07-31 |
| 7 | JOSE WALTER | 22.8 | baixo | 41.2% (muito baixa) | 73.3% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 8 | PLANALTO AYRTON SENNA | 20.2 | baixo | 40.5% (muito baixa) | 71.2% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 9 | MONDUBIM | 16.2 | baixo | 39.3% (muito baixa) | 68.6% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
| 10 | CONJUNTO PALMEIRAS | 10.2 | baixo | 53.6% (muito baixa) | 65.5% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 11 | PASSARE | 10.2 | baixo | 37.2% (muito baixa) | 64.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 12 | EDSON QUEIROZ | 9.7 | baixo | 53.5% (muito baixa) | 62.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-31 |
| 13 | QUINTINO CUNHA | 9.2 | baixo | 37.4% (muito baixa) | 60.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 14 | VICENTE PINZON | 7.9 | baixo | 37.1% (muito baixa) | 58.8% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 15 | LAGOA REDONDA | 7.7 | baixo | 36.7% (muito baixa) | 57.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 16 | GRANJA PORTUGAL | 7.4 | baixo | 36.9% (muito baixa) | 55.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 17 | PICI | 6.6 | baixo | 36.4% (muito baixa) | 53.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 18 | BOM JARDIM | 5.7 | baixo | 35.6% (muito baixa) | 52.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 19 | JARDIM DAS OLIVEIRAS | 5.5 | baixo | 35.6% (muito baixa) | 50.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 20 | BONSUCESSO | 5.3 | baixo | 35.3% (muito baixa) | 48.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 21 | CARLITO PAMPLONA | 5.1 | baixo | 35.3% (muito baixa) | 47.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 22 | CANINDEZINHO | 5.0 | baixo | 34.7% (muito baixa) | 45.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 23 | CENTRO | 4.1 | baixo | 33.7% (muito baixa) | 44.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 24 | VILA VELHA | 4.1 | baixo | 34.3% (muito baixa) | 42.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 25 | MANOEL SATIRO | 4.0 | baixo | 34.3% (muito baixa) | 40.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 26 | ANTONIO BEZERRA | 3.6 | baixo | 33.8% (muito baixa) | 39.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 27 | PAUPINA | 3.6 | baixo | 33.8% (muito baixa) | 37.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 28 | CAJAZEIRAS | 3.3 | baixo | 33.0% (muito baixa) | 36.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 29 | ANCURI | 2.5 | baixo | 32.2% (muito baixa) | 34.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-31 |
| 30 | FLORESTA | 2.3 | baixo | 31.5% (muito baixa) | 32.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-31 |
