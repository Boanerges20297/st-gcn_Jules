# Artefato Hermes de Risco

- Gerado em: 2026-08-08T11:32:08
- Base de dados ate: 2026-08-05
- Base de dados formatada: 05/08/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20260808_113208.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20260808_113208.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20260808_113208.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 223 registros ate 2026-08-05 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 98.5 | crítico | 99.0% (alta) | 94.2% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 2 | CAUCAIA | 82.2 | crítico | 92.9% (alta) | 89.7% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-05 |
| 3 | SOBRAL | 79.7 | crítico | 93.6% (alta) | 87.6% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-05 |
| 4 | MARACANAU | 73.2 | crítico | 94.5% (alta) | 84.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 5 | CRATO | 52.8 | alto | 89.8% (alta) | 79.6% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 6 | MARANGUAPE | 49.0 | moderado | 77.6% (moderada) | 77.3% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 7 | BOA VIAGEM | 47.8 | moderado | 88.8% (alta) | 75.5% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 8 | ITAPAJE | 42.2 | moderado | 87.6% (alta) | 72.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 9 | CASCAVEL | 31.6 | moderado | 76.5% (moderada) | 69.4% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 10 | AMONTADA | 31.5 | moderado | 84.6% (moderada) | 67.8% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-08-05 |
| 11 | TIANGUA | 31.5 | moderado | 74.5% (moderada) | 66.2% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 12 | HORIZONTE | 30.2 | baixo | 76.4% (moderada) | 64.4% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 13 | PENTECOSTE | 27.1 | baixo | 78.6% (moderada) | 62.2% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 14 | PACATUBA | 19.6 | baixo | 66.1% (baixa) | 59.3% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 15 | AQUIRAZ | 15.4 | baixo | 65.8% (baixa) | 57.0% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 16 | CHOROZINHO | 15.0 | baixo | 65.7% (baixa) | 55.3% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 17 | BARBALHA | 13.6 | baixo | 81.3% (moderada) | 53.5% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 18 | SAO BENEDITO | 8.8 | baixo | 53.4% (muito baixa) | 51.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 19 | ITAPIPOCA | 8.5 | baixo | 37.4% (muito baixa) | 49.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 20 | MORADA NOVA | 6.8 | baixo | 37.5% (muito baixa) | 47.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 21 | QUIXADA | 6.0 | baixo | 37.5% (muito baixa) | 45.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 22 | RUSSAS | 5.2 | baixo | 37.5% (muito baixa) | 44.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 23 | IGUATU | 4.2 | baixo | 37.5% (muito baixa) | 42.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 24 | FORQUILHA | 4.0 | baixo | 37.4% (muito baixa) | 40.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 25 | ITAREMA | 3.7 | baixo | 37.4% (muito baixa) | 39.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 26 | ARACATI | 3.7 | baixo | 37.4% (muito baixa) | 37.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 27 | CANINDE | 2.4 | baixo | 48.0% (muito baixa) | 35.6% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-05 |
| 28 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 33.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 29 | PACAJUS | 1.4 | baixo | 35.3% (muito baixa) | 32.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 30 | PARACURU | 1.1 | baixo | 35.3% (muito baixa) | 30.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 82.2 | crítico | 92.9% (alta) | 92.6% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-05 |
| 2 | MARACANAU | 73.2 | crítico | 94.5% (alta) | 87.5% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 3 | MARANGUAPE | 49.0 | moderado | 77.6% (moderada) | 79.5% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 4 | CASCAVEL | 31.6 | moderado | 76.5% (moderada) | 72.8% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 5 | HORIZONTE | 30.2 | baixo | 76.4% (moderada) | 69.2% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 6 | PACATUBA | 19.6 | baixo | 66.1% (baixa) | 63.8% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 7 | AQUIRAZ | 15.4 | baixo | 65.8% (baixa) | 59.7% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 8 | CHOROZINHO | 15.0 | baixo | 65.7% (baixa) | 56.3% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 9 | PACAJUS | 1.4 | baixo | 35.3% (muito baixa) | 50.3% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 10 | PARACURU | 1.1 | baixo | 35.3% (muito baixa) | 46.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 11 | SAO GONCALO DO AMARANTE | 1.1 | baixo | 35.3% (muito baixa) | 43.6% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 12 | EUSEBIO | 0.8 | baixo | 35.2% (muito baixa) | 40.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 13 | ITAITINGA | 0.7 | baixo | 35.2% (muito baixa) | 36.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 14 | BEBERIBE | 0.6 | baixo | 35.2% (muito baixa) | 33.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 15 | PARAIPABA | 0.5 | baixo | 33.7% (muito baixa) | 30.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 16 | TRAIRI | 0.5 | baixo | 35.1% (muito baixa) | 26.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 17 | GUAIUBA | 0.3 | baixo | 35.1% (muito baixa) | 23.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 18 | PINDORETAMA | 0.3 | baixo | 32.1% (muito baixa) | 20.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 19 | SAO LUIS DO CURU | 0.0 | baixo | 30.8% (muito baixa) | 16.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 98.5 | crítico | 99.0% (alta) | 93.2% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 2 | SOBRAL | 79.7 | crítico | 93.6% (alta) | 86.7% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-05 |
| 3 | CRATO | 52.8 | alto | 89.8% (alta) | 78.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 4 | BOA VIAGEM | 47.8 | moderado | 88.8% (alta) | 74.7% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 5 | ITAPAJE | 42.2 | moderado | 87.6% (alta) | 70.6% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 6 | AMONTADA | 31.5 | moderado | 84.6% (moderada) | 65.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-08-05 |
| 7 | TIANGUA | 31.5 | moderado | 74.5% (moderada) | 62.4% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 8 | PENTECOSTE | 27.1 | baixo | 78.6% (moderada) | 58.5% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 9 | BARBALHA | 13.6 | baixo | 81.3% (moderada) | 52.9% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 10 | SAO BENEDITO | 8.8 | baixo | 53.4% (muito baixa) | 48.9% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 11 | ITAPIPOCA | 8.5 | baixo | 37.4% (muito baixa) | 45.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 12 | MORADA NOVA | 6.8 | baixo | 37.5% (muito baixa) | 42.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 13 | QUIXADA | 6.0 | baixo | 37.5% (muito baixa) | 38.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 14 | RUSSAS | 5.2 | baixo | 37.5% (muito baixa) | 35.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 15 | IGUATU | 4.2 | baixo | 37.5% (muito baixa) | 32.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 16 | FORQUILHA | 4.0 | baixo | 37.4% (muito baixa) | 29.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 17 | ITAREMA | 3.7 | baixo | 37.4% (muito baixa) | 25.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 18 | ARACATI | 3.7 | baixo | 37.4% (muito baixa) | 22.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 19 | CANINDE | 2.4 | baixo | 48.0% (muito baixa) | 19.4% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-05 |
| 20 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 16.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MESSEJANA | 93.2 | crítico | 95.0% (alta) | 100.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 2 | JOSE DE ALENCAR | 46.7 | moderado | 83.7% (moderada) | 87.5% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 3 | SIQUEIRA | 40.9 | moderado | 76.5% (moderada) | 84.4% (alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-05 |
| 4 | BARROSO | 40.0 | moderado | 76.2% (moderada) | 82.6% (alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-05 |
| 5 | BARRA DO CEARA | 37.0 | moderado | 45.1% (muito baixa) | 80.3% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 6 | LAGOA REDONDA | 35.6 | moderado | 75.2% (moderada) | 78.4% (alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-05 |
| 7 | GRANJA LISBOA | 31.3 | moderado | 54.6% (muito baixa) | 75.7% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-05 |
| 8 | JOSE WALTER | 21.2 | baixo | 40.8% (muito baixa) | 71.6% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 9 | PLANALTO AYRTON SENNA | 18.8 | baixo | 40.1% (muito baixa) | 69.4% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 10 | MONDUBIM | 15.1 | baixo | 38.9% (muito baixa) | 66.8% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 11 | PASSARE | 9.5 | baixo | 37.3% (muito baixa) | 63.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 12 | CONJUNTO PALMEIRAS | 9.0 | baixo | 53.5% (muito baixa) | 62.2% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 13 | QUINTINO CUNHA | 8.7 | baixo | 37.4% (muito baixa) | 60.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 14 | EDSON QUEIROZ | 8.5 | baixo | 53.4% (muito baixa) | 59.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 15 | VICENTE PINZON | 7.4 | baixo | 37.1% (muito baixa) | 57.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 16 | GRANJA PORTUGAL | 7.0 | baixo | 36.9% (muito baixa) | 55.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 17 | PICI | 6.2 | baixo | 36.4% (muito baixa) | 53.8% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 18 | BOM JARDIM | 5.3 | baixo | 35.6% (muito baixa) | 52.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 19 | JARDIM DAS OLIVEIRAS | 5.2 | baixo | 35.7% (muito baixa) | 50.4% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 20 | BONSUCESSO | 5.0 | baixo | 35.3% (muito baixa) | 48.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 21 | CARLITO PAMPLONA | 4.8 | baixo | 35.3% (muito baixa) | 47.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 22 | CANINDEZINHO | 4.7 | baixo | 34.7% (muito baixa) | 45.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 23 | CENTRO | 3.9 | baixo | 33.8% (muito baixa) | 43.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 24 | VILA VELHA | 3.8 | baixo | 34.3% (muito baixa) | 42.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 25 | MANOEL SATIRO | 3.8 | baixo | 34.3% (muito baixa) | 40.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 26 | ANTONIO BEZERRA | 3.4 | baixo | 33.8% (muito baixa) | 39.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 27 | PAUPINA | 3.4 | baixo | 33.8% (muito baixa) | 37.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 28 | CAJAZEIRAS | 3.1 | baixo | 33.0% (muito baixa) | 36.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 29 | ANCURI | 2.4 | baixo | 32.2% (muito baixa) | 34.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 30 | FLORESTA | 2.2 | baixo | 31.5% (muito baixa) | 32.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
