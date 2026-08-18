# Artefato Hermes de Risco

- Gerado em: 2026-08-14T15:03:03
- Base de dados ate: 2026-08-05
- Base de dados formatada: 05/08/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20260814_150303.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20260814_150303.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20260814_150303.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 229 registros ate 2026-08-05 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 98.5 | crítico | 99.0% (alta) | 93.6% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 2 | CAUCAIA | 96.2 | crítico | 99.0% (alta) | 91.7% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 3 | SOBRAL | 80.3 | crítico | 93.5% (alta) | 87.2% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-05 |
| 4 | MARACANAU | 72.1 | crítico | 94.2% (alta) | 84.2% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 5 | CRATO | 53.3 | alto | 89.9% (alta) | 79.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 6 | MARANGUAPE | 48.1 | moderado | 77.6% (moderada) | 76.8% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 7 | BOA VIAGEM | 48.1 | moderado | 88.9% (alta) | 75.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 8 | ITAPAJE | 42.2 | moderado | 87.6% (alta) | 72.7% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 9 | PACATUBA | 32.2 | moderado | 76.7% (moderada) | 69.3% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 10 | TIANGUA | 31.8 | moderado | 74.6% (moderada) | 67.6% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 11 | AMONTADA | 31.5 | moderado | 84.6% (moderada) | 66.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-08-05 |
| 12 | CASCAVEL | 31.3 | moderado | 76.5% (moderada) | 64.4% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 13 | HORIZONTE | 30.0 | baixo | 76.4% (moderada) | 62.6% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 14 | PENTECOSTE | 27.1 | baixo | 78.6% (moderada) | 60.5% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 15 | AQUIRAZ | 15.3 | baixo | 65.8% (baixa) | 56.9% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 16 | CHOROZINHO | 14.9 | baixo | 65.7% (baixa) | 55.2% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 17 | BARBALHA | 13.7 | baixo | 81.3% (moderada) | 53.4% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 18 | SAO BENEDITO | 8.8 | baixo | 53.4% (muito baixa) | 51.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 19 | ITAPIPOCA | 8.6 | baixo | 37.4% (muito baixa) | 49.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 20 | MORADA NOVA | 6.8 | baixo | 37.5% (muito baixa) | 47.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 21 | QUIXADA | 6.0 | baixo | 37.5% (muito baixa) | 45.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 22 | RUSSAS | 5.2 | baixo | 37.5% (muito baixa) | 44.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 23 | IGUATU | 4.2 | baixo | 37.5% (muito baixa) | 42.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 24 | FORQUILHA | 4.0 | baixo | 37.4% (muito baixa) | 40.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 25 | ITAREMA | 3.7 | baixo | 37.4% (muito baixa) | 39.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 26 | ARACATI | 3.7 | baixo | 37.4% (muito baixa) | 37.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 27 | CANINDE | 2.4 | baixo | 48.0% (muito baixa) | 35.6% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-05 |
| 28 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 33.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 29 | PACAJUS | 1.3 | baixo | 35.3% (muito baixa) | 32.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 30 | PARACURU | 1.1 | baixo | 35.3% (muito baixa) | 30.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 96.2 | crítico | 99.0% (alta) | 93.9% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 2 | MARACANAU | 72.1 | crítico | 94.2% (alta) | 86.2% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 3 | MARANGUAPE | 48.1 | moderado | 77.6% (moderada) | 78.7% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 4 | PACATUBA | 32.2 | moderado | 76.7% (moderada) | 72.5% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 5 | CASCAVEL | 31.3 | moderado | 76.5% (moderada) | 69.0% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 6 | HORIZONTE | 30.0 | baixo | 76.4% (moderada) | 65.4% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 7 | AQUIRAZ | 15.3 | baixo | 65.8% (baixa) | 59.5% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 8 | CHOROZINHO | 14.9 | baixo | 65.7% (baixa) | 56.1% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 9 | PACAJUS | 1.3 | baixo | 35.3% (muito baixa) | 50.3% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 10 | PARACURU | 1.1 | baixo | 35.3% (muito baixa) | 46.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 11 | SAO GONCALO DO AMARANTE | 1.1 | baixo | 35.3% (muito baixa) | 43.6% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 12 | EUSEBIO | 0.7 | baixo | 35.2% (muito baixa) | 40.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 13 | ITAITINGA | 0.6 | baixo | 35.2% (muito baixa) | 36.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 14 | BEBERIBE | 0.6 | baixo | 35.2% (muito baixa) | 33.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 15 | PARAIPABA | 0.5 | baixo | 33.7% (muito baixa) | 30.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 16 | TRAIRI | 0.5 | baixo | 35.1% (muito baixa) | 26.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 17 | GUAIUBA | 0.3 | baixo | 35.1% (muito baixa) | 23.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 18 | PINDORETAMA | 0.3 | baixo | 32.1% (muito baixa) | 20.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 19 | SAO LUIS DO CURU | 0.0 | baixo | 30.8% (muito baixa) | 16.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 98.5 | crítico | 99.0% (alta) | 93.2% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 2 | SOBRAL | 80.3 | crítico | 93.5% (alta) | 86.8% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-05 |
| 3 | CRATO | 53.3 | alto | 89.9% (alta) | 78.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 4 | BOA VIAGEM | 48.1 | moderado | 88.9% (alta) | 74.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 5 | ITAPAJE | 42.2 | moderado | 87.6% (alta) | 70.5% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 6 | TIANGUA | 31.8 | moderado | 74.6% (moderada) | 65.6% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 7 | AMONTADA | 31.5 | moderado | 84.6% (moderada) | 62.4% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-08-05 |
| 8 | PENTECOSTE | 27.1 | baixo | 78.6% (moderada) | 58.5% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 9 | BARBALHA | 13.7 | baixo | 81.3% (moderada) | 52.9% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 10 | SAO BENEDITO | 8.8 | baixo | 53.4% (muito baixa) | 48.9% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 11 | ITAPIPOCA | 8.6 | baixo | 37.4% (muito baixa) | 45.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
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
| 2 | JOSE DE ALENCAR | 46.4 | moderado | 83.6% (moderada) | 87.4% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-05 |
| 3 | SIQUEIRA | 40.7 | moderado | 76.4% (moderada) | 84.4% (alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-05 |
| 4 | BARROSO | 39.7 | moderado | 76.1% (moderada) | 82.6% (alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-05 |
| 5 | BARRA DO CEARA | 37.6 | moderado | 45.3% (muito baixa) | 80.4% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 6 | LAGOA REDONDA | 35.3 | moderado | 75.2% (moderada) | 78.3% (alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-05 |
| 7 | GRANJA LISBOA | 31.1 | moderado | 54.6% (muito baixa) | 75.7% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-05 |
| 8 | JOSE WALTER | 21.1 | baixo | 40.8% (muito baixa) | 71.5% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 9 | PLANALTO AYRTON SENNA | 18.7 | baixo | 40.0% (muito baixa) | 69.3% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 10 | MONDUBIM | 14.9 | baixo | 38.8% (muito baixa) | 66.8% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-05 |
| 11 | PICI | 14.2 | baixo | 53.8% (muito baixa) | 65.1% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 12 | PASSARE | 9.3 | baixo | 37.3% (muito baixa) | 62.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 13 | CONJUNTO PALMEIRAS | 8.9 | baixo | 53.5% (muito baixa) | 60.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 14 | QUINTINO CUNHA | 8.5 | baixo | 37.4% (muito baixa) | 59.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 15 | EDSON QUEIROZ | 8.2 | baixo | 53.4% (muito baixa) | 57.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-05 |
| 16 | VICENTE PINZON | 7.5 | baixo | 37.3% (muito baixa) | 55.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 17 | GRANJA PORTUGAL | 7.0 | baixo | 37.1% (muito baixa) | 54.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 18 | BOM JARDIM | 5.1 | baixo | 35.7% (muito baixa) | 51.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 19 | JARDIM DAS OLIVEIRAS | 5.0 | baixo | 35.7% (muito baixa) | 50.3% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 20 | BONSUCESSO | 5.0 | baixo | 35.5% (muito baixa) | 48.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 21 | CARLITO PAMPLONA | 4.6 | baixo | 35.3% (muito baixa) | 47.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 22 | CANINDEZINHO | 4.5 | baixo | 34.7% (muito baixa) | 45.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 23 | CENTRO | 3.6 | baixo | 33.8% (muito baixa) | 43.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 24 | VILA VELHA | 3.6 | baixo | 34.3% (muito baixa) | 42.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 25 | MANOEL SATIRO | 3.5 | baixo | 34.3% (muito baixa) | 40.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 26 | CAJAZEIRAS | 3.2 | baixo | 33.4% (muito baixa) | 39.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 27 | ANTONIO BEZERRA | 3.2 | baixo | 33.8% (muito baixa) | 37.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 28 | PAUPINA | 3.1 | baixo | 33.8% (muito baixa) | 36.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 29 | ANCURI | 2.1 | baixo | 32.3% (muito baixa) | 34.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
| 30 | FLORESTA | 1.9 | baixo | 31.5% (muito baixa) | 32.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-05 |
