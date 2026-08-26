# Artefato Hermes de Risco

- Gerado em: 2026-08-26T15:54:52
- Base de dados ate: 2026-08-24
- Base de dados formatada: 24/08/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20260826_155452.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20260826_155452.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20260826_155452.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 136 registros ate 2026-08-24 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 79.9 | crítico | 92.9% (alta) | 94.4% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-24 |
| 2 | JUAZEIRO DO NORTE | 64.4 | alto | 75.9% (moderada) | 89.3% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-24 |
| 3 | SOBRAL | 60.4 | alto | 76.3% (moderada) | 86.8% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-24 |
| 4 | BOA VIAGEM | 54.3 | alto | 90.5% (alta) | 83.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 5 | AMONTADA | 47.6 | moderado | 83.7% (moderada) | 80.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 6 | TIANGUA | 47.2 | moderado | 83.8% (moderada) | 79.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 7 | ITAPAJE | 44.4 | moderado | 83.2% (moderada) | 76.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 8 | MARACANAU | 33.2 | moderado | 77.7% (moderada) | 72.8% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-24 |
| 9 | BARBALHA | 29.3 | baixo | 74.1% (moderada) | 70.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 10 | CRATO | 25.8 | baixo | 82.5% (moderada) | 67.9% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 11 | CASCAVEL | 20.6 | baixo | 76.2% (moderada) | 65.2% (moderada) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-24 |
| 12 | AQUIRAZ | 18.4 | baixo | 66.0% (baixa) | 63.2% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 13 | MORADA NOVA | 14.4 | baixo | 53.8% (muito baixa) | 60.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 14 | ITAITINGA | 13.1 | baixo | 65.5% (baixa) | 58.8% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 15 | RUSSAS | 12.0 | baixo | 53.7% (muito baixa) | 57.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 16 | ITAPIPOCA | 9.1 | baixo | 37.4% (muito baixa) | 54.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 17 | PENTECOSTE | 7.8 | baixo | 53.3% (muito baixa) | 52.8% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 18 | QUIXADA | 6.7 | baixo | 37.5% (muito baixa) | 51.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 19 | SAO BENEDITO | 4.6 | baixo | 37.5% (muito baixa) | 48.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 20 | IGUATU | 4.6 | baixo | 37.5% (muito baixa) | 47.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 21 | BEBERIBE | 4.5 | baixo | 48.5% (muito baixa) | 45.8% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-24 |
| 22 | FORQUILHA | 4.2 | baixo | 37.5% (muito baixa) | 44.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 23 | ARACATI | 4.1 | baixo | 37.5% (muito baixa) | 42.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 24 | ITAREMA | 4.1 | baixo | 37.5% (muito baixa) | 41.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 25 | EUSEBIO | 2.5 | baixo | 35.7% (muito baixa) | 39.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 26 | PARACURU | 2.1 | baixo | 35.5% (muito baixa) | 37.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 27 | PINDORETAMA | 2.0 | baixo | 32.5% (muito baixa) | 35.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 28 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 34.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 29 | CANINDE | 1.4 | baixo | 35.4% (muito baixa) | 32.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 30 | SAO LUIS DO CURU | 1.3 | baixo | 34.3% (muito baixa) | 30.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 79.9 | crítico | 92.9% (alta) | 98.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-24 |
| 2 | MARACANAU | 33.2 | moderado | 77.7% (moderada) | 82.6% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-24 |
| 3 | CASCAVEL | 20.6 | baixo | 76.2% (moderada) | 76.1% (alta) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-24 |
| 4 | AQUIRAZ | 18.4 | baixo | 66.0% (baixa) | 72.2% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 5 | ITAITINGA | 13.1 | baixo | 65.5% (baixa) | 67.6% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 6 | BEBERIBE | 4.5 | baixo | 48.5% (muito baixa) | 62.0% (moderada) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-24 |
| 7 | EUSEBIO | 2.5 | baixo | 35.7% (muito baixa) | 58.2% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 8 | PARACURU | 2.1 | baixo | 35.5% (muito baixa) | 54.7% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 9 | PINDORETAMA | 2.0 | baixo | 32.5% (muito baixa) | 51.4% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 10 | SAO LUIS DO CURU | 1.3 | baixo | 34.3% (muito baixa) | 47.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 11 | PACATUBA | 1.2 | baixo | 35.3% (muito baixa) | 44.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 12 | PARAIPABA | 1.1 | baixo | 35.3% (muito baixa) | 41.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 13 | SAO GONCALO DO AMARANTE | 1.1 | baixo | 35.3% (muito baixa) | 37.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 14 | MARANGUAPE | 0.7 | baixo | 65.2% (baixa) | 34.4% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-24 |
| 15 | TRAIRI | 0.7 | baixo | 35.2% (muito baixa) | 31.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 16 | CHOROZINHO | 0.5 | baixo | 47.6% (muito baixa) | 27.7% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-24 |
| 17 | PACAJUS | 0.5 | baixo | 47.6% (muito baixa) | 24.3% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-24 |
| 18 | GUAIUBA | 0.4 | baixo | 35.1% (muito baixa) | 21.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 19 | HORIZONTE | 0.0 | baixo | 65.0% (baixa) | 17.5% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-08-24 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 64.4 | alto | 75.9% (moderada) | 89.3% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-24 |
| 2 | SOBRAL | 60.4 | alto | 76.3% (moderada) | 85.3% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-24 |
| 3 | BOA VIAGEM | 54.3 | alto | 90.5% (alta) | 80.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 4 | AMONTADA | 47.6 | moderado | 83.7% (moderada) | 76.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 5 | TIANGUA | 47.2 | moderado | 83.8% (moderada) | 72.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 6 | ITAPAJE | 44.4 | moderado | 83.2% (moderada) | 69.1% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 7 | BARBALHA | 29.3 | baixo | 74.1% (moderada) | 62.6% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 8 | CRATO | 25.8 | baixo | 82.5% (moderada) | 58.7% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 9 | MORADA NOVA | 14.4 | baixo | 53.8% (muito baixa) | 53.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 10 | RUSSAS | 12.0 | baixo | 53.7% (muito baixa) | 49.2% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 11 | ITAPIPOCA | 9.1 | baixo | 37.4% (muito baixa) | 45.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 12 | PENTECOSTE | 7.8 | baixo | 53.3% (muito baixa) | 42.0% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 13 | QUIXADA | 6.7 | baixo | 37.5% (muito baixa) | 38.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 14 | SAO BENEDITO | 4.6 | baixo | 37.5% (muito baixa) | 35.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 15 | IGUATU | 4.6 | baixo | 37.5% (muito baixa) | 31.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 16 | FORQUILHA | 4.2 | baixo | 37.5% (muito baixa) | 28.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 17 | ARACATI | 4.1 | baixo | 37.5% (muito baixa) | 25.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 18 | ITAREMA | 4.1 | baixo | 37.5% (muito baixa) | 22.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 19 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 18.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 20 | CANINDE | 1.4 | baixo | 35.4% (muito baixa) | 15.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MESSEJANA | 60.6 | alto | 85.9% (alta) | 94.7% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-08-24 |
| 2 | JOSE WALTER | 57.3 | alto | 78.6% (moderada) | 92.1% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 3 | PASSARE | 39.9 | moderado | 75.9% (moderada) | 85.3% (muito alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-08-24 |
| 4 | BARRA DO CEARA | 38.5 | moderado | 45.5% (muito baixa) | 83.3% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 5 | LAGOA REDONDA | 36.5 | moderado | 80.3% (moderada) | 81.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 6 | CONJUNTO PALMEIRAS | 32.2 | moderado | 74.8% (moderada) | 78.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 7 | CAJAZEIRAS | 28.7 | baixo | 74.0% (moderada) | 75.7% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-08-24 |
| 8 | GRANJA LISBOA | 26.4 | baixo | 42.2% (muito baixa) | 73.5% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 9 | MONDUBIM | 25.7 | baixo | 53.5% (muito baixa) | 71.7% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-24 |
| 10 | JOSE DE ALENCAR | 21.0 | baixo | 77.1% (moderada) | 68.7% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 11 | PLANALTO AYRTON SENNA | 16.6 | baixo | 39.4% (muito baixa) | 65.9% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 12 | BARROSO | 14.7 | baixo | 53.9% (muito baixa) | 63.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-08-24 |
| 13 | SIQUEIRA | 13.0 | baixo | 38.2% (muito baixa) | 61.6% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-08-24 |
| 14 | QUINTINO CUNHA | 8.8 | baixo | 37.4% (muito baixa) | 58.8% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 15 | JACARECANGA | 8.1 | baixo | 53.3% (muito baixa) | 57.1% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-08-24 |
| 16 | GRANJA PORTUGAL | 7.2 | baixo | 37.1% (muito baixa) | 55.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 17 | PICI | 7.1 | baixo | 36.5% (muito baixa) | 53.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 18 | VICENTE PINZON | 7.0 | baixo | 37.3% (muito baixa) | 52.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 19 | BOM JARDIM | 5.2 | baixo | 35.7% (muito baixa) | 50.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 20 | JARDIM DAS OLIVEIRAS | 5.1 | baixo | 35.7% (muito baixa) | 48.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 21 | BONSUCESSO | 5.1 | baixo | 35.5% (muito baixa) | 46.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 22 | CARLITO PAMPLONA | 4.7 | baixo | 35.3% (muito baixa) | 45.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 23 | EDSON QUEIROZ | 4.5 | baixo | 34.5% (muito baixa) | 43.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 24 | CANINDEZINHO | 4.0 | baixo | 34.7% (muito baixa) | 42.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 25 | VILA VELHA | 3.7 | baixo | 34.3% (muito baixa) | 40.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 26 | MANOEL SATIRO | 3.6 | baixo | 34.3% (muito baixa) | 38.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 27 | ANTONIO BEZERRA | 3.2 | baixo | 33.8% (muito baixa) | 37.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 28 | PAUPINA | 3.2 | baixo | 33.8% (muito baixa) | 35.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 29 | CENTRO | 3.2 | baixo | 33.8% (muito baixa) | 34.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
| 30 | PARQUE DOIS IRMAOS | 2.0 | baixo | 32.5% (muito baixa) | 32.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-08-24 |
