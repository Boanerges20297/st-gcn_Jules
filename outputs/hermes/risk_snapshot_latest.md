# Artefato Hermes de Risco

- Gerado em: 2026-09-28T16:10:50
- Base de dados ate: 2026-09-28
- Base de dados formatada: 28/09/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20260928_161050.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20260928_161050.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20260928_161050.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 165 registros ate 2026-09-28 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOBRAL | 98.5 | crítico | 99.0% (alta) | 95.9% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 2 | CAUCAIA | 78.4 | crítico | 92.9% (alta) | 90.4% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-09-28 |
| 3 | MARACANAU | 75.3 | crítico | 85.1% (alta) | 88.2% (muito alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-09-28 |
| 4 | JUAZEIRO DO NORTE | 71.9 | crítico | 93.8% (alta) | 86.0% (muito alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 5 | TIANGUA | 48.3 | moderado | 89.0% (alta) | 79.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 6 | RUSSAS | 32.9 | moderado | 89.7% (alta) | 75.2% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 7 | BOA VIAGEM | 31.5 | moderado | 74.7% (moderada) | 73.4% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 8 | SAO BENEDITO | 28.7 | baixo | 73.9% (moderada) | 71.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 9 | IGUATU | 28.4 | baixo | 83.9% (moderada) | 69.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 10 | BARBALHA | 25.6 | baixo | 73.4% (moderada) | 67.5% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 11 | AQUIRAZ | 18.3 | baixo | 66.0% (baixa) | 64.5% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 12 | TRAIRI | 16.7 | baixo | 65.9% (baixa) | 62.6% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 13 | ITAPAJE | 15.2 | baixo | 81.5% (moderada) | 60.7% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 14 | CRATO | 13.2 | baixo | 53.8% (muito baixa) | 58.8% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 15 | ITAPIPOCA | 11.7 | baixo | 53.7% (muito baixa) | 56.9% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 16 | QUIXADA | 10.4 | baixo | 53.6% (muito baixa) | 55.1% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 17 | MORADA NOVA | 6.1 | baixo | 37.6% (muito baixa) | 52.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 18 | AMONTADA | 5.2 | baixo | 37.5% (muito baixa) | 50.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 19 | FORQUILHA | 3.6 | baixo | 37.4% (muito baixa) | 49.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 20 | ARACATI | 3.5 | baixo | 37.4% (muito baixa) | 47.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 21 | ITAREMA | 3.5 | baixo | 37.4% (muito baixa) | 45.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 22 | CASCAVEL | 3.4 | baixo | 35.9% (muito baixa) | 44.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 23 | BEBERIBE | 3.2 | baixo | 35.8% (muito baixa) | 42.6% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 24 | PENTECOSTE | 3.1 | baixo | 37.3% (muito baixa) | 41.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 25 | EUSEBIO | 3.1 | baixo | 35.8% (muito baixa) | 39.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 26 | PARACURU | 2.6 | baixo | 35.7% (muito baixa) | 37.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 27 | PINDORETAMA | 2.6 | baixo | 32.7% (muito baixa) | 36.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 28 | SAO LUIS DO CURU | 1.8 | baixo | 34.4% (muito baixa) | 34.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 29 | PARAIPABA | 1.7 | baixo | 35.4% (muito baixa) | 32.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 30 | SAO GONCALO DO AMARANTE | 1.6 | baixo | 35.4% (muito baixa) | 31.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 78.4 | crítico | 92.9% (alta) | 94.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-09-28 |
| 2 | MARACANAU | 75.3 | crítico | 85.1% (alta) | 90.0% (muito alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-09-28 |
| 3 | AQUIRAZ | 18.3 | baixo | 66.0% (baixa) | 74.8% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 4 | TRAIRI | 16.7 | baixo | 65.9% (baixa) | 71.1% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 5 | CASCAVEL | 3.4 | baixo | 35.9% (muito baixa) | 65.1% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 6 | BEBERIBE | 3.2 | baixo | 35.8% (muito baixa) | 61.7% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 7 | EUSEBIO | 3.1 | baixo | 35.8% (muito baixa) | 58.3% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 8 | PARACURU | 2.6 | baixo | 35.7% (muito baixa) | 54.9% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 9 | PINDORETAMA | 2.6 | baixo | 32.7% (muito baixa) | 51.6% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 10 | SAO LUIS DO CURU | 1.8 | baixo | 34.4% (muito baixa) | 48.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 11 | PARAIPABA | 1.7 | baixo | 35.4% (muito baixa) | 44.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 12 | SAO GONCALO DO AMARANTE | 1.6 | baixo | 35.4% (muito baixa) | 41.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 13 | PACATUBA | 1.5 | baixo | 35.4% (muito baixa) | 38.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 14 | CHOROZINHO | 0.8 | baixo | 35.2% (muito baixa) | 34.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 15 | GUAIUBA | 0.8 | baixo | 35.2% (muito baixa) | 31.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 16 | MARANGUAPE | 0.4 | baixo | 35.1% (muito baixa) | 27.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 17 | PACAJUS | 0.2 | baixo | 47.5% (muito baixa) | 24.4% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-28 |
| 18 | ITAITINGA | 0.2 | baixo | 35.0% (muito baixa) | 21.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 19 | HORIZONTE | 0.0 | baixo | 35.0% (muito baixa) | 17.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOBRAL | 98.5 | crítico | 99.0% (alta) | 94.7% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 2 | JUAZEIRO DO NORTE | 71.9 | crítico | 93.8% (alta) | 86.4% (muito alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 3 | TIANGUA | 48.3 | moderado | 89.0% (alta) | 78.7% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 4 | RUSSAS | 32.9 | moderado | 89.7% (alta) | 72.6% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 5 | BOA VIAGEM | 31.5 | moderado | 74.7% (moderada) | 69.1% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 6 | SAO BENEDITO | 28.7 | baixo | 73.9% (moderada) | 65.5% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 7 | IGUATU | 28.4 | baixo | 83.9% (moderada) | 62.2% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 8 | BARBALHA | 25.6 | baixo | 73.4% (moderada) | 58.6% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 9 | ITAPAJE | 15.2 | baixo | 81.5% (moderada) | 53.4% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 10 | CRATO | 13.2 | baixo | 53.8% (muito baixa) | 49.8% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 11 | ITAPIPOCA | 11.7 | baixo | 53.7% (muito baixa) | 46.4% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 12 | QUIXADA | 10.4 | baixo | 53.6% (muito baixa) | 43.0% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 13 | MORADA NOVA | 6.1 | baixo | 37.6% (muito baixa) | 39.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 14 | AMONTADA | 5.2 | baixo | 37.5% (muito baixa) | 35.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 15 | FORQUILHA | 3.6 | baixo | 37.4% (muito baixa) | 32.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 16 | ARACATI | 3.5 | baixo | 37.4% (muito baixa) | 29.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 17 | ITAREMA | 3.5 | baixo | 37.4% (muito baixa) | 25.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 18 | PENTECOSTE | 3.1 | baixo | 37.3% (muito baixa) | 22.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 19 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 19.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 20 | CANINDE | 0.6 | baixo | 35.2% (muito baixa) | 15.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MESSEJANA | 86.7 | crítico | 87.9% (alta) | 100.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-09-28 |
| 2 | LAGOA REDONDA | 51.2 | alto | 89.6% (alta) | 90.5% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 3 | CONJUNTO PALMEIRAS | 32.0 | moderado | 74.7% (moderada) | 83.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 4 | BARRA DO CEARA | 30.2 | baixo | 43.3% (muito baixa) | 81.3% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 5 | ANTONIO BEZERRA | 28.6 | baixo | 79.0% (moderada) | 79.2% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 6 | MONDUBIM | 19.8 | baixo | 53.8% (muito baixa) | 75.1% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-09-28 |
| 7 | JOSE WALTER | 19.4 | baixo | 40.3% (muito baixa) | 73.5% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 8 | GRANJA LISBOA | 19.4 | baixo | 40.3% (muito baixa) | 71.9% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 9 | PLANALTO AYRTON SENNA | 13.3 | baixo | 38.3% (muito baixa) | 68.6% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 10 | PASSARE | 12.4 | baixo | 53.8% (muito baixa) | 66.8% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 11 | SIQUEIRA | 9.7 | baixo | 37.3% (muito baixa) | 64.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 12 | BARROSO | 8.5 | baixo | 37.4% (muito baixa) | 62.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 13 | QUINTINO CUNHA | 7.2 | baixo | 37.5% (muito baixa) | 60.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 14 | GRANJA PORTUGAL | 6.0 | baixo | 37.2% (muito baixa) | 58.8% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 15 | VICENTE PINZON | 5.8 | baixo | 37.4% (muito baixa) | 57.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 16 | PICI | 5.3 | baixo | 36.6% (muito baixa) | 55.4% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 17 | JOSE DE ALENCAR | 4.7 | baixo | 35.5% (muito baixa) | 53.8% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 18 | CARLITO PAMPLONA | 4.6 | baixo | 35.5% (muito baixa) | 52.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 19 | BOM JARDIM | 4.4 | baixo | 35.7% (muito baixa) | 50.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 20 | JARDIM DAS OLIVEIRAS | 4.3 | baixo | 35.7% (muito baixa) | 49.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 21 | BONSUCESSO | 4.3 | baixo | 35.5% (muito baixa) | 47.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 22 | MANOEL SATIRO | 4.2 | baixo | 34.7% (muito baixa) | 45.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 23 | CANINDEZINHO | 3.4 | baixo | 34.7% (muito baixa) | 44.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 24 | EDSON QUEIROZ | 3.3 | baixo | 34.5% (muito baixa) | 42.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 25 | VILA VELHA | 3.1 | baixo | 34.3% (muito baixa) | 41.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 26 | CAJAZEIRAS | 2.9 | baixo | 33.6% (muito baixa) | 39.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 27 | PAUPINA | 2.7 | baixo | 33.8% (muito baixa) | 37.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 28 | CENTRO | 2.7 | baixo | 33.8% (muito baixa) | 36.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 29 | JACARECANGA | 2.4 | baixo | 32.4% (muito baixa) | 34.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 30 | PARQUE DOIS IRMAOS | 1.7 | baixo | 32.4% (muito baixa) | 32.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
