# Artefato Hermes de Risco

- Gerado em: 2026-07-24T16:52:05
- Base de dados ate: 2026-07-24
- Base de dados formatada: 24/07/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20260724_165205.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20260724_165205.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20260724_165205.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 193 registros ate 2026-07-24 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 97.2 | crítico | 99.0% (alta) | 96.4% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-07-24 |
| 2 | SOBRAL | 72.8 | crítico | 88.6% (alta) | 89.8% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-07-24 |
| 3 | CAUCAIA | 65.3 | alto | 76.8% (moderada) | 86.6% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-07-24 |
| 4 | MARACANAU | 52.7 | alto | 93.2% (alta) | 82.4% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 5 | CRATO | 50.0 | moderado | 79.6% (moderada) | 80.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 6 | AMONTADA | 45.4 | moderado | 78.4% (moderada) | 77.7% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 7 | ITAPAJE | 41.3 | moderado | 82.5% (moderada) | 75.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 8 | BARBALHA | 39.9 | moderado | 82.1% (moderada) | 73.4% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 9 | BOA VIAGEM | 32.9 | moderado | 90.0% (alta) | 70.3% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-24 |
| 10 | SAO BENEDITO | 28.0 | baixo | 74.0% (moderada) | 67.7% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 11 | PENTECOSTE | 26.5 | baixo | 73.6% (moderada) | 65.9% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 12 | ITAPIPOCA | 22.5 | baixo | 83.2% (moderada) | 63.5% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 13 | CANINDE | 16.4 | baixo | 70.8% (moderada) | 60.6% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 14 | MORADA NOVA | 15.4 | baixo | 54.9% (muito baixa) | 58.8% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 15 | IGUATU | 12.1 | baixo | 54.4% (muito baixa) | 56.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-24 |
| 16 | RUSSAS | 12.1 | baixo | 54.4% (muito baixa) | 54.9% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-24 |
| 17 | MARANGUAPE | 10.3 | baixo | 37.6% (muito baixa) | 53.0% (moderada) | Sinal Poisson do ranking operacional; Atividade recente e vizinhança; Tensão territorial | 2026-07-24 |
| 18 | TIANGUA | 9.2 | baixo | 38.7% (muito baixa) | 51.2% (moderada) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 19 | QUIXADA | 9.1 | baixo | 38.7% (muito baixa) | 49.6% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 20 | FORQUILHA | 6.8 | baixo | 38.2% (muito baixa) | 47.5% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 21 | ITAREMA | 6.7 | baixo | 38.1% (muito baixa) | 45.9% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 22 | ARACATI | 6.6 | baixo | 38.1% (muito baixa) | 44.3% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 23 | PACATUBA | 4.8 | baixo | 35.8% (muito baixa) | 42.4% (baixa) | Sinal Poisson do ranking operacional; Atividade recente e vizinhança; Tensão territorial | 2026-07-24 |
| 24 | GROAIRAS | 4.2 | baixo | 35.1% (muito baixa) | 40.7% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 25 | PACAJUS | 4.0 | baixo | 48.3% (muito baixa) | 39.0% (baixa) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 26 | HORIZONTE | 3.6 | baixo | 48.3% (muito baixa) | 37.4% (baixa) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 27 | CASCAVEL | 3.3 | baixo | 48.3% (muito baixa) | 35.8% (baixa) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 28 | SAO GONCALO DO AMARANTE | 3.0 | baixo | 35.9% (muito baixa) | 34.1% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 29 | PARACURU | 3.0 | baixo | 35.8% (muito baixa) | 32.5% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 30 | EUSEBIO | 2.7 | baixo | 35.8% (muito baixa) | 30.9% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 65.3 | alto | 76.8% (moderada) | 95.6% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-07-24 |
| 2 | MARACANAU | 52.7 | alto | 93.2% (alta) | 88.7% (muito alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 3 | MARANGUAPE | 10.3 | baixo | 37.6% (muito baixa) | 73.7% (alta) | Sinal Poisson do ranking operacional; Atividade recente e vizinhança; Tensão territorial | 2026-07-24 |
| 4 | PACATUBA | 4.8 | baixo | 35.8% (muito baixa) | 68.8% (moderada) | Sinal Poisson do ranking operacional; Atividade recente e vizinhança; Tensão territorial | 2026-07-24 |
| 5 | PACAJUS | 4.0 | baixo | 48.3% (muito baixa) | 65.3% (moderada) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 6 | HORIZONTE | 3.6 | baixo | 48.3% (muito baixa) | 61.8% (moderada) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 7 | CASCAVEL | 3.3 | baixo | 48.3% (muito baixa) | 58.4% (moderada) | CVLI recente na janela de 30 dias; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 8 | SAO GONCALO DO AMARANTE | 3.0 | baixo | 35.9% (muito baixa) | 55.0% (moderada) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 9 | PARACURU | 3.0 | baixo | 35.8% (muito baixa) | 51.7% (moderada) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 10 | EUSEBIO | 2.7 | baixo | 35.8% (muito baixa) | 48.2% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 11 | ITAITINGA | 2.5 | baixo | 35.8% (muito baixa) | 44.8% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 12 | BEBERIBE | 2.4 | baixo | 35.8% (muito baixa) | 41.5% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 13 | TRAIRI | 2.4 | baixo | 35.8% (muito baixa) | 38.1% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 14 | PARAIPABA | 2.3 | baixo | 34.4% (muito baixa) | 34.8% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 15 | AQUIRAZ | 2.1 | baixo | 35.8% (muito baixa) | 31.4% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 16 | GUAIUBA | 2.1 | baixo | 35.8% (muito baixa) | 28.1% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 17 | PINDORETAMA | 2.1 | baixo | 32.8% (muito baixa) | 24.7% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 18 | SAO LUIS DO CURU | 1.7 | baixo | 31.5% (muito baixa) | 21.3% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |
| 19 | CHOROZINHO | 1.7 | baixo | 32.6% (muito baixa) | 18.0% (baixa) | Atividade recente e vizinhança; Sinal Poisson do ranking operacional; Tensão territorial | 2026-07-24 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 97.2 | crítico | 99.0% (alta) | 94.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-07-24 |
| 2 | SOBRAL | 72.8 | crítico | 88.6% (alta) | 85.9% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-07-24 |
| 3 | CRATO | 50.0 | moderado | 79.6% (moderada) | 78.2% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 4 | AMONTADA | 45.4 | moderado | 78.4% (moderada) | 74.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 5 | ITAPAJE | 41.3 | moderado | 82.5% (moderada) | 70.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 6 | BARBALHA | 39.9 | moderado | 82.1% (moderada) | 66.7% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 7 | BOA VIAGEM | 32.9 | moderado | 90.0% (alta) | 62.1% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-24 |
| 8 | SAO BENEDITO | 28.0 | baixo | 74.0% (moderada) | 58.0% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 9 | PENTECOSTE | 26.5 | baixo | 73.6% (moderada) | 54.5% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 10 | ITAPIPOCA | 22.5 | baixo | 83.2% (moderada) | 50.5% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 11 | CANINDE | 16.4 | baixo | 70.8% (moderada) | 46.2% (baixa) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 12 | MORADA NOVA | 15.4 | baixo | 54.9% (muito baixa) | 42.8% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 13 | IGUATU | 12.1 | baixo | 54.4% (muito baixa) | 38.9% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-24 |
| 14 | RUSSAS | 12.1 | baixo | 54.4% (muito baixa) | 35.8% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-07-24 |
| 15 | TIANGUA | 9.2 | baixo | 38.7% (muito baixa) | 32.0% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 16 | QUIXADA | 9.1 | baixo | 38.7% (muito baixa) | 28.9% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 17 | FORQUILHA | 6.8 | baixo | 38.2% (muito baixa) | 25.3% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 18 | ITAREMA | 6.7 | baixo | 38.1% (muito baixa) | 22.1% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 19 | ARACATI | 6.6 | baixo | 38.1% (muito baixa) | 18.9% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |
| 20 | GROAIRAS | 4.2 | baixo | 35.1% (muito baixa) | 15.3% (baixa) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-07-24 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MESSEJANA | 81.4 | crítico | 77.9% (moderada) | 100.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-24 |
| 2 | GRANJA LISBOA | 44.3 | moderado | 56.8% (baixa) | 88.9% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-07-24 |
| 3 | BARRA DO CEARA | 41.0 | moderado | 46.2% (muito baixa) | 86.4% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-24 |
| 4 | JOSE DE ALENCAR | 35.4 | moderado | 79.6% (moderada) | 83.1% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-07-24 |
| 5 | JOSE WALTER | 22.3 | baixo | 41.1% (muito baixa) | 77.4% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-24 |
| 6 | PLANALTO AYRTON SENNA | 19.8 | baixo | 40.4% (muito baixa) | 75.1% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-24 |
| 7 | MONDUBIM | 15.9 | baixo | 39.2% (muito baixa) | 72.3% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-24 |
| 8 | LAGOA REDONDA | 14.7 | baixo | 39.8% (muito baixa) | 70.5% (alta) | Atividade recente e vizinhança; Tensão territorial; Sinal Poisson do ranking operacional | 2026-07-24 |
| 9 | SIQUEIRA | 12.0 | baixo | 37.8% (muito baixa) | 68.1% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-24 |
| 10 | CONJUNTO PALMEIRAS | 11.6 | baixo | 53.6% (muito baixa) | 66.4% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 11 | EDSON QUEIROZ | 11.0 | baixo | 53.6% (muito baixa) | 64.7% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 12 | BARROSO | 10.6 | baixo | 37.2% (muito baixa) | 63.0% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-07-24 |
| 13 | PASSARE | 9.6 | baixo | 37.3% (muito baixa) | 61.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 14 | QUINTINO CUNHA | 9.1 | baixo | 37.4% (muito baixa) | 59.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 15 | VICENTE PINZON | 7.8 | baixo | 37.1% (muito baixa) | 57.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 16 | GRANJA PORTUGAL | 7.3 | baixo | 36.9% (muito baixa) | 55.8% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 17 | FLORESTA | 7.2 | baixo | 53.2% (muito baixa) | 54.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 18 | ANCURI | 6.9 | baixo | 53.2% (muito baixa) | 52.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-07-24 |
| 19 | PICI | 6.5 | baixo | 36.4% (muito baixa) | 50.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 20 | BOM JARDIM | 5.6 | baixo | 35.6% (muito baixa) | 49.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 21 | JARDIM DAS OLIVEIRAS | 5.4 | baixo | 35.6% (muito baixa) | 47.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 22 | BONSUCESSO | 5.2 | baixo | 35.3% (muito baixa) | 46.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 23 | CARLITO PAMPLONA | 5.0 | baixo | 35.3% (muito baixa) | 44.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 24 | CANINDEZINHO | 4.9 | baixo | 34.7% (muito baixa) | 42.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 25 | CENTRO | 4.0 | baixo | 33.7% (muito baixa) | 41.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 26 | VILA VELHA | 4.0 | baixo | 34.3% (muito baixa) | 39.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 27 | MANOEL SATIRO | 3.9 | baixo | 34.3% (muito baixa) | 37.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 28 | ANTONIO BEZERRA | 3.5 | baixo | 33.8% (muito baixa) | 36.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 29 | PAUPINA | 3.5 | baixo | 33.8% (muito baixa) | 34.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
| 30 | CAJAZEIRAS | 3.0 | baixo | 33.0% (muito baixa) | 32.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-07-24 |
