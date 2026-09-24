# Artefato Hermes de Risco

- Gerado em: 2026-09-24T14:42:42
- Base de dados ate: 2026-09-21
- Base de dados formatada: 21/09/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20260924_144242.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20260924_144242.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20260924_144242.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 161 registros ate 2026-09-21 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 96.2 | crítico | 99.0% (alta) | 93.8% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 2 | JUAZEIRO DO NORTE | 93.9 | crítico | 99.0% (alta) | 91.8% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 3 | SOBRAL | 91.5 | crítico | 99.0% (alta) | 89.8% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 4 | MARACANAU | 61.8 | alto | 78.6% (moderada) | 83.0% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 5 | TIANGUA | 55.2 | alto | 90.3% (alta) | 80.2% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 6 | RUSSAS | 44.7 | moderado | 88.3% (alta) | 76.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 7 | BOA VIAGEM | 40.1 | moderado | 76.0% (moderada) | 74.4% (alta) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-09-21 |
| 8 | ITAPIPOCA | 32.7 | moderado | 74.9% (moderada) | 71.5% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 9 | ITAPAJE | 31.6 | moderado | 84.7% (moderada) | 69.7% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-21 |
| 10 | QUIXADA | 30.8 | baixo | 74.6% (moderada) | 68.0% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 11 | BARBALHA | 27.7 | baixo | 73.8% (moderada) | 65.9% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 12 | CRATO | 20.7 | baixo | 53.8% (muito baixa) | 63.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-09-21 |
| 13 | IGUATU | 15.5 | baixo | 76.5% (moderada) | 60.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-21 |
| 14 | SAO BENEDITO | 8.5 | baixo | 53.4% (muito baixa) | 57.8% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-21 |
| 15 | MORADA NOVA | 7.3 | baixo | 37.5% (muito baixa) | 56.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 16 | AMONTADA | 6.5 | baixo | 37.5% (muito baixa) | 54.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 17 | FORQUILHA | 4.0 | baixo | 37.4% (muito baixa) | 52.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 18 | ARACATI | 4.0 | baixo | 37.4% (muito baixa) | 50.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 19 | ITAREMA | 3.9 | baixo | 37.4% (muito baixa) | 49.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 20 | BEBERIBE | 3.9 | baixo | 48.4% (muito baixa) | 47.5% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 21 | CASCAVEL | 3.7 | baixo | 48.4% (muito baixa) | 45.9% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 22 | PENTECOSTE | 3.4 | baixo | 37.4% (muito baixa) | 44.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 23 | EUSEBIO | 2.8 | baixo | 35.7% (muito baixa) | 42.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 24 | AQUIRAZ | 2.5 | baixo | 35.7% (muito baixa) | 40.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 25 | PARACURU | 2.4 | baixo | 35.6% (muito baixa) | 39.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 26 | PINDORETAMA | 2.3 | baixo | 32.6% (muito baixa) | 37.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 27 | SAO GONCALO DO AMARANTE | 1.9 | baixo | 65.5% (baixa) | 36.1% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 28 | PARAIPABA | 1.7 | baixo | 47.9% (muito baixa) | 34.4% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 29 | SAO LUIS DO CURU | 1.6 | baixo | 34.4% (muito baixa) | 32.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 30 | PACATUBA | 1.6 | baixo | 47.9% (muito baixa) | 31.3% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 96.2 | crítico | 99.0% (alta) | 97.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 2 | MARACANAU | 61.8 | alto | 78.6% (moderada) | 86.9% (muito alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 3 | BEBERIBE | 3.9 | baixo | 48.4% (muito baixa) | 72.2% (alta) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 4 | CASCAVEL | 3.7 | baixo | 48.4% (muito baixa) | 68.8% (moderada) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 5 | EUSEBIO | 2.8 | baixo | 35.7% (muito baixa) | 65.3% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 6 | AQUIRAZ | 2.5 | baixo | 35.7% (muito baixa) | 61.9% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 7 | PARACURU | 2.4 | baixo | 35.6% (muito baixa) | 58.6% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 8 | PINDORETAMA | 2.3 | baixo | 32.6% (muito baixa) | 55.2% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 9 | SAO GONCALO DO AMARANTE | 1.9 | baixo | 65.5% (baixa) | 51.8% (moderada) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 10 | PARAIPABA | 1.7 | baixo | 47.9% (muito baixa) | 48.4% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 11 | SAO LUIS DO CURU | 1.6 | baixo | 34.4% (muito baixa) | 45.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 12 | PACATUBA | 1.6 | baixo | 47.9% (muito baixa) | 41.7% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 13 | TRAIRI | 1.0 | baixo | 35.3% (muito baixa) | 38.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 14 | CHOROZINHO | 0.8 | baixo | 35.2% (muito baixa) | 34.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 15 | GUAIUBA | 0.8 | baixo | 35.2% (muito baixa) | 31.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 16 | PACAJUS | 0.5 | baixo | 47.6% (muito baixa) | 28.2% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 17 | ITAITINGA | 0.5 | baixo | 47.6% (muito baixa) | 24.8% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-21 |
| 18 | MARANGUAPE | 0.4 | baixo | 35.1% (muito baixa) | 21.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 19 | HORIZONTE | 0.0 | baixo | 35.0% (muito baixa) | 18.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JUAZEIRO DO NORTE | 93.9 | crítico | 99.0% (alta) | 92.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 2 | SOBRAL | 91.5 | crítico | 99.0% (alta) | 88.3% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 3 | TIANGUA | 55.2 | alto | 90.3% (alta) | 78.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 4 | RUSSAS | 44.7 | moderado | 88.3% (alta) | 73.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 5 | BOA VIAGEM | 40.1 | moderado | 76.0% (moderada) | 69.8% (moderada) | Tensão territorial; Atividade recente e vizinhança; Sinal Poisson do ranking operacional | 2026-09-21 |
| 6 | ITAPIPOCA | 32.7 | moderado | 74.9% (moderada) | 65.4% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 7 | ITAPAJE | 31.6 | moderado | 84.7% (moderada) | 62.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-21 |
| 8 | QUIXADA | 30.8 | baixo | 74.6% (moderada) | 58.7% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 9 | BARBALHA | 27.7 | baixo | 73.8% (moderada) | 55.0% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 10 | CRATO | 20.7 | baixo | 53.8% (muito baixa) | 50.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-09-21 |
| 11 | IGUATU | 15.5 | baixo | 76.5% (moderada) | 46.5% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-21 |
| 12 | SAO BENEDITO | 8.5 | baixo | 53.4% (muito baixa) | 42.1% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-21 |
| 13 | MORADA NOVA | 7.3 | baixo | 37.5% (muito baixa) | 38.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 14 | AMONTADA | 6.5 | baixo | 37.5% (muito baixa) | 35.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 15 | FORQUILHA | 4.0 | baixo | 37.4% (muito baixa) | 31.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 16 | ARACATI | 4.0 | baixo | 37.4% (muito baixa) | 28.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 17 | ITAREMA | 3.9 | baixo | 37.4% (muito baixa) | 25.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 18 | PENTECOSTE | 3.4 | baixo | 37.4% (muito baixa) | 22.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 19 | GROAIRAS | 1.4 | baixo | 35.0% (muito baixa) | 18.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 20 | CANINDE | 0.8 | baixo | 35.2% (muito baixa) | 15.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MESSEJANA | 94.7 | crítico | 90.0% (alta) | 100.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 2 | ANTONIO BEZERRA | 41.5 | moderado | 77.3% (moderada) | 87.1% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 3 | LAGOA REDONDA | 35.3 | moderado | 85.2% (alta) | 83.9% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-21 |
| 4 | CONJUNTO PALMEIRAS | 32.4 | moderado | 74.6% (moderada) | 81.5% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-21 |
| 5 | BARRA DO CEARA | 28.6 | baixo | 42.9% (muito baixa) | 78.9% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 6 | MONDUBIM | 22.2 | baixo | 53.8% (muito baixa) | 75.6% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-09-21 |
| 7 | PASSARE | 18.7 | baixo | 76.9% (moderada) | 73.0% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-21 |
| 8 | JOSE WALTER | 18.4 | baixo | 40.0% (muito baixa) | 71.4% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 9 | GRANJA LISBOA | 18.4 | baixo | 40.0% (muito baixa) | 69.9% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 10 | MANOEL SATIRO | 13.3 | baixo | 76.3% (moderada) | 66.9% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-21 |
| 11 | PLANALTO AYRTON SENNA | 12.7 | baixo | 38.0% (muito baixa) | 65.2% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-21 |
| 12 | SIQUEIRA | 10.0 | baixo | 37.2% (muito baixa) | 62.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 13 | BARROSO | 8.8 | baixo | 37.4% (muito baixa) | 61.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 14 | QUINTINO CUNHA | 6.9 | baixo | 37.5% (muito baixa) | 58.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 15 | GRANJA PORTUGAL | 5.8 | baixo | 37.2% (muito baixa) | 57.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 16 | VICENTE PINZON | 5.6 | baixo | 37.4% (muito baixa) | 55.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 17 | JOSE DE ALENCAR | 5.5 | baixo | 35.4% (muito baixa) | 53.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 18 | PICI | 5.1 | baixo | 36.6% (muito baixa) | 52.3% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 19 | BOM JARDIM | 4.2 | baixo | 35.7% (muito baixa) | 50.4% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 20 | CARLITO PAMPLONA | 4.2 | baixo | 35.5% (muito baixa) | 48.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 21 | JARDIM DAS OLIVEIRAS | 4.1 | baixo | 35.7% (muito baixa) | 47.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 22 | BONSUCESSO | 4.1 | baixo | 35.5% (muito baixa) | 45.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 23 | CANINDEZINHO | 3.3 | baixo | 34.7% (muito baixa) | 44.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 24 | EDSON QUEIROZ | 3.2 | baixo | 34.5% (muito baixa) | 42.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 25 | VILA VELHA | 3.0 | baixo | 34.3% (muito baixa) | 40.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 26 | CAJAZEIRAS | 2.8 | baixo | 33.6% (muito baixa) | 39.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 27 | PAUPINA | 2.6 | baixo | 33.8% (muito baixa) | 37.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 28 | CENTRO | 2.6 | baixo | 33.8% (muito baixa) | 36.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 29 | JACARECANGA | 2.3 | baixo | 32.4% (muito baixa) | 34.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
| 30 | PARQUE DOIS IRMAOS | 1.6 | baixo | 32.4% (muito baixa) | 32.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-21 |
