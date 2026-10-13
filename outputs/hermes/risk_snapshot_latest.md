# Artefato Hermes de Risco

- Gerado em: 2026-10-13T16:11:51
- Base de dados ate: 2026-10-09
- Base de dados formatada: 09/10/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20261013_161151.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20261013_161151.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20261013_161151.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 244 registros ate 2026-10-09 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOBRAL | 98.5 | crítico | 99.0% (alta) | 98.8% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 2 | CAUCAIA | 82.2 | crítico | 92.9% (alta) | 93.7% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-10-09 |
| 3 | ITAPAJE | 42.4 | moderado | 77.5% (moderada) | 83.4% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-10-09 |
| 4 | JUAZEIRO DO NORTE | 41.5 | moderado | 82.6% (moderada) | 81.6% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-10-09 |
| 5 | SAO BENEDITO | 39.1 | moderado | 76.8% (moderada) | 79.6% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-10-09 |
| 6 | TIANGUA | 38.9 | moderado | 90.7% (alta) | 77.9% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-10-09 |
| 7 | ITAPIPOCA | 31.3 | moderado | 74.5% (moderada) | 74.7% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-10-09 |
| 8 | RUSSAS | 16.4 | baixo | 81.6% (moderada) | 69.8% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 9 | BOA VIAGEM | 10.4 | baixo | 53.6% (muito baixa) | 66.9% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 10 | IGUATU | 8.5 | baixo | 53.4% (muito baixa) | 65.0% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 11 | CRATO | 7.5 | baixo | 37.5% (muito baixa) | 63.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 12 | MORADA NOVA | 5.4 | baixo | 37.5% (muito baixa) | 61.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 13 | QUIXADA | 5.1 | baixo | 37.5% (muito baixa) | 59.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 14 | MARACANAU | 4.7 | baixo | 37.5% (muito baixa) | 57.8% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 15 | AMONTADA | 4.5 | baixo | 37.5% (muito baixa) | 56.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 16 | LIMOEIRO DO NORTE | 4.5 | baixo | 37.5% (muito baixa) | 54.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 17 | FORQUILHA | 3.2 | baixo | 37.3% (muito baixa) | 52.8% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 18 | ITAREMA | 3.1 | baixo | 37.3% (muito baixa) | 51.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 19 | ARACATI | 3.1 | baixo | 37.3% (muito baixa) | 49.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 20 | AQUIRAZ | 3.0 | baixo | 48.2% (muito baixa) | 48.0% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-10-09 |
| 21 | CASCAVEL | 2.8 | baixo | 35.7% (muito baixa) | 46.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 22 | BEBERIBE | 2.5 | baixo | 35.6% (muito baixa) | 44.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 23 | PENTECOSTE | 2.4 | baixo | 37.2% (muito baixa) | 43.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 24 | TRAIRI | 2.2 | baixo | 48.0% (muito baixa) | 41.5% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-10-09 |
| 25 | PARACURU | 2.2 | baixo | 35.6% (muito baixa) | 39.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 26 | PINDORETAMA | 2.1 | baixo | 33.1% (muito baixa) | 38.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 27 | EUSEBIO | 2.0 | baixo | 35.5% (muito baixa) | 36.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 28 | CARIRE | 1.9 | baixo | 37.0% (muito baixa) | 35.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 29 | PACATUBA | 1.5 | baixo | 35.4% (muito baixa) | 33.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 30 | PARAIPABA | 1.4 | baixo | 35.4% (muito baixa) | 31.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 82.2 | crítico | 92.9% (alta) | 100.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-10-09 |
| 2 | MARACANAU | 4.7 | baixo | 37.5% (muito baixa) | 76.4% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 3 | AQUIRAZ | 3.0 | baixo | 48.2% (muito baixa) | 72.6% (alta) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-10-09 |
| 4 | CASCAVEL | 2.8 | baixo | 35.7% (muito baixa) | 69.2% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 5 | BEBERIBE | 2.5 | baixo | 35.6% (muito baixa) | 65.8% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 6 | TRAIRI | 2.2 | baixo | 48.0% (muito baixa) | 62.4% (moderada) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-10-09 |
| 7 | PARACURU | 2.2 | baixo | 35.6% (muito baixa) | 59.0% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 8 | PINDORETAMA | 2.1 | baixo | 33.1% (muito baixa) | 55.7% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 9 | EUSEBIO | 2.0 | baixo | 35.5% (muito baixa) | 52.4% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 10 | PACATUBA | 1.5 | baixo | 35.4% (muito baixa) | 48.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 11 | PARAIPABA | 1.4 | baixo | 35.4% (muito baixa) | 45.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 12 | SAO LUIS DO CURU | 1.3 | baixo | 34.6% (muito baixa) | 42.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 13 | CHOROZINHO | 0.8 | baixo | 35.2% (muito baixa) | 38.6% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 14 | SAO GONCALO DO AMARANTE | 0.7 | baixo | 35.2% (muito baixa) | 35.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 15 | MARANGUAPE | 0.5 | baixo | 35.1% (muito baixa) | 31.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 16 | GUAIUBA | 0.4 | baixo | 35.1% (muito baixa) | 28.6% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 17 | ITAITINGA | 0.2 | baixo | 35.1% (muito baixa) | 25.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 18 | PACAJUS | 0.1 | baixo | 35.0% (muito baixa) | 21.8% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 19 | HORIZONTE | 0.0 | baixo | 35.0% (muito baixa) | 18.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOBRAL | 98.5 | crítico | 99.0% (alta) | 96.3% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 2 | ITAPAJE | 42.4 | moderado | 77.5% (moderada) | 81.7% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-10-09 |
| 3 | JUAZEIRO DO NORTE | 41.5 | moderado | 82.6% (moderada) | 78.4% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; CVLI recente na janela de 30 dias | 2026-10-09 |
| 4 | SAO BENEDITO | 39.1 | moderado | 76.8% (moderada) | 74.7% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-10-09 |
| 5 | TIANGUA | 38.9 | moderado | 90.7% (alta) | 71.5% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-10-09 |
| 6 | ITAPIPOCA | 31.3 | moderado | 74.5% (moderada) | 66.8% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-10-09 |
| 7 | RUSSAS | 16.4 | baixo | 81.6% (moderada) | 60.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 8 | BOA VIAGEM | 10.4 | baixo | 53.6% (muito baixa) | 56.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 9 | IGUATU | 8.5 | baixo | 53.4% (muito baixa) | 52.7% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 10 | CRATO | 7.5 | baixo | 37.5% (muito baixa) | 49.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 11 | MORADA NOVA | 5.4 | baixo | 37.5% (muito baixa) | 45.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 12 | QUIXADA | 5.1 | baixo | 37.5% (muito baixa) | 42.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 13 | AMONTADA | 4.5 | baixo | 37.5% (muito baixa) | 39.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 14 | LIMOEIRO DO NORTE | 4.5 | baixo | 37.5% (muito baixa) | 36.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 15 | FORQUILHA | 3.2 | baixo | 37.3% (muito baixa) | 32.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 16 | ITAREMA | 3.1 | baixo | 37.3% (muito baixa) | 29.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 17 | ARACATI | 3.1 | baixo | 37.3% (muito baixa) | 26.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 18 | PENTECOSTE | 2.4 | baixo | 37.2% (muito baixa) | 23.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 19 | CARIRE | 1.9 | baixo | 37.0% (muito baixa) | 19.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 20 | CANINDE | 0.2 | baixo | 35.0% (muito baixa) | 16.3% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JANGURUSSU | 81.4 | crítico | 82.9% (moderada) | 100.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 2 | MESSEJANA | 40.9 | moderado | 86.5% (alta) | 90.5% (muito alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-10-09 |
| 3 | BARRA DO CEARA | 20.1 | baixo | 40.5% (muito baixa) | 81.7% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 4 | LAGOA REDONDA | 12.0 | baixo | 81.1% (moderada) | 77.3% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 5 | GRANJA LISBOA | 6.7 | baixo | 37.5% (muito baixa) | 73.9% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 6 | MONDUBIM | 6.2 | baixo | 37.6% (muito baixa) | 72.2% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 7 | CONJUNTO PALMEIRAS | 6.1 | baixo | 53.1% (muito baixa) | 70.7% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 8 | PLANALTO AYRTON SENNA | 5.8 | baixo | 37.5% (muito baixa) | 69.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 9 | ANTONIO BEZERRA | 5.5 | baixo | 53.0% (muito baixa) | 67.4% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-10-09 |
| 10 | PREFEITO JOSE WALTER | 4.1 | baixo | 36.1% (muito baixa) | 65.4% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-10-09 |
| 11 | SIQUEIRA | 3.8 | baixo | 37.4% (muito baixa) | 63.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 12 | QUINTINO CUNHA | 3.4 | baixo | 37.4% (muito baixa) | 62.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 13 | BARROSO | 3.4 | baixo | 37.4% (muito baixa) | 60.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 14 | PASSARE | 3.3 | baixo | 37.4% (muito baixa) | 59.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 15 | VICENTE PINZON | 3.0 | baixo | 37.3% (muito baixa) | 57.3% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 16 | CRISTO REDENTOR | 3.0 | baixo | 37.3% (muito baixa) | 55.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 17 | CARLITO PAMPLONA | 2.7 | baixo | 37.3% (muito baixa) | 54.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 18 | GRANJA PORTUGAL | 2.6 | baixo | 37.3% (muito baixa) | 52.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 19 | PICI | 2.6 | baixo | 37.3% (muito baixa) | 51.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 20 | MANOEL SATIRO | 2.5 | baixo | 37.3% (muito baixa) | 49.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 21 | JARDIM DAS OLIVEIRAS | 2.5 | baixo | 37.3% (muito baixa) | 47.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 22 | BOM JARDIM | 2.5 | baixo | 37.2% (muito baixa) | 46.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 23 | CANINDEZINHO | 2.4 | baixo | 37.2% (muito baixa) | 44.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 24 | EDSON QUEIROZ | 2.4 | baixo | 37.2% (muito baixa) | 43.2% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 25 | VILA VELHA | 2.3 | baixo | 37.2% (muito baixa) | 41.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 26 | CAJAZEIRAS | 2.3 | baixo | 37.2% (muito baixa) | 40.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 27 | BONSUCESSO | 2.3 | baixo | 37.2% (muito baixa) | 38.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 28 | PAUPINA | 2.3 | baixo | 37.2% (muito baixa) | 37.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 29 | CENTRO | 2.3 | baixo | 37.2% (muito baixa) | 35.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
| 30 | PARQUE DOIS IRMAOS | 2.2 | baixo | 37.2% (muito baixa) | 33.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-10-09 |
