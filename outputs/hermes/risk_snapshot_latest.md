# Artefato Hermes de Risco

- Gerado em: 2026-10-13T11:34:32
- Base de dados ate: 2026-09-28
- Base de dados formatada: 28/09/2026
- Origem: src/core/orchestrator.py:get_combined_risk
- Fonte oficial para o Hermes: outputs/hermes/
- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_20261013_113432.json
- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_20261013_113432.md
- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv
- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_20261013_113432.csv

## Leitura operacional

- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.
- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.
- O CSV enriquecido complementar cobre 165 registros ate 2026-09-28 para analise independente de convergencia.

## Ranking das cidades - Top 30 (Geral)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOBRAL | 98.5 | crítico | 99.0% (alta) | 98.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 2 | CAUCAIA | 78.4 | crítico | 92.9% (alta) | 92.2% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-09-28 |
| 3 | JUAZEIRO DO NORTE | 67.3 | alto | 93.9% (alta) | 88.2% (muito alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 4 | TIANGUA | 47.0 | moderado | 88.7% (alta) | 82.3% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 5 | RUSSAS | 31.6 | moderado | 89.5% (alta) | 77.5% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 6 | BOA VIAGEM | 30.3 | baixo | 74.4% (moderada) | 75.6% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 7 | IGUATU | 27.5 | baixo | 83.8% (moderada) | 73.4% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 8 | SAO BENEDITO | 27.4 | baixo | 73.7% (moderada) | 71.8% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 9 | AQUIRAZ | 17.6 | baixo | 66.0% (baixa) | 68.1% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 10 | TRAIRI | 16.3 | baixo | 65.8% (baixa) | 66.3% (moderada) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 11 | ITAPAJE | 14.3 | baixo | 81.4% (moderada) | 64.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 12 | CRATO | 12.3 | baixo | 53.7% (muito baixa) | 62.3% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 13 | ITAPIPOCA | 11.2 | baixo | 53.6% (muito baixa) | 60.5% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 14 | QUIXADA | 9.6 | baixo | 53.5% (muito baixa) | 58.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 15 | MORADA NOVA | 5.6 | baixo | 37.5% (muito baixa) | 56.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 16 | MARACANAU | 4.9 | baixo | 37.5% (muito baixa) | 54.4% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 17 | AMONTADA | 4.5 | baixo | 37.5% (muito baixa) | 52.7% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 18 | LIMOEIRO DO NORTE | 4.5 | baixo | 37.5% (muito baixa) | 51.2% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 19 | ITAREMA | 3.2 | baixo | 37.3% (muito baixa) | 49.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 20 | FORQUILHA | 3.1 | baixo | 37.3% (muito baixa) | 47.7% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 21 | ARACATI | 3.0 | baixo | 37.3% (muito baixa) | 46.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 22 | CASCAVEL | 3.0 | baixo | 35.8% (muito baixa) | 44.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 23 | BEBERIBE | 2.8 | baixo | 35.7% (muito baixa) | 42.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 24 | PENTECOSTE | 2.6 | baixo | 37.3% (muito baixa) | 41.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 25 | PARACURU | 2.3 | baixo | 35.6% (muito baixa) | 39.6% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 26 | PINDORETAMA | 2.2 | baixo | 33.1% (muito baixa) | 38.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 27 | EUSEBIO | 2.1 | baixo | 35.5% (muito baixa) | 36.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 28 | CARIRE | 1.9 | baixo | 37.0% (muito baixa) | 34.8% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 29 | PACATUBA | 1.6 | baixo | 35.4% (muito baixa) | 33.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 30 | PARAIPABA | 1.5 | baixo | 35.4% (muito baixa) | 31.6% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |

## Ranking das cidades - Top 20 (RMF)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAUCAIA | 78.4 | crítico | 92.9% (alta) | 99.6% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; CVLI recente na janela de 30 dias | 2026-09-28 |
| 2 | AQUIRAZ | 17.6 | baixo | 66.0% (baixa) | 79.5% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 3 | TRAIRI | 16.3 | baixo | 65.8% (baixa) | 75.8% (alta) | Atividade recente e vizinhança; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 4 | MARACANAU | 4.9 | baixo | 37.5% (muito baixa) | 69.3% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 5 | CASCAVEL | 3.0 | baixo | 35.8% (muito baixa) | 65.5% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 6 | BEBERIBE | 2.8 | baixo | 35.7% (muito baixa) | 62.1% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 7 | PARACURU | 2.3 | baixo | 35.6% (muito baixa) | 58.7% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 8 | PINDORETAMA | 2.2 | baixo | 33.1% (muito baixa) | 55.3% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 9 | EUSEBIO | 2.1 | baixo | 35.5% (muito baixa) | 52.0% (moderada) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 10 | PACATUBA | 1.6 | baixo | 35.4% (muito baixa) | 48.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 11 | PARAIPABA | 1.5 | baixo | 35.4% (muito baixa) | 45.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 12 | SAO LUIS DO CURU | 1.4 | baixo | 34.6% (muito baixa) | 41.7% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 13 | CHOROZINHO | 0.8 | baixo | 35.2% (muito baixa) | 38.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 14 | SAO GONCALO DO AMARANTE | 0.7 | baixo | 35.2% (muito baixa) | 34.9% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 15 | MARANGUAPE | 0.5 | baixo | 35.1% (muito baixa) | 31.5% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 16 | GUAIUBA | 0.4 | baixo | 35.1% (muito baixa) | 28.2% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 17 | PACAJUS | 0.2 | baixo | 47.6% (muito baixa) | 24.7% (baixa) | CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional; Tensão territorial | 2026-09-28 |
| 18 | ITAITINGA | 0.2 | baixo | 35.0% (muito baixa) | 21.4% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 19 | HORIZONTE | 0.0 | baixo | 35.0% (muito baixa) | 18.0% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |

## Ranking das cidades - Top 30 (Interior)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOBRAL | 98.5 | crítico | 99.0% (alta) | 95.1% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 2 | JUAZEIRO DO NORTE | 67.3 | alto | 93.9% (alta) | 85.9% (muito alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 3 | TIANGUA | 47.0 | moderado | 88.7% (alta) | 78.9% (alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 4 | RUSSAS | 31.6 | moderado | 89.5% (alta) | 72.7% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 5 | BOA VIAGEM | 30.3 | baixo | 74.4% (moderada) | 69.3% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 6 | IGUATU | 27.5 | baixo | 83.8% (moderada) | 65.6% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 7 | SAO BENEDITO | 27.4 | baixo | 73.7% (moderada) | 62.4% (moderada) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 8 | ITAPAJE | 14.3 | baixo | 81.4% (moderada) | 56.8% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 9 | CRATO | 12.3 | baixo | 53.7% (muito baixa) | 53.2% (moderada) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 10 | ITAPIPOCA | 11.2 | baixo | 53.6% (muito baixa) | 49.8% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 11 | QUIXADA | 9.6 | baixo | 53.5% (muito baixa) | 46.4% (baixa) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 12 | MORADA NOVA | 5.6 | baixo | 37.5% (muito baixa) | 42.4% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 13 | AMONTADA | 4.5 | baixo | 37.5% (muito baixa) | 39.0% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 14 | LIMOEIRO DO NORTE | 4.5 | baixo | 37.5% (muito baixa) | 35.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 15 | ITAREMA | 3.2 | baixo | 37.3% (muito baixa) | 32.5% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 16 | FORQUILHA | 3.1 | baixo | 37.3% (muito baixa) | 29.3% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 17 | ARACATI | 3.0 | baixo | 37.3% (muito baixa) | 26.1% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 18 | PENTECOSTE | 2.6 | baixo | 37.3% (muito baixa) | 22.9% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 19 | CARIRE | 1.9 | baixo | 37.0% (muito baixa) | 19.6% (baixa) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 20 | CANINDE | 0.2 | baixo | 35.0% (muito baixa) | 16.1% (baixa) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |

## Ranking dos bairros - Top 30 (Fortaleza)

| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JANGURUSSU | 79.9 | crítico | 77.9% (moderada) | 100.0% (muito alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 2 | MESSEJANA | 49.8 | moderado | 87.3% (alta) | 99.5% (muito alta) | Tensão territorial; CVLI recente na janela de 30 dias; Atividade recente e vizinhança | 2026-09-28 |
| 3 | LAGOA REDONDA | 38.0 | moderado | 86.6% (alta) | 94.8% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 4 | PEDRAS | 35.6 | moderado | 76.0% (moderada) | 93.2% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 5 | CONJUNTO PALMEIRAS | 24.4 | baixo | 73.1% (moderada) | 87.6% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 6 | ANTONIO BEZERRA | 23.5 | baixo | 78.0% (moderada) | 86.6% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 7 | CURIO | 23.1 | baixo | 77.9% (moderada) | 86.0% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 8 | SAO BENTO | 22.9 | baixo | 72.8% (moderada) | 85.4% (muito alta) | Tensão territorial; Atividade recente e vizinhança; CVLI recente na janela de 30 dias | 2026-09-28 |
| 9 | BARRA DO CEARA | 21.1 | baixo | 40.8% (muito baixa) | 84.0% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 10 | MONDUBIM | 10.5 | baixo | 53.6% (muito baixa) | 78.6% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 11 | GRANJA LISBOA | 7.1 | baixo | 37.5% (muito baixa) | 76.6% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 12 | PASSARE | 6.7 | baixo | 53.2% (muito baixa) | 75.9% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 13 | PLANALTO AYRTON SENNA | 6.2 | baixo | 37.6% (muito baixa) | 75.1% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 14 | FATIMA | 5.2 | baixo | 53.0% (muito baixa) | 74.2% (alta) | Tensão territorial; CVLI recente na janela de 30 dias; Sinal Poisson do ranking operacional | 2026-09-28 |
| 15 | PREFEITO JOSE WALTER | 4.5 | baixo | 36.2% (muito baixa) | 73.4% (alta) | Sinal Poisson do ranking operacional; Tensão territorial; Atividade recente e vizinhança | 2026-09-28 |
| 16 | SIQUEIRA | 4.1 | baixo | 37.4% (muito baixa) | 72.7% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 17 | QUINTINO CUNHA | 3.7 | baixo | 37.4% (muito baixa) | 72.0% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 18 | BARROSO | 3.6 | baixo | 37.4% (muito baixa) | 71.5% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 19 | VICENTE PINZON | 3.2 | baixo | 37.4% (muito baixa) | 70.8% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 20 | CRISTO REDENTOR | 3.2 | baixo | 37.3% (muito baixa) | 70.3% (alta) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 21 | CARLITO PAMPLONA | 2.9 | baixo | 37.3% (muito baixa) | 69.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 22 | GRANJA PORTUGAL | 2.9 | baixo | 37.3% (muito baixa) | 69.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 23 | PICI | 2.8 | baixo | 37.3% (muito baixa) | 68.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 24 | JOSE DE ALENCAR | 2.8 | baixo | 37.3% (muito baixa) | 68.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 25 | MANOEL SATIRO | 2.8 | baixo | 37.3% (muito baixa) | 67.6% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 26 | JARDIM DAS OLIVEIRAS | 2.7 | baixo | 37.3% (muito baixa) | 67.1% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 27 | BOM JARDIM | 2.7 | baixo | 37.3% (muito baixa) | 66.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 28 | CANINDEZINHO | 2.6 | baixo | 37.3% (muito baixa) | 66.0% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 29 | EDSON QUEIROZ | 2.6 | baixo | 37.3% (muito baixa) | 65.5% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
| 30 | VILA VELHA | 2.5 | baixo | 37.3% (muito baixa) | 64.9% (moderada) | Tensão territorial; Sinal Poisson do ranking operacional; Atividade recente e vizinhança | 2026-09-28 |
