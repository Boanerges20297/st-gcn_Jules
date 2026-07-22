# Validação walk-forward ST-GAT v5

- Gerado em: 2026-07-20 09:26:35
- Corte: 2026-03-01 a 2026-07-03
- Horizonte futuro: 14 dias
- Passo: 7 dia(s)
- Observação: avalia o checkpoint ativo sem retreino; se o checkpoint foi treinado com dados posteriores ao corte, isto mede inferência temporal, não prova prospectiva estrita.

| model | scope | windows | active_locations_avg | total_cvli | p5 | r5 | p10 | r10 | p20 | r20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EWMA_baseline | fortaleza | 18 | 4.7778 | 104 | 0.2 | 0.2259 | 0.1778 | 0.3681 | 0.1556 | 0.5954 |
| EWMA_baseline | global | 18 | 19.8889 | 527 | 0.6222 | 0.1608 | 0.5833 | 0.3011 | 0.4694 | 0.4833 |
| EWMA_baseline | interior | 18 | 10.2222 | 287 | 0.8 | 0.3985 | 0.6778 | 0.6701 | 0.5111 | 1.0 |
| EWMA_baseline | rmf | 18 | 4.8889 | 136 | 0.3778 | 0.381 | 0.3167 | 0.6713 | 0.2573 | 1.0 |
| ST-GAT_v5 | fortaleza | 18 | 4.7778 | 104 | 0.1778 | 0.1755 | 0.1556 | 0.3171 | 0.1417 | 0.5551 |
| ST-GAT_v5 | global | 18 | 19.8889 | 527 | 0.7 | 0.1814 | 0.5889 | 0.3036 | 0.4833 | 0.4939 |
| ST-GAT_v5 | interior | 18 | 10.2222 | 287 | 0.8 | 0.3976 | 0.6889 | 0.682 | 0.5111 | 1.0 |
| ST-GAT_v5 | rmf | 18 | 4.8889 | 136 | 0.3778 | 0.3897 | 0.3389 | 0.7077 | 0.2573 | 1.0 |
