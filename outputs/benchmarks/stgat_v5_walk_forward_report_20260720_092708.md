# Validação walk-forward ST-GAT v5

- Gerado em: 2026-07-20 09:27:08
- Corte: 2026-03-01 a 2026-07-03
- Horizonte futuro: 14 dias
- Passo: 1 dia(s)
- Observação: avalia o checkpoint ativo sem retreino; se o checkpoint foi treinado com dados posteriores ao corte, isto mede inferência temporal, não prova prospectiva estrita.

| model | scope | windows | active_locations_avg | total_cvli | p5 | r5 | p10 | r10 | p20 | r20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EWMA_baseline | fortaleza | 125 | 4.792 | 717 | 0.216 | 0.2293 | 0.1816 | 0.3722 | 0.1584 | 0.6215 |
| EWMA_baseline | global | 125 | 19.704 | 3625 | 0.6624 | 0.1728 | 0.596 | 0.3103 | 0.4676 | 0.4852 |
| EWMA_baseline | interior | 125 | 10.096 | 1989 | 0.8224 | 0.4148 | 0.6784 | 0.6785 | 0.5048 | 1.0 |
| EWMA_baseline | rmf | 125 | 4.816 | 919 | 0.392 | 0.4117 | 0.3264 | 0.7039 | 0.2535 | 1.0 |
| ST-GAT_v5 | fortaleza | 125 | 4.792 | 717 | 0.1952 | 0.1996 | 0.1488 | 0.2985 | 0.1448 | 0.5768 |
| ST-GAT_v5 | global | 125 | 19.704 | 3625 | 0.6864 | 0.1795 | 0.6096 | 0.3179 | 0.4796 | 0.4971 |
| ST-GAT_v5 | interior | 125 | 10.096 | 1989 | 0.784 | 0.3928 | 0.68 | 0.6794 | 0.5048 | 1.0 |
| ST-GAT_v5 | rmf | 125 | 4.816 | 919 | 0.3872 | 0.4315 | 0.3264 | 0.6886 | 0.2535 | 1.0 |
