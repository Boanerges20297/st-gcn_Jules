# CVLI Stochastic Benchmark Suite

- Gerado em: 2026-06-12 22:04:43
- Device: cpu
- Cutoff bruto: 2025-12-31
- Treino: 2022-01-01 -> 2024-12-31
- Validacao: 2025-01-01 -> 2025-12-31
- Horizonte: 14 dias
- Deep epochs: 1
- Regioes: fortaleza

## Sinal Estocastico no ENRIQUECIDO

| rows_cvli | locations | date_min   | date_max   | daily_mean        | daily_var          | daily_fano         | daily_zero_rate       | daily_acf_1         | daily_acf_7         | positive_days_share_eq_1 | positive_days_share_gt_1 | positive_days_p95 | positive_days_max |
|-----------|-----------|------------|------------|-------------------|--------------------|--------------------|-----------------------|---------------------|---------------------|--------------------------|--------------------------|-------------------|-------------------|
| 11704     | 296       | 2022-01-01 | 2025-12-31 | 8.017796030116358 | 10.443518700833543 | 1.3025423272936494 | 0.0006844626967830253 | 0.08881421659683299 | 0.12153915830830957 | 0.0027397260273972603    | 0.9972602739726028       | 14.0              | 20.0              |

## Top-3 por Regiao

| model_type | days_scored | positive_days | hit1_event | p5_event | p10_event | p20_event | recall10_event      | recall20_event      | overlap10_rank | mrr_event          | model            | region    | train_samples | val_samples |
|------------|-------------|---------------|------------|----------|-----------|-----------|---------------------|---------------------|----------------|--------------------|------------------|-----------|---------------|-------------|
| deep       | 2           | 2             | 0.5        | 0.2      | 0.1       | 0.125     | 0.04772727272727273 | 0.11818181818181818 | 0.1            | 0.75               | ShallowGAT       | fortaleza | 4             | 2           |
| classic    | 2           | 2             | 0.5        | 0.2      | 0.1       | 0.175     | 0.05                | 0.1659090909090909  | 0.05           | 0.5384615384615384 | zero_baseline    | fortaleza | 4             | 2           |
| classic    | 2           | 2             | 0.0        | 0.5      | 0.5       | 0.45      | 0.23409090909090907 | 0.425               | 0.25           | 0.375              | logit_classifier | fortaleza | 4             | 2           |

## Benchmark Completo

| model_type | days_scored | positive_days | hit1_event | p5_event | p10_event | p20_event | recall10_event      | recall20_event      | overlap10_rank | mrr_event          | model            | region    | train_samples | val_samples |
|------------|-------------|---------------|------------|----------|-----------|-----------|---------------------|---------------------|----------------|--------------------|------------------|-----------|---------------|-------------|
| deep       | 2           | 2             | 0.5        | 0.2      | 0.1       | 0.125     | 0.04772727272727273 | 0.11818181818181818 | 0.1            | 0.75               | ShallowGAT       | fortaleza | 4             | 2           |
| classic    | 2           | 2             | 0.5        | 0.2      | 0.1       | 0.175     | 0.05                | 0.1659090909090909  | 0.05           | 0.5384615384615384 | zero_baseline    | fortaleza | 4             | 2           |
| classic    | 2           | 2             | 0.0        | 0.5      | 0.5       | 0.45      | 0.23409090909090907 | 0.425               | 0.25           | 0.375              | logit_classifier | fortaleza | 4             | 2           |
