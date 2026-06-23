# Benchmark: Poisson Retreinado vs LGBMRanker — RMF

- Gerado em: 2026-06-16 16:05:49
- Treino: 2022-01-01 → 2025-12-31
- Validação: 2026-01-01 → 2026-05-31
- Horizonte: 14 dias

## Resultados

| Modelo                    |  hit@1 |    p@5 |   p@10 |   p@20 |   r@10 |    mrr |  dias |
|---------------------------|--------|--------|--------|--------|--------|--------|-------|
| poisson_retreinado (2025) |  0.746 |  0.342 |  0.324 |  0.275 |  0.625 |  0.848 |   138 |
| lgbm_ranker (lambdamart)  |  0.638 |  0.326 |  0.285 |  0.275 |  0.568 |  0.793 |   138 |

**Vencedor (p@10):** `poisson_retreinado (2025)` com p@10 = 0.324

## Interpretação

- **p@10**: dos 10 bairros mais alertados, quantos tiveram CVLI real (primária de promoção)
- **hit@1**: o bairro #1 acertou? (operacionalmente crítico)
- **mrr**: posição média do primeiro acerto no ranking

## Delta LGBMRanker vs Poisson Retreinado

| Métrica | Delta |
|---------|-------|
| p@10    | -0.039 (-3.9pp) |
| mrr     | -0.055 |