# Benchmark: Poisson Retreinado vs LGBMRanker — FORTALEZA

- Gerado em: 2026-06-16 16:04:06
- Treino: 2022-01-01 → 2025-12-31
- Validação: 2026-01-01 → 2026-05-31
- Horizonte: 14 dias

## Resultados

| Modelo                    |  hit@1 |    p@5 |   p@10 |   p@20 |   r@10 |    mrr |  dias |
|---------------------------|--------|--------|--------|--------|--------|--------|-------|
| poisson_retreinado (2025) |  0.355 |  0.270 |  0.248 |  0.208 |  0.319 |  0.564 |   138 |
| lgbm_ranker (lambdamart)  |  0.196 |  0.190 |  0.193 |  0.155 |  0.261 |  0.395 |   138 |

**Vencedor (p@10):** `poisson_retreinado (2025)` com p@10 = 0.248

## Interpretação

- **p@10**: dos 10 bairros mais alertados, quantos tiveram CVLI real (primária de promoção)
- **hit@1**: o bairro #1 acertou? (operacionalmente crítico)
- **mrr**: posição média do primeiro acerto no ranking

## Delta LGBMRanker vs Poisson Retreinado

| Métrica | Delta |
|---------|-------|
| p@10    | -0.054 (-5.4pp) |
| mrr     | -0.169 |