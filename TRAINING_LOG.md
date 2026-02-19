# 📈 Histórico de Treinamento - ST-GAT (Report Preview)


## Tentativa 1 - 2026-02-19 14:33
- **Arquitetura:** DeepSTGAT_64/32
- **Loss:** Weighted MSE (log1p)
- **Resultado:** Platô em 19.5% (P@10)
- **Status:** Interrompido para ajustes estruturais.


## Tentativa 2 - 2026-02-19 15:06
- **Arquitetura:** DeepSTGAT_64/32
- **Loss:** Power-Weighted MSE (target^2 * 20)
- **Resultado:** Ineficaz (Platô mantido em ~20%)
- **Observação:** O modelo ainda foca excessivamente em minimizar o erro numérico em vez da ordem de prioridade.


## Tentativa 3 - 2026-02-19 15:30
- **Arquitetura:** DeepSTGAT_64/32
- **Loss:** Contrastive Ranking + Hard Negative Mining
- **Resultado:** Nada feito (Platô persistente)
- **Diagnóstico:** O modelo está sofrendo com a esparsidade dos dados; os dias 'comuns' estão 'lavando' o aprendizado dos dias de crise.


## Tentativa 4 - 2026-02-19 15:49 (Abortada)
- **Ajuste:** Oversampling + Top-K Loss
- **Status:** Pulado para upgrade arquitetural imediato.

## Tentativa 5 - 2026-02-19 15:49
- **Arquitetura:** Híbrida (Spatial Transformer + Relational GCN)
- **Loss:** Top-K Focal Loss
- **Estratégia:** Autoconsciência global. O modelo agora pode aprender correlações entre bairros distantes sem depender apenas das matrizes de adjacência.

