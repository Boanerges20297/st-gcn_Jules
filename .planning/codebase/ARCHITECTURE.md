# Architecture Overview

## Hybrid Paradigm (Champion/Challenger)
The system operates as an ensemble of two distinct modeling approaches:

1. **Champion (ST-GAT):**
   - **Type:** Spatio-Temporal Graph Attention Network.
   - **Input:** 37-38 channels (Spatial, Temporal, Momentum, and Memory).
   - **Focus:** Capturing complex, non-linear dependencies between neighborhoods over 120-day windows.
   - **Implementation:** `DeepSTGAT_64` / `DeepSTGAT_80` in `src/core/architectures.py`.

2. **Challenger (Sentinela V3):**
   - **Type:** LightGBM Lean Ranker.
   - **Input:** 10 highly calibrated features (Top-10 Importance).
   - **Focus:** Stable, explainable ranking based on CVP/CVLI ratios and troop intelligence.
   - **Implementation:** `lgbm_lean_v3_freeze.pkl` managed via `src/core/champion_challenger.py`.

## Decision Engine
- **EMA Blend:** A dynamic weight `w_cc` adjusts every hour based on the last 14 days of real performance (P@10).
- **Fallback:** If Sentinela fails, the system defaults to ST-GAT scores.

## Memory System (MemPalace V4) - [ACTIVE / GATED]
- **Conceito:** Atenção residual baseada em falhas anteriores.
- **Mecanismo:** O `TrainingVault` captura surpresas (crimes reais fora do Top 20) de forma consolidada no fim de cada ciclo.
- **Disciplina Gated:** Durante o treinamento, o canal de memória sofre Dropout espacial de 50%, impedindo que o modelo vicie em atalhos e forçando o uso de features espaciais reais.
- **Status:** Implementado na Sentinela V4 para estabilizar P@10 > 50%.
