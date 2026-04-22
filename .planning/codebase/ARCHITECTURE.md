# Architecture

**Analysis Date:** 2026-04-19

## Pattern Overview

**Overall:** Paradigma Híbrido Champion/Challenger (Modelo de Grafo + Gradiente Boosting)

**Key Characteristics:**
- **Champion (ST-GAT):** Rede neural `Spatial-Temporal Graph Attention Network` para predição de longo prazo (120 dias) e 37 canais.
- **Challenger (Sentinela V3):** Modelo `LightGBM Ranker` (LGBM Lean) otimizado para ranking tático e reação rápida.
- **Blend Dinâmico:** Ajuste de peso via EMA (Exponential Moving Average) baseado em P@10 contra dados reais em tempo de execução.

## Layers

**Core Logic (Model Orchestration):**
- Purpose: Coordena a execução dos modelos regionais e a fusão de scores.
- Location: `src/core/`
- Contains: `orchestrator.py`, `champion_challenger.py`, `architectures.py`
- Depends on: `torch`, `pandas`, `lightgbm`
- Used by: `app.py`

**API / Integration:**
- Purpose: Expõe os resultados para o dashboard e gerencia shocks exógenos.
- Location: `app.py`
- Contains: Endpoints Flask, Lógica de Shocks exógenos.
- Depends on: `src/core/`

**Data Enrichment:**
- Purpose: Processamento de features, enriquecimento com dados de inteligência (faccoes, ruas críticas).
- Location: `src/core/data_processing.py`, `scripts/`
- Contains: Geradores de cache geoespacial e importação de inteligência.

**Validation & Promotion:**
- Purpose: Treino, validação sombra e promoção de modelos candidatos.
- Location: `tests/Sentinela/`
- Contains: `freeze_total_v3.py`, `promote_model.py`, `train_validate_v3.py`

## Data Flow

**Inference Flow:**

1. `app.py` recebe requisição em `/api/risk`.
2. Carrega eventos exógenos de `data/exogenous_events.json`.
3. Chama `orchestrator.get_combined_risk()` (ST-GAT Champion).
4. Aplica Shocks exógenos aos scores (Canais 23 e 25).
5. Chama `champion_challenger.apply()` (Sentinela Challenger).
6. O Blend ajusta o score final de Fortaleza baseado na performance recente.
7. Retorna JSON para o Frontend.

**State Management:**
- Shocks exógenos persistidos em `data/exogenous_events.json`.
- Pesos do blend persistidos em `data/cc_state.json`.
- Histórico de decisões em `logs/cc_decisions.jsonl`.

## Key Abstractions

**StateOrchestrator:**
- Purpose: Gerencia modelos regionais (Fortaleza, RMF, Interior).
- Examples: `src/core/orchestrator.py`

**ChampionChallenger:**
- Purpose: Arbitra entre ST-GAT e LGBM usando métricas de precisão.
- Examples: `src/core/champion_challenger.py`

**HealthMonitor:**
- Purpose: Monitora latência da API e saúde do sistema (CPU/Memória).
- Examples: `src/core/health_monitor.py`

## Entry Points

**Flask App:**
- Location: `app.py`
- Triggers: HTTP Requests (Dashboard)
- Responsibilities: Orquestração final, API REST, Servir templates.

**Sentinela Retrain:**
- Location: `tests/Sentinela/freeze_total_v3.py`
- Triggers: Execução manual via CLI
- Responsibilities: Retreino do modelo Challenger com dados mais recentes.

## Error Handling

**Strategy:** Fail-soft com fallbacks para o modelo Champion (ST-GAT).

**Patterns:**
- Try/Except em blocos de importação e inicialização de modelos.
- Fallback para scores base se o challenger falhar.
- Monitoramento de erros via `server_err.txt` e `health_monitor.py`.

## Cross-Cutting Concerns

**Logging:** Registro de decisões do CC em `logs/cc_decisions.jsonl` e rankings diários em `logs/rankings/`.
**Validation:** Avaliação contínua de P@10/P@20 via `efficiency_monitor.py`.
**Authentication:** Não detectado (API aberta ou via Firewall/Docker).

---

*Architecture analysis: 2026-04-19*
