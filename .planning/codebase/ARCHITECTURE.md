# Architecture

**Analysis Date:** 2026-05-12

## Pattern Overview

**Overall:** Hybrid champion/challenger stack with operational guardrails on the final risk score.

**Key Characteristics:**
- **Champion (ST-GAT):** Primary risk engine for Fortaleza, RMF, and Interior using spatial-temporal graph models over 120-day windows.
- **Challenger (Sentinela V3):** LightGBM/EWMA tactical ranker loaded behind a safety gate; it only blends into the API response when recent measured performance justifies a non-zero `cc_weight`.
- **Operational Guardrails:** `CVP` remains contextual only, and faction/tension pressure must be backed by real recent `CVLI` or meaningful historical `CVLI` support before affecting final risk.

## Layers

**Core Logic (Model Orchestration):**
- Purpose: Coordinate regional models, compute final risk, and enforce scoring guardrails.
- Location: `src/core/`
- Contains: `orchestrator.py`, `champion_challenger.py`, `architectures.py`
- Depends on: `torch`, `pandas`, `numpy`, `lightgbm`
- Used by: `app.py`

**API / Integration:**
- Purpose: Expose risk results to the dashboard, manage exogenous shocks, and trigger static exports.
- Location: `app.py`
- Contains: Flask endpoints, startup loading, static snapshot hooks.
- Depends on: `src/core/`

**Data Enrichment:**
- Purpose: Build processed node features, enrich nodes with faction intelligence, and maintain geospatial/tactical caches.
- Location: `src/core/data_processing.py`, `scripts/`
- Contains: processed pickles, faction joins, route/street caches, static snapshot generation.

**Validation & Promotion:**
- Purpose: Train, shadow-validate, and promote challenger candidates without directly changing production scores.
- Location: `tests/Sentinela/`
- Contains: `freeze_total_v3.py`, `promote_model.py`, `train_validate_v3.py`, `sentinela_inference.py`

## Data Flow

**Inference Flow:**

1. `app.py` receives a request on `/api/risk`.
2. Exogenous events are loaded from `data/exogenous_events.json`.
3. `StateOrchestrator.get_combined_risk()` computes the champion score.
4. Territorial tension is attenuated unless there is recent tensor-level `CVLI` or meaningful historical `CVLI` support.
5. `ChampionChallenger.apply()` optionally blends challenger scores for Fortaleza only when `cc_weight > 0`.
6. `CVP`-derived challenger features are neutralized in the external scoring path.
7. The API returns the final JSON payload to the frontend or to static snapshot exporters.

**State Management:**
- Exogenous shocks persist in `data/exogenous_events.json`.
- Champion/challenger blend state persists in `data/cc_state.json`.
- Decision history is recorded in `logs/cc_decisions.jsonl`.
- Generated static artifacts are written under `static_export/data/`.

## Key Abstractions

**StateOrchestrator:**
- Purpose: Manage the three regional ST-GAT models and compute guarded final risk scores.
- Example: `src/core/orchestrator.py`

**ChampionChallenger:**
- Purpose: Evaluate challenger value, restore persisted blend state, and safely blend challenger output when allowed.
- Example: `src/core/champion_challenger.py`

**HealthMonitor:**
- Purpose: Track API/system health and surface operational issues.
- Example: `src/core/health_monitor.py`

## Entry Points

**Flask App:**
- Location: `app.py`
- Triggers: HTTP requests and startup initialization
- Responsibilities: load models, expose dashboard/API, run export hooks, orchestrate final responses.

**Sentinela Retrain / Validation:**
- Location: `tests/Sentinela/freeze_total_v3.py`, `tests/Sentinela/train_validate_v3.py`
- Triggers: manual CLI execution
- Responsibilities: retrain and validate challenger candidates offline.

## Error Handling

**Strategy:** Fail-soft with fallback to champion-only output.

**Patterns:**
- Try/except around model initialization and challenger loading.
- `cc_weight` can remain `0%`, keeping the API on pure ST-GAT output.
- Static exports call the same `/api/risk` path, reducing divergence between dashboard and export output.
- Operational logs and diagnostics are written to `server_err.txt`, `server_log.txt`, and health endpoints.

## Cross-Cutting Concerns

**Logging:** Champion/challenger decisions in `logs/cc_decisions.jsonl`; daily rankings in `logs/rankings/`.
**Validation:** Continuous P@10/P@20 evaluation via `efficiency_monitor.py` and Sentinela validation scripts.
**Operational Safety:** Context signals must not inflate cold neighborhoods; recent/live `CVLI` remains the primary escalation signal.

---

*Architecture analysis: 2026-05-12*
