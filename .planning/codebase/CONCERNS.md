<<<<<<< HEAD
# Codebase Concerns

**Analysis Date:** 2026-04-19

## Tech Debt

**Manual Model Promotion:**
- Issue: A promoção de modelos do Challenger (`Sentinela`) ainda exige execução manual de script e revisão humana.
- Files: `tests/Sentinela/promote_model.py`
- Impact: Risco de erro humano ou atraso na atualização de modelos em produção.
- Fix approach: Implementar pipeline automático de CD (Continuous Deployment) que promove o modelo se as métricas de validação sombra excederem o Champion por N dias.

**Large File (app.py):**
- Issue: O arquivo `app.py` ultrapassa 3.000 linhas, acumulando lógica de rotas, processamento de dados, orquestração e exportação.
- Files: `app.py`
- Impact: Dificuldade de manutenção e legibilidade.
- Fix approach: Refatorar rotas para Blueprints Flask (ex: `api_risk.py`, `api_simulation.py`) e mover lógica de negócios pesada para serviços em `src/services/`.

## Known Bugs

**Incompatibilidade NumPy/Pickle:**
- Symptoms: Erros ao carregar modelos se a versão do NumPy no ambiente de treino diferir do servidor (especialmente NumPy 1.x vs 2.x).
- Files: `src/core/orchestrator.py`, `src/core/champion_challenger.py`
- Trigger: Atualização de pacotes sem retreino sincronizado.
- Workaround: Injeção manual de canais de contexto e verificações de versão no carregamento.

## Security Considerations

**API Aberta:**
- Risk: Endpoints `/api/risk` e `/api/admin/health` parecem não possuir autenticação por token (JWT/API Key) no código.
- Files: `app.py`, `src/core/admin_health_routes.py`
- Current mitigation: Provável controle via infraestrutura (Firewall, VPN ou Proxy reverso).
- Recommendations: Implementar autenticação básica ou middleware de API Key.

## Performance Bottlenecks

**Feature Construction no Startup:**
- Problem: O cálculo de features para o Challenger no `ChampionChallenger.__init__` envolve ler CSVs grandes de ocorrências.
- Files: `src/core/champion_challenger.py`
- Cause: Processamento de features não é cacheado em disco entre reinícios.
- Improvement path: Persistir a matriz de features processada em `.pkl` e carregar apenas o diferencial diário.

## Fragile Areas

**Exogenous Shocks Logic:**
- Files: `app.py` (função `get_risk`)
- Why fragile: A lógica de mapeamento de eventos de texto para intensidades numéricas é baseada em heurísticas e palavras-chave.
- Safe modification: Testar exaustivamente novas palavras-chave para evitar inflação artificial de scores.
- Test coverage: Baixa.

## Scaling Limits

**Local Filesystem Dependency:**
- Current capacity: Centenas de MB de dados em disco.
- Limit: Performance de I/O em discos lentos pode afetar o tempo de resposta da API ao ler GeoJSONs e Modelos.
- Scaling path: Migrar para um banco de dados geoespacial (PostGIS) e Redis para cache.

## Dependencies at Risk

**LightGBM Versão:**
- Risk: O modelo `.pkl` é sensível à versão da biblioteca.
- Impact: Quebra da inferência se o ambiente de produção for atualizado sem o modelo.
- Migration plan: Usar `ONNX` para exportação de modelos, desacoplando o framework de treino da inferência.

## Missing Critical Features

**Automated Retraining Trigger:**
- Problem: Não há um gatilho automático (webhook ou cron) para o `freeze_total_v3.py`.
- Blocks: Atualização proativa do modelo quando novos dados de CVLI são ingeridos.

## Test Coverage Gaps

**Unit Tests para Orquestração:**
- What's not tested: Lógica de blend EMA e calculo de pesos.
- Files: `src/core/champion_challenger.py`
- Risk: Regressões em como os modelos são fundidos podem passar despercebidas.
- Priority: High

---

*Concerns audit: 2026-04-19*
=======
# Technical Concerns

## 1. Falha do Paradigma MemPalace (Canal 38)
- **Problema:** A "memória de aprendizado" entre épocas via `TrainingVault` (tentativa de ensinar o modelo a não repetir erros) não se mostrou eficiente para atingir a meta de 70% P@20.
- **Causa Técnica:** O feedback de surpresas cria um viés espacial estático que compete com os sinais temporais dinâmicos do GCN, gerando instabilidade no gradiente em vez de refinamento.
- **Impacto:** O modelo estagna em ~53% P@20 em Fortaleza, com degradação após poucas épocas.

## 2. Hybrid System Calibration
- **Issue:** The blend between Champion (ST-GAT) and Challenger (Sentinela) relies on real-time evaluation.
- **Concern:** If both models underperform in a specific period, the EMA might fluctuate significantly.

## 3. Data Leakage Risks
- **Precedent:** Past attempts (T48) identified data leakage from `random.shuffle`. 
- **Check:** Ensure the Temporal Split (85/15) in `train_all_specialists.py` is strictly enforced and that no future information leaks into the window normalization.

## 4. Hardware Constraints
- **Observation:** Training is running on CPU (`device=cpu` in logs). 
- **Impact:** Slow iteration cycles (18h for full training) hinder rapid experimentation with new strategies.
>>>>>>> f0c5e832fdb68777a137ddf4ac9d9c9f82fa9032
