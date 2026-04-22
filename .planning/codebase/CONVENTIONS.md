# Coding Conventions

**Analysis Date:** 2026-04-19

## Naming Patterns

**Files:**
- Snake Case para módulos Python: `champion_challenger.py`, `orchestrator.py`.
- Lowercase/Kebab Case para arquivos de dados: `exogenous_events.json`, `lgbm_lean_v3_freeze.pkl`.

**Functions:**
- Snake Case: `get_combined_risk()`, `apply_shocks()`, `normalize_name()`.
- Prefixos internos: `_load_challenger()`, `_evaluate_and_update()`.

**Variables:**
- Snake Case: `scores_map`, `cc_weight`, `last_date_base`.
- Constantes em Uppercase: `EVAL_DAYS`, `MAX_CC_WEIGHT`, `PESO_NATUREZA`.

**Types:**
- Pascal Case para Classes: `StateOrchestrator`, `ChampionChallenger`, `HealthMonitor`.

## Code Style

**Formatting:**
- Indentação de 4 espaços (Padrão PEP 8).
- Uso extensivo de Docstrings no topo de classes e módulos para documentação de fase e arquitetura.

**Linting:**
- Não detectado arquivo de configuração (ESLint/Pylint), mas o código segue padrões consistentes de PEP 8.

## Import Organization

**Order:**
1. Built-in (os, json, time, sys)
2. Dependências externas (numpy, pandas, torch, flask)
3. Módulos internos (`from src.core...`)

**Path Aliases:**
- Não utilizado. Adição manual ao `sys.path` em `app.py` se necessário.

## Error Handling

**Patterns:**
- Try/Except com logs informativos via `print` ou `logging`.
- Fallbacks silenciosos para manter a estabilidade da API em produção.
- Monitoramento de erros via `health_monitor.py` que rastreia taxas de sucesso de requisições.

## Logging

**Framework:** `logging` (Python Standard Library) e `jsonl` customizado.

**Patterns:**
- Logs de console para eventos de startup.
- Registro estruturado em `cc_decisions.jsonl` para decisões de modelo.
- Relatórios em Markdown (`logs/rankings/*.md`) para auditoria humana.

## Comments

**When to Comment:**
- Cabeçalhos de módulos detalhando versão, fase do projeto e integração.
- Comentários em blocos de lógica complexa (cálculo de features, blend EMA).

**JSDoc/TSDoc:**
- Não aplicável (Uso de Docstrings Python).

## Function Design

**Size:** Variável. Algumas funções de processamento em `app.py` e `champion_challenger.py` são longas devido à manipulação densa de dataframes.

**Parameters:** Majoritariamente explícitos. Uso moderado de `**kwargs`.

**Return Values:** Geralmente dicionários (para APIs) ou arrays NumPy/DataFrames Pandas.

## Module Design

**Exports:** Classes e instâncias únicas (Singleton Pattern) inicializadas no startup do `app.py`.

**Barrel Files:** Não utilizado.

---

*Convention analysis: 2026-04-19*
