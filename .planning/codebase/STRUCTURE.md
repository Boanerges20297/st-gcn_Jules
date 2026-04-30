<<<<<<< HEAD
# Codebase Structure

**Analysis Date:** 2026-04-19

## Directory Layout

```
[project-root]/
├── config/             # Configurações de sistema
├── data/               # Armazenamento de dados (brutos, processados, caches)
│   ├── processed/      # Dados de grafo para ST-GAT (.pkl)
│   ├── raw/            # CSVs e GeoJSONs de inteligência
│   └── archives/       # Eventos exógenos antigos
├── logs/               # Logs de execução e performance
│   ├── rankings/       # Relatórios Markdown diários
│   └── cc_decisions.jsonl # Histórico de blend Champion/Challenger
├── models/             # Modelos treinados
│   ├── active/         # Modelos em produção (Champion e Challenger)
│   └── archive/        # Backups de modelos anteriores
├── outputs/            # Saídas de predição (GeoJSONs para o mapa)
├── scripts/            # Scripts de utilidade e processamento em lote
├── src/                # Código fonte principal
│   └── core/           # Núcleo da inteligência e orquestração
├── templates/          # Templates HTML (Flask)
├── tests/              # Testes e scripts de laboratório
│   └── Sentinela/      # Pipeline do modelo Challenger
└── app.py              # Entry point do servidor Flask
```

## Directory Purposes

**src/core/:**
- Purpose: Lógica de inferência e orquestração de modelos.
- Contains: Implementações de Champion/Challenger, Orchestrator e Health Monitoring.
- Key files: `orchestrator.py`, `champion_challenger.py`, `health_monitor.py`

**data/raw/:**
- Purpose: Fonte de dados de inteligência territorial.
- Contains: CSVs de ocorrências, latlong de bairros, polígonos de facções.
- Key files: `dados_status_ocorrencias_gerais_ENRIQUECIDO.csv`, `inteligencia_faccoes.csv`, `micronodos_faccoes_2026.geojson`

**tests/Sentinela/:**
- Purpose: Ambiente de desenvolvimento e treinamento do modelo Challenger (LGBM).
- Contains: Scripts de treinamento, validação sombra e promoção para produção.
- Key files: `freeze_total_v3.py`, `promote_model.py`, `train_validate_v3.py`

**models/active/:**
- Purpose: Repositório de modelos ativos utilizados pelo servidor.
- Contains: Pesos PyTorch (`.pth`) e modelos LightGBM (`.pkl`).

## Key File Locations

**Entry Points:**
- `app.py`: Servidor principal e API REST.
- `src/core/orchestrator.py`: Ponto central da lógica ST-GAT.

**Configuration:**
- `.env`: Variáveis de ambiente (não versionado).
- `config/`: Configurações adicionais.

**Core Logic:**
- `src/core/champion_challenger.py`: Lógica de blend dinâmico.
- `src/core/architectures.py`: Definições das redes neurais.

**Testing & Validation:**
- `src/core/efficiency_monitor.py`: Monitoramento contínuo de P@10/P@20.
- `tests/Sentinela/train_validate_v3.py`: Validação de performance do Challenger.

## Naming Conventions

**Files:**
- Snake Case: `champion_challenger.py`, `data_processing.py`
- Prefixos em dados: `processed_*.pkl`, `inteligencia_*.csv`

**Directories:**
- Snake Case ou Lowercase: `src/core`, `tests/Sentinela`

## Where to Add New Code

**New Feature (Model):**
- Implementation: `src/core/`
- Training script: `scripts/training/` ou `tests/Sentinela/`

**New API Endpoint:**
- Implementation: `app.py` ou novo blueprint em `src/core/` (similar a `admin_health_routes.py`).

**Utilities:**
- Shared helpers: `scripts/` ou módulo específico em `src/` (ex: `src/enrichment.py`).

## Special Directories

**cache/:**
- Purpose: Caches de requisições ou processamentos temporários.
- Generated: Sim
- Committed: Não (em grande parte)

**weather_cache/:**
- Purpose: Dados climáticos baixados para enriquecimento.
- Generated: Sim
- Committed: Sim (archive em `data/weather_archive_cache.json`)

---

*Structure analysis: 2026-04-19*
=======
# Project Structure

## Root Directory
- `app.py`: Main application entry point (Flask).
- `GEMINI.md`: Current project status and rules.
- `TRAINING_LOG.md`: Detailed history of all training attempts.
- `README.md` / `GETTING_STARTED.md`: Documentation.

## Key Directories
- `src/core/`:
  - `architectures.py`: ST-GAT model definitions (DeepSTGAT_64, DeepSTGAT_80).
  - `orchestrator.py`: Logic for running inferences and managing models.
  - `champion_challenger.py`: Hybrid blend logic (EMA weight adjustment).
  - `training_vault.py`: MemPalace memory system.
- `scripts/training/Active/`:
  - `train_all_specialists.py`: Official training script for regional models.
- `tests/Sentinela/`:
  - `ROADMAP.md`: Sentinela development plan.
  - `train_validate_v3.py`: Validation script for LGBM.
  - `finetune_realtime_v1.py`: Real-time adjustment logic.
- `models/active/`:
  - Production-ready model files (`.pth` and `.pkl`).
- `data/`:
  - `processed/`: Serialized features for training.
  - `raw/`: Raw CSV data.
- `logs/`:
  - Training and system logs.
>>>>>>> f0c5e832fdb68777a137ddf4ac9d9c9f82fa9032
