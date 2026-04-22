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
