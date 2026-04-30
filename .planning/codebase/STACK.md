<<<<<<< HEAD
# Technology Stack

**Analysis Date:** 2026-04-19

## Languages

**Primary:**
- Python 3.x - Backend (Flask), Data Science (PyTorch, LightGBM, Pandas).

**Secondary:**
- HTML/JavaScript - Dashboard Frontend (Bootstrap, Leaflet/Mapbox).
- Shell (PowerShell/Bash) - Scripts de automação e deploy.

## Runtime

**Environment:**
- Python 3.10+ (Inferido pelas dependências)

**Package Manager:**
- pip - Gerenciador de pacotes.
- Lockfile: `requirements.txt` presente.

## Frameworks

**Core:**
- Flask 3.0.0 - Web Server / API REST.
- PyTorch 2.1.2 - Motor Champion (ST-GAT).
- LightGBM - Motor Challenger (Sentinela V3).

**Testing:**
- Pytest (implícito pelo `.pytest_cache`).
- Scripts customizados em `tests/Sentinela/` para validação de ML.

**Build/Dev:**
- Flask-CORS 4.0.0 - Suporte a requisições Cross-Origin.
- Python-dotenv 1.0.0 - Gestão de variáveis de ambiente.

## Key Dependencies

**Critical:**
- `pandas` & `numpy` - Manipulação massiva de dados e features.
- `geopandas` & `shapely` - Processamento geoespacial de micronodos e ruas.
- `google-generativeai` - Provável uso em explicações gerenciais ou análise de descrição de eventos.

**Infrastructure:**
- `psutil` - Monitoramento de recursos do sistema em `health_monitor.py`.
- `requests` - Integração com APIs externas (ex: Clima).

## Configuration

**Environment:**
- Arquivo `.env` (baseado em `.env.example`).
- Variáveis para caminhos de dados e chaves de API (Gemini).

**Build:**
- `Dockerfile` e `docker-compose.yml` para containerização.

## Platform Requirements

**Development:**
- Windows/Linux com Python 3.10+.
- Acesso a arquivos de dados locais em `data/`.

**Production:**
- Servidor com suporte a Docker ou ambiente Python isolado (venv).
- GPU opcional (PyTorch configurado para CPU/CUDA).

---

*Stack analysis: 2026-04-19*
=======
# Tech Stack

## Core Technologies
- **Language:** Python 3.x
- **Deep Learning Framework:** PyTorch
- **Machine Learning Framework:** LightGBM (Sentinela V3)
- **Web Framework:** Flask (app.py)
- **Database:** CSV/Pickle files (data/processed, data/raw)

## Key Libraries
- `torch`: ST-GAT implementation
- `lightgbm`: Challenger model
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `geopandas`: Spatial data handling
- `pickle`: Model and feature serialization

## Infrastructure
- **Model Storage:** `models/active/`
- **Logging:** `logs/`
- **Orchestration:** `src/core/orchestrator.py`
- **Hybrid System:** `src/core/champion_challenger.py`
>>>>>>> f0c5e832fdb68777a137ddf4ac9d9c9f82fa9032
