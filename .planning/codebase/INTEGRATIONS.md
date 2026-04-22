# External Integrations

**Analysis Date:** 2026-04-19

## APIs & External Services

**Generative AI:**
- Google Gemini (via `google-generativeai`) - Utilizado para geração de explicações gerenciais e análise qualitativa de descrições de crimes.

**Geocoding:**
- Nominatim (OpenStreetMap) via `geopy` - Utilizado em `/api/geocode` para localizar ruas e bairros no Ceará.

**Weather:**
- OpenWeatherMap (implícito pelo cache de clima) - Utilizado para enriquecer modelos com precipitação diária.

## Data Storage

**Databases:**
- **Local Filesystem:** Uso intensivo de arquivos `.pkl` (Pandas/Pickle), `.json`, `.csv` e `.geojson`.
- **Cache:** Sistema de cache baseado em arquivos JSON em `cache/` e `weather_cache/`.

**File Storage:**
- Armazenamento local em disco. Backups automáticos de modelos em `models/archive/`.

**Caching:**
- `manager_explanations_cache.json` - Cache de explicações geradas por IA.
- `geo_streets_cache.json` - Cache de ruas críticas geocodificadas.

## Authentication & Identity

**Auth Provider:**
- Custom - Não há provedor externo visível; provável controle de acesso via rede ou headers customizados.

## Monitoring & Observability

**Error Tracking:**
- Local (`server_err.txt`) e através do blueprint `Admin Health` (`src/core/admin_health_routes.py`).

**Logs:**
- `logs/cc_decisions.jsonl` - Decisões do Champion/Challenger.
- `logs/rankings/` - Snapshots diários de performance.

## CI/CD & Deployment

**Hosting:**
- Docker - `Dockerfile` e `docker-compose.yml` configurados.

**CI Pipeline:**
- Not detected.

## Environment Configuration

**Required env vars:**
- `GOOGLE_API_KEY` - Para o Gemini.
- `PORT` - Porta do servidor Flask.

**Secrets location:**
- Arquivo `.env` na raiz do projeto (não versionado).

## Webhooks & Callbacks

**Incoming:**
- `/api/export_static_snapshot` - Trigger manual para exportação de dados estáticos para repositório externo.

**Outgoing:**
- Git push automático no repositório `screenshot-report_preview` durante exportação de snapshot.

---

*Integration audit: 2026-04-19*
