# Guia de Integracao de Ranking

O projeto hoje usa ranking operacional via artefatos Poisson promovidos.

## Integracao atual

Fluxo oficial:

1. `app.py`
2. `src/core/orchestrator.py`
3. `src/core/fortaleza_poisson_backend.py`
4. `models/active/production/poisson/*.pkl`

## Regioes cobertas

- `fortaleza`
- `rmf`
- `interior`

## O que ficou legado

Este arquivo nao descreve mais o antigo fluxo de:

- ST-GCN produzindo score inicial
- ranking corretivo ajustando top-k
- blend de validacao neural em tempo real como arquitetura oficial

Esse material antigo deve ser tratado apenas como historico tecnico.

## Referencias corretas

- [CURRENT_ARCHITECTURE.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/CURRENT_ARCHITECTURE.md)
- [CVLI_STOCHASTIC_BENCHMARK.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/CVLI_STOCHASTIC_BENCHMARK.md)
- [MODEL_UPDATE_SYSTEM.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/MODEL_UPDATE_SYSTEM.md)
