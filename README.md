# Report Preview

Sistema operacional de priorização territorial para CVLI no Ceará.

## Arquitetura atual

O runtime de produção atual usa:

- `Poisson Ranker Estadual`
- três regiões:
  - Fortaleza
  - RMF
  - Interior
- horizonte de `14 dias`

Os modelos ativos ficam em:

- [models/active/production/poisson](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/models/active/production/poisson)

## Onde começar

- arquitetura viva: [docs/CURRENT_ARCHITECTURE.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/docs/CURRENT_ARCHITECTURE.md)
- operação atual: [docs/CURRENT_OPERATIONS.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/docs/CURRENT_OPERATIONS.md)
- benchmark oficial: [docs/CVLI_STOCHASTIC_BENCHMARK.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/docs/CVLI_STOCHASTIC_BENCHMARK.md)
- mapa de docs: [docs/INDEX.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/docs/INDEX.md)
- mapa do core: [src/core/README.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/README.md)
- mapa de scripts: [scripts/README.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/README.md)
- modelos: [models/README.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/models/README.md)

## Entrada principal

- app: [app.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/app.py:1)
- orquestrador: [src/core/orchestrator.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/orchestrator.py:1)
- backend de modelo: [src/core/fortaleza_poisson_backend.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/fortaleza_poisson_backend.py:1)

## Fluxo operacional mínimo

Atualizar processados:

```powershell
.\.venv\Scripts\python.exe src\core\data_processing.py
```

Retreinar os três modelos ativos:

```powershell
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

Subir a aplicação:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Acompanhamento

- log de validação: [VALIDATION_LOG.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/VALIDATION_LOG.md)
- log de treino/experimentos: [TRAINING_LOG.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/TRAINING_LOG.md)
- snapshot Hermes: [outputs/hermes/risk_snapshot_latest.json](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/outputs/hermes/risk_snapshot_latest.json)

## Legado

Artefatos e documentação antigos foram preservados, mas não representam o fluxo atual:

- `models/active/legacy_torch/`
- `models/archive/`
- `scripts/training/Legacy/`
- `docs/archive/` e documentos marcados como históricos/legados
