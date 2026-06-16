# Getting Started

Este onboarding foi reduzido para o fluxo real do projeto atual.

## Pre-requisitos

- Python com ambiente virtual configurado
- dependencias do projeto instaladas
- base de dados presente em `data/raw/`

## Subir a aplicacao

```powershell
.\.venv\Scripts\python.exe app.py
```

## Endpoints e checagens iniciais

- dashboard local: `http://127.0.0.1:5000/`
- risco: `http://127.0.0.1:5000/api/risk`
- log de validacao: `VALIDATION_LOG.md`

## Quando houver atualizacao de base

1. reprocessar:

```powershell
.\.venv\Scripts\python.exe src\core\data_processing.py
```

2. retreinar/promover:

```powershell
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

3. reiniciar:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Ler em seguida

- [CURRENT_ARCHITECTURE.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/CURRENT_ARCHITECTURE.md)
- [CURRENT_OPERATIONS.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/CURRENT_OPERATIONS.md)
- [CVLI_STOCHASTIC_BENCHMARK.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/CVLI_STOCHASTIC_BENCHMARK.md)
