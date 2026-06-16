# Resposta a Incidentes

Guia rapido para incidentes na arquitetura atual.

## Escopo

Este documento cobre a operacao do backend oficial:

- `Poisson Ranker Estadual`

## Incidentes mais provaveis

### App nao sobe

Verificar:

1. ambiente virtual e dependencias
2. presenca dos processados em `data/processed/`
3. presenca dos artefatos em `models/active/production/poisson/`
4. erros no console ao rodar `app.py`

Comando base:

```powershell
.\.venv\Scripts\python.exe app.py
```

### `/api/risk` falha

Validar:

1. `src/core/orchestrator.py` carregou as tres regioes
2. `VALIDATION_LOG.md` foi atualizado
3. os arquivos `.pkl` ativos existem
4. os dados processados estao coerentes com a base

### Modelo aparentemente degradou

Conferir:

1. `VALIDATION_LOG.md`
2. `outputs/benchmarks/`
3. benchmark oficial em `docs/CVLI_STOCHASTIC_BENCHMARK.md`

Se a piora persistir por `2 ou 3 ciclos`, antecipar retreino.

## Recuperacao operacional

### Reprocessar dados

```powershell
.\.venv\Scripts\python.exe src\core\data_processing.py
```

### Repromover modelos oficiais

```powershell
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

### Reiniciar aplicacao

```powershell
.\.venv\Scripts\python.exe app.py
```

## Evidencias minimas apos recuperacao

- `/api/risk` retorna `200`
- `model_architecture` volta como `Poisson Ranker Estadual`
- `VALIDATION_LOG.md` recebe nova entrada
- Hermes volta a exportar snapshot coerente
