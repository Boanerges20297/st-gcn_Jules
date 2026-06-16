# Operação Atual

## Rotina diária

1. Atualizar/mesclar dados brutos.
2. Garantir que `data/processed/*.pkl` estejam coerentes.
3. Subir ou reiniciar o app.
4. Conferir:
   - `/api/risk`
   - `VALIDATION_LOG.md`
   - `outputs/hermes/`

## Retreino oficial

Script principal:

```powershell
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

Esse script:

- treina Fortaleza
- treina RMF
- treina Interior
- salva artefatos em `models/active/production/poisson/`

## Frequência recomendada

- regular: `1x por semana`
- obrigatório: após atualização relevante do `ENRIQUECIDO`
- antecipado: se `VALIDATION_LOG.md` cair por 2 ou 3 ciclos seguidos

## Pipeline mínimo de atualização

```powershell
.\.venv\Scripts\python.exe src\core\data_processing.py
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

Depois:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Benchmark oficial

Arquivos:

- [docs/CVLI_STOCHASTIC_BENCHMARK.md](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/docs/CVLI_STOCHASTIC_BENCHMARK.md)
- [scripts/benchmark_cvli_stochastic_suite.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/benchmark_cvli_stochastic_suite.py:1)
- [scripts/validate_cvli_finalists_monthly.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/validate_cvli_finalists_monthly.py:1)

## Onde olhar quando algo degradar

- `VALIDATION_LOG.md`
- `logs/efficiency_history.json`
- `outputs/hermes/risk_snapshot_latest.json`
- `outputs/hermes/risk_brief_latest.md`

## Scripts realmente operacionais hoje

- [scripts/merge_new_data.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/merge_new_data.py:1)
- [scripts/promote_statewide_poisson_regressors.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/promote_statewide_poisson_regressors.py:1)
- [scripts/export_static_snapshot.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/export_static_snapshot.py:1)
- [scripts/generate_pipeline_artifacts.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/generate_pipeline_artifacts.py:1)

## Itens históricos

Se você estiver vendo algo sobre:

- ST-GAT champion oficial
- blend champion/challenger como caminho principal
- `models/active/*.pth` como pesos ativos

isso é documentação ou script legado, não o fluxo de produção atual.
