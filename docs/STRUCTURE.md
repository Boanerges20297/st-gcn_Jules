# Structure

## Diretórios principais

```text
Report Preview/
├── app.py
├── src/
│   ├── core/
│   └── agent/
├── scripts/
│   ├── training/
│   │   ├── Active/
│   │   └── Legacy/
│   ├── diversos/
│   ├── linux/
│   └── nodes/
├── docs/
│   ├── CURRENT_ARCHITECTURE.md
│   ├── CURRENT_OPERATIONS.md
│   ├── CVLI_STOCHASTIC_BENCHMARK.md
│   └── archive/
├── models/
│   ├── active/
│   │   ├── production/
│   │   ├── legacy_torch/
│   │   └── metadata/
│   ├── archive/
│   ├── experiments/
│   └── metadata/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── logs/
└── templates/
```

## Onde achar cada coisa

### Runtime

- gateway web: `app.py`
- inferência: `src/core/orchestrator.py`
- backend Poisson: `src/core/fortaleza_poisson_backend.py`
- validação: `src/core/validation_logger.py`
- monitor: `src/core/efficiency_monitor.py`

### Dados

- bruto: `data/raw/`
- processado: `data/processed/`

### Modelos

- produção: `models/active/production/`
- legado torch: `models/active/legacy_torch/`
- histórico: `models/archive/`
- experimentos: `models/experiments/`

### Scripts

- operação atual: `scripts/`
- deep legado: `scripts/training/Active/`
- histórico: `scripts/training/Legacy/`

## Regra prática

Se o arquivo não estiver em um destes grupos, confirme antes de tratá-lo como parte da arquitetura ativa:

- `app.py`
- `src/core/`
- `scripts/` operacionais
- `models/active/production/`
- `data/processed/`
