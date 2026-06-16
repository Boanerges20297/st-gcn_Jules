# Sistema de Atualizacao de Modelos

O fluxo oficial atual e manual, controlado e orientado por validacao.

## Estado atual

Nao usamos mais como caminho principal:

- retreino automatico em background ao detectar mudancas em `data/raw/`
- promocao automatica de modelos deep como champion

O caminho oficial agora e:

1. atualizar a base enriquecida
2. reprocessar os dados
3. rodar o script de promocao Poisson estadual
4. reiniciar a aplicacao
5. registrar e acompanhar a validacao

## Script oficial

```powershell
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

Esse script:

- treina e promove `poisson_regressor` para `fortaleza`
- replica a estrategia para `rmf`
- replica a estrategia para `interior`
- grava artefatos em `models/active/production/poisson/`

## Dependencias operacionais

- base enriquecida em `data/raw/`
- processados coerentes com a base
- `scikit-learn` disponivel no ambiente

## Cadencia recomendada

- rotina normal: `1x por semana`
- com nova carga relevante de dados: retreinar apos atualizacao
- com perda persistente de qualidade: antecipar retreino

## Evidencia de qualidade

As principais fontes de acompanhamento sao:

- `docs/CVLI_STOCHASTIC_BENCHMARK.md`
- `outputs/benchmarks/`
- `VALIDATION_LOG.md`

## Resultado esperado apos promocao

- `fortaleza_poisson_regressor.pkl`
- `rmf_poisson_regressor.pkl`
- `interior_poisson_regressor.pkl`

Todos em:

- `models/active/production/poisson/`
