# Models Layout

## Active

- `active/production/poisson/`
  - modelos de produção ativos por região
  - `fortaleza_poisson_regressor.pkl/.json`
  - `rmf_poisson_regressor.pkl/.json`
  - `interior_poisson_regressor.pkl/.json`

- `active/production/challengers/`
  - challengers opcionais e artefatos auxiliares ainda relevantes em produção
  - `lgbm_lean_v3_freeze.pkl`
  - `lgbm_solo_challenger*.pkl`

- `active/legacy_torch/`
  - checkpoints ST-GAT/ST-GCN antigos preservados para auditoria, rollback ou benchmark legado

- `active/metadata/`
  - arquivos informativos ligados ao estado operacional, sem serem pesos de modelo

## Archive

- `archive/backup/`
  - backups históricos

- `archive/known_bad/backup_viciado_shuffle/`
  - checkpoints marcados como problemáticos por contaminação/shuffle

## Experiments

- `experiments/test/`
  - checkpoints de teste e validação

- `experiments/tests/`
  - artefatos pequenos de testes rápidos

## Root metadata

- `metadata/`
  - inventários e logs administrativos da pasta `models`
