# REQUIREMENTS — Retreino Tático do Challenger

## 1. Escopo Técnico
- **Input:** `data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv` (atualizado).
- **Processamento:**
  - Retreino total do LGBM Lean (Base).
  - Fine-tuning incremental (30 dias).
- **Output:**
  - `models/active/lgbm_lean_v3_freeze.pkl`
  - `tests/Sentinela/lgbm_finetune_current.pkl`
  - `tests/Sentinela/ranking_realtime.csv`

## 2. Critérios de Aceite (UAT)
- [ ] **Cobertura:** Ranking deve cobrir o Top 40 bairros de Fortaleza.
- [ ] **Performance:** P@10 ≥ 50% na validação sombra (últimos 14 dias).
- [ ] **Integridade:** O arquivo `.pkl` deve ser compatível com o `app.py` atual.
- [ ] **Explicabilidade:** O relatório de importância de features deve destacar `cvp_cvli_ratio` como principal preditor.

## 3. Restrições
- Não alterar a arquitetura de 10 features do Sentinela V3.
- Manter o idioma de interação e logs em Português.
