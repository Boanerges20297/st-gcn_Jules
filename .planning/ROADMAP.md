# 🗺️ ROADMAP — Milestone: Retreino Tático do Challenger

## ✅ Fase 1: Sincronização e Auditoria de Dados (CONCLUÍDO)
- [x] Validar integridade dos CSVs.
- [x] Sincronização de limites de risco.

## ✅ Fase 2: Retreino do Modelo Challenger (CONCLUÍDO)
- [x] Executar `freeze_total_v3.py`.
- [x] Promoção para `models/active/`.

### ✅ Fase 2.1: Experimento Solo Flight (CONCLUÍDO)
- [x] Treinar LightGBM sem Ensemble (`solo_lgbm_flight.py`).
- [x] Comparar métricas P@10/P@20 (Baseline vs Solo).
- [x] Registrar conclusões no `TRAINING_LOG.md`.
- [x] Promover modelo vencedor (`lgbm_solo_challenger.pkl`) para `models/active/`.

## 🔵 Fase 3: Upgrade Neural do Champion (ST-GAT)
- [ ] Implementar Recency Bias no `train_all_specialists.py`.
- [ ] Expandir dados para 4 anos (2022).
- [ ] Executar treino e atingir P@20 >= 50%.

## 🔵 Fase 4: Fine-Tuning de 30 Dias (Sentinela)
- [ ] Executar `finetune_realtime_v1.py`.

---
*Próximo passo: /gsd-plan-phase 1*
