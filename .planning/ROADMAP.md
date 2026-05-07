# 🗺️ ROADMAP — Milestone: Retreino Tático do Challenger

## 🔵 Fase 1: Sincronização e Auditoria de Dados
- [ ] Validar integridade dos CSVs em `data/raw/` (Timestamps de Maio/2026).
- [ ] Verificar presença de eventos CVLI em Abril/2026 no dataset enriquecido.

## 🔵 Fase 2: Retreino do Modelo Base (Freeze)
- [ ] Executar `tests/Sentinela/freeze_total_v3.py`.
- [ ] Analisar `freeze_report.txt` para conferir se o modelo incorporou o histórico completo.
- [ ] Registrar tentativa em `TRAINING_LOG.md`.

## 🔵 Fase 3: Fine-Tuning de 30 Dias
- [ ] Executar `tests/Sentinela/finetune_realtime_v1.py` com `--janela 30`.
- [ ] Validar se o Fine-Tuner superou a base no período recente (Critério de Ativação).
- [ ] Gerar `ranking_realtime.csv`.

## 🔵 Fase 4: Validação e Promoção
- [ ] Comparar resultados com o baseline de Abril (T57).
- [ ] Promoção automática para `models/active/` se P@10 ≥ 50%.
- [ ] Atualizar status no `GEMINI.md`.

---
*Próximo passo: /gsd-plan-phase 1*
