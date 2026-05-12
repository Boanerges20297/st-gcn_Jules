# Estado Atual do Milestone: Retreino Tático do Challenger

## 📌 Progresso Geral
- **Fase 1:** Concluído ✅
- **Fase 2:** Concluído ✅
- **Fase 3:** Concluído ✅
- **Fase 4:** Planejado ⏳
# Estado Atual do Milestone: Retreino Tático do Challenger

## 📌 Progresso Geral
- **Fase 1:** Concluído ✅
- **Fase 2:** Concluído ✅
- **Fase 3:** Concluído ✅
- **Fase 4:** Planejado ⏳

## 🛠️ Notas de Contexto
- **Correção Crítica:** Corrigido `FileNotFoundError` no script `freeze_total_v3.py` causado por `BASE_PATH` fixo (hardcoded) para outro usuário. Agora o caminho é dinâmico.
- O modelo `lgbm_lean_v3_freeze.pkl` foi gerado e promovido para `models/active/`.
- A janela de foco para o modelo é de 14 dias (horizonte) e 60 dias (window).
- A promoção foi realizada manualmente após validação técnica da execução.
- **Incidente (2026-05-12):** Detectado e corrigido `SyntaxError` no `src/core/orchestrator.py` causado por conflitos de mesclagem (git conflict markers). Sistema restabelecido com arquitetura **ShallowGAT (ResGAT)** para Fortaleza.
- **Correção de UI (2026-05-12):** Resolvida discrepância entre cards de métricas e mapa. As contagens estavam infladas por nós duplicados; implementada deduplicação no backend e aumentada a visibilidade dos polígonos no mapa.

## 🚨 Bloqueios / Riscos
- Monitorar a performance do novo modelo em tempo real para detectar possível drift.
- Ambiente virtual (.venv) apresentou erro de launcher no `pip`, mas a execução via script está funcional.
- Limpeza de logs e arquivos de dados necessária devido a conflitos de mesclagem residuais.
