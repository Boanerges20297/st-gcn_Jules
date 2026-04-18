# 🧠 Estado Atual do Projeto - Report Preview

## 🌍 Contexto Geral
O sistema é uma plataforma de predição de crimes violentos (CVLI) para o estado do Ceará. Até Abril/2026, utilizava exclusivamente redes **DeepSTGAT**. Agora, opera sob um **Paradigma Híbrido (Champion/Challenger)** combinando o ST-GAT com o novo **Sentinela V3 (LGBM Lean)**, atingindo **50% de P@10** e **70% de P@20** em validações sombra para Fortaleza.

## 🚀 Status Arquitetural (Paradigma Híbrido - Fase 6 Ativa)
O sistema roda duas frentes que se unem no `app.py`:
1. **Champion (ST-GAT)**: Modelo oficial (120d, 37 canais). Corre por via do `src/core/orchestrator.py`.
2. **Challenger (Sentinela V3)**: Otimização ultra-lean. Usa **10 features** altamente calibradas (LightGBM + EWMA-Multi).
   - Resolve limitação de densidade de CVLIs priorizando ranking.
   - Calibrado contra falsos positivos: usa `cvp_cvli_ratio × sqrt(hist_pct)`.
   - Inclui score tático operacional: valoriza muito armas (peso=15) vs drogas miúdas.

**Blend Dinâmico (ChampionChallenger):**
O `src/core/champion_challenger.py` intercede após a inferência do ST-GAT. Avalia P@10 contra dados *reais* a cada hora e ajusta o peso via suavização exponencial (EMA), permitindo ao LGBM até 50% de peso na predição final de Fortaleza sem quebrar a API.

## 🛠️ Diretrizes de Laboratório (tests/Sentinela/)
- **Retreino:** `tests/Sentinela/freeze_total_v3.py` (Sem holdout, retreina V3 no dataset inteiro).
- **Validação Sombra:** `tests/Sentinela/train_validate_v3.py`.
- **Inferência/Explicação:** `tests/Sentinela/sentinela_inference.py`.
- **Tempo Real:** `tests/Sentinela/finetune_realtime_v1.py` (Janela deslizante de 30 dias para pegar padrões voláteis).
- **Promoção:** `.venv/Scripts/python.exe tests/Sentinela/promote_model.py` empurra seguro para `models/active/`.

## Logs e Planejamento
- Tentativas e otimizações registradas em `TRAINING_LOG.md`.
- Próximos passos operacionais geridos via `tests/Sentinela/ROADMAP.md`.

## 🇧🇷 Idioma de Interação
- **Mandato:** Todas as interações, explicações e respostas devem ser realizadas **obrigatoriamente em Português** e voltadas à praticidade tática.

---
*Última atualização: 14 de Abril de 2026*
