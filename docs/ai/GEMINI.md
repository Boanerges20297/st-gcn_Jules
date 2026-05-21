# 🧠 Estado Atual do Projeto - Report Preview

## 🌍 Contexto Geral
O sistema é uma plataforma de predição de crimes violentos (CVLI) para o estado do Ceará. Até Abril/2026, utilizava exclusivamente redes **DeepSTGAT**. Agora, opera sob um **Paradigma Híbrido (Champion/Challenger)** combinando o ST-GAT com o novo **Sentinela V3 (LGBM Lean)**, atingindo **50% de P@10** e **70% de P@20** em validações sombra para Fortaleza.

## 🚀 Status Arquitetural (Fase 7.5 - Intervenção Tática Ativa)
O sistema opera em regime de upgrade do núcleo neural:
1. **Intervenção Tática (A_tactical):** Substituição da adjacência geográfica por pesos de inteligência (Atrito de Facção e Fragilidade Viária 15x).
2. **ResGAT (Tactical Residual GAT):** Evolução do ShallowGAT para 2 camadas com skip connections, otimizando o "tempero" tático sem perder a identidade histórica (Window 60d).
3. **Normalização Row-Stochastic:** Controle de volume via $D^{-1} A$, preservando recordes de P@20 (54.2%).
4. **Paradigma Híbrido (Champion/Challenger):** O `app.py` continua unindo ST-GAT com Sentinela V3.

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
- Próximos passos operacionais geridos via `.planning/` e `tests/Sentinela/ROADMAP.md`.

## 🇧🇷 Idioma de Interação
- **Mandato:** Todas as interações, explicações e respostas devem ser realizadas **obrigatoriamente em Português** e voltadas à praticidade tática.

## Registro
- **Registro de Logs:** A cada nova tentativa SEMPRE REGISTRAR EM TRAININGS_LOG.MD E ATUALIZAR OS ARQUIVOS DO .PLANING.

---
*Última atualização: 01 de Maio de 2026*
