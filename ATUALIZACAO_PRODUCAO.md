# Registro de Atualização em Produção - Foco CVLI (Report Preview)

## Data: 2026-05-21
## Objetivo: Transição do modelo para foco exclusivo em CVLI e mitigação de vício histórico (Caso Aerolândia).

### Alterações Realizadas:
1. **Janela Temporal:** Reduzida de 120 para 14 dias no Orquestrador para Fortaleza e Interior.
2. **Pesos de Risco (CVLI):**
   - **Neural (0.50):** Agora com Fator Anti-Amnésia.
   - **Gatilho de Conflito/Inclusão (0.40):** Peso quadruplicado para responder instantaneamente a novos eventos de sangue.
   - **Tensão Territorial (0.10):** Reduzida para ser apenas um critério de desempate/contexto.
3. **Fator Anti-Amnésia (Decay Factor):** Implementado decaimento de 50% na confiança neural em áreas de calmaria severa (30 dias sem crimes). Isso evita que o histórico antigo da Aerolândia mantenha o status como CRÍTICO sem fatos novos.
4. **Calibração Dinâmica:** Expandidos limites do `ModelCalibrator` (tension_factor min 0.10) para permitir que o sistema se livre de ruído estatístico de facções se a precisão cair.

---
### LOG DE EXECUÇÃO:
- [x] Ajuste da janela temporal para 14 dias.
- [x] Implementação da lógica de pesos CVLI + Decay Factor no Orchestrator.
- [x] Expansão dos limites do ModelCalibrator.
- [x] Sincronização de referências internas (Comentários e Logs).
- [x] Validação da persistência via reinicialização simulada.
- [x] Treinamento "Honesty Paradigm" (Tentativa 79):
    - **Fortaleza P@10:** 37.61% (Recorde E6).
    - **Interior P@10:** 38.51% (Estável).
    - **RMF:** Corrigida e integrada ao fluxo de 41 canais.

### Hiperparâmetros de Produção (Sentinela V4):
- **Arquitetura:** ShallowGAT (Blindagem contra Overthinking).
- **Canais:** 41 (37 base + 4 Momentum/ColdStreak).
- **Janela (Lookback):** 14 dias (Foco em dinâmica reativa).
- **Loss:** Binary Focal Ranking + Honesty Constraint (Penalty em Calmaria).
- **Vault:** MemPalace V4 Ativado (Injeção via Canal 37).

**Status Atual:** O modelo de Fortaleza atingiu 37.61%, aproximando-se da meta de 40%. O "dedo" já está apontado sem os vícios históricos da Aerolândia.
