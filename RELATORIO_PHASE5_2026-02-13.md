# Relatório de Operação: Phase 5 - Escala Massiva (Fortaleza 121)
**Data:** 13 de Fevereiro de 2026
**Status:** Treino em Regime de Consolidação (2024-2026) ativo.

---

## 1. Sumário das Últimas 24 Horas
Saímos de um modelo estagnado em dados reduzidos (181 dias) para uma arquitetura de alta capacidade processando **1001 dias de histórico**, com foco cirúrgico nos **121 bairros oficiais de Fortaleza** e integração de inteligência tática de rua.

---

## 2. Cronologia de Manobras e Pivotagens

### A. Expansão de Dados e Limpeza (Manhã)
*   **Problema:** O modelo operava com apenas 181 dias, causando overfitting severo.
*   **Solução:** Reprocessamento massivo via CSV enriquecido, atingindo **1001 dias (2022-2026)**.
*   **Ajuste de Hardware:** Implementação de `LazyCrimeDataset` e `num_workers=0` para estabilizar o i5 com 48GB de RAM no Windows.

### B. Refino de Features e Inteligência de Rua (Tarde)
*   **Separação Logística:** O CVP geral foi removido por gerar ruído. Criamos o **Canal 1 (VEHICLE_LOGISTICS)**, focando apenas em roubos/furtos de veículos (preditor direto de CVLI).
*   **Expansão do Alvo (CVLI_PLUS):** Incluímos tentativas, lesões a bala e intervenções letais no Canal 0, dobrando a densidade de eventos positivos (**1121 alvos**).
*   **Alerta de Incursão (Canal 26):** Implementamos lógica cross-faction: se uma facção rouba veículo, os bairros rivais acendem no ranking automaticamente por 7 dias.
*   **Choque de Inteligência (Canal 25):** Integração do `exogenous_events.json` para capturar "eventos silenciosos" (expulsão de moradores e vácuo de poder).

### C. A Cirurgia dos 121 Bairros
*   **Problema:** O sistema via 138 bairros (intrusos da RMF e duplicatas).
*   **Ação:** Limpeza manual rigorosa. Removemos Caucaia/Maracanaú e unificamos subdivisões (ex: Conjunto Ceará I e II).
*   **Resultado:** Chegamos aos **121 bairros oficiais**, aumentando drasticamente a densidade estatística por nó.

### D. Otimização de Performance (ST-GAT -> FastRelationalGCN)
*   **Mudança:** O GAT dinâmico era pesado demais para o i5 (11 horas de treino).
*   **Solução:** Criamos a `FastRelationalGCN`. Ela usa multiplicações de matrizes pré-normalizadas (Geo e Conflito), mantendo a inteligência de longa distância mas reduzindo o tempo de época de 15 min para **~1.5 min**.

---

## 3. O Problema do Centro e a Solução de Pulso
*   **Observação:** O Centro aparecia em #2 no ranking indevidamente.
*   **Causa:** Ruído de conflitos interpessoais não letais.
*   **Solução:** Criamos o **Canal 28 (CITY_PULSE)**. O modelo agora aprende o "ritmo da cidade", diferenciando dias de calmaria de dias de anomalia. Aplicamos um filtro antirruído no Canal 0 (apenas letalidade por arma de fogo).

---

## 4. Estratégia Atual: SHIFT DE REGIME (O FOCO DE HOJE)
**Contexto Tático:** Identificamos que a guerra de 2022 (conquista de território) é diferente da realidade de 2025 (consolidação do CV e ordens vindas do RJ).

**Manobra Executada:**
Implementamos a `TemporalRegimeLoss`.
*   **Contexto Histórico (2022-2023):** Peso 0.2. Serve apenas para o modelo entender a geografia e as alianças.
*   **Momentum Atual (2024-2026):** Peso Exponencial (até 5.0). O modelo é severamente punido se errar o padrão de consolidação atual.
*   **Objetivo:** Romper o platô de 0.07 e atingir a Taxa de Captura (Recall@10) acima de 50%.

---

## 5. Estado Atual do Modelo (Snapshot)
*   **Arquitetura:** 29 canais, 3 camadas residuais, 64 canais internos.
*   **Dataset:** Fortaleza 121 (1001 dias).
*   **Métrica Alvo:** Recall@10 (Taxa de Captura de crimes reais).
*   **Checkpoint:** `models/phase5/best_stgat_v5_massive.pth`.

---

## 6. Roadmap para Amanhã
1.  **Validar Recall:** Verificar se o peso temporal rompeu os 50% de captura.
2.  **Live Test:** Executar o ranking para os bairros do CV consolidado (Barroso, Jardim das Oliveiras, etc).
3.  **Ajuste de Hiperparâmetros:** Se a oscilação persistir, reduzir o `learning_rate` na fase final.

---
**Assinado:** Gemini CLI (Peer Programmer)
**Data de Referência:** 13/02/2026
