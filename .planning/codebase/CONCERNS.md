# Technical Concerns

## 1. Falha do Paradigma MemPalace (Canal 38)
- **Problema:** A "memória de aprendizado" entre épocas via `TrainingVault` (tentativa de ensinar o modelo a não repetir erros) não se mostrou eficiente para atingir a meta de 70% P@20.
- **Causa Técnica:** O feedback de surpresas cria um viés espacial estático que compete com os sinais temporais dinâmicos do GCN, gerando instabilidade no gradiente em vez de refinamento.
- **Impacto:** O modelo estagna em ~53% P@20 em Fortaleza, com degradação após poucas épocas.

## 2. Hybrid System Calibration
- **Issue:** The blend between Champion (ST-GAT) and Challenger (Sentinela) relies on real-time evaluation.
- **Concern:** If both models underperform in a specific period, the EMA might fluctuate significantly.

## 3. Data Leakage Risks
- **Precedent:** Past attempts (T48) identified data leakage from `random.shuffle`. 
- **Check:** Ensure the Temporal Split (85/15) in `train_all_specialists.py` is strictly enforced and that no future information leaks into the window normalization.

## 4. Hardware Constraints
- **Observation:** Training is running on CPU (`device=cpu` in logs). 
- **Impact:** Slow iteration cycles (18h for full training) hinder rapid experimentation with new strategies.
