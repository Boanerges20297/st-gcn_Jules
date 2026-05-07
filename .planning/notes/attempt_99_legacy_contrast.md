# 🧠 Nota de Estratégia: Tentativa 99 - Operação Legacy Contrast

## 🎯 Objetivo
Romper o platô matemático da Época 5 e resgatar o teto técnico de elite observados em meados de 2025 (T46), adaptando a inteligência para os padrões de criminalidade de 2026.

## 🏗️ O Retorno da Profundidade (DeepSTGAT_64)
Após 98 tentativas, identificamos que as arquiteturas rasas (1 ou 2 camadas) atingiam a saturação precoce devido à incapacidade de processar o "contraste fino" entre bairros hotspots.
1.  **Arquitetura:** Restauração dos 3 blocos STGCN (DeepSTGAT_64).
2.  **Contexto Longo:** Janela expandida para **120 dias** para fornecer estabilidade estatística à rede profunda.
3.  **Normalização V6 Pure:** Remoção do self-loop automático para isolar o sinal tático das armas (15x) da influência de vizinhança.

## 💉 TacticalContrastLoss (A Quebra do Platô)
A grande inovação da T99 foi a introdução do **Hard Negative Mining** via Loss de Contraste.
*   **Mecânica:** O modelo agora é punido pela distância entre a média dos hotspots e a média dos **Top 10% Falsos Positivos**.
*   **Resultado:** Isso impediu que a rede "desistisse" de aprender após identificar os hotspots óbvios, forçando o refinamento do ranking durante as 13 horas de treino.

## 📊 Veredito da Vitória (T99)
- **Recorde de P@20:** **55.14%** (Consolidação do teto histórico).
- **Recorde de P@10 (Foco):** **37.65%** (Um salto de 10% em relação ao baseline).
- **Resiliência:** O modelo sustentou o patamar de 54% sob Learning Rate criogênico (0.00001) por mais de 40 épocas.

## 🚀 Impacto em Produção (Champion-Challenger)
O modelo T99 assume a posição de **Champion** de Fortaleza. Sua estabilidade no Top 10 fornece a "âncora" necessária para que o **Challenger (Sentinela V3)** realize ajustes finos horários, com potencial de levar o P@20 final para a casa dos **60%**.

---
*Última Atualização: 02 de Maio de 2026*
