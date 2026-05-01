# 🧠 Nota de Estratégia: Tentativa 94 - Precisão Criogênica

## 🎯 Objetivo
Superar o recorde de 53.49% (T92) através da estabilização da convergência e refinamento da autoridade de ranking.

## 🛠️ Ajuste de Rota (Pós-Caos T93)
A Tentativa 93 provou que o modelo tem "musculatura" para lidar com ranking agressivo, mas o passo (`lr=0.003`) era grande demais para o peso da bota. O modelo estava "quicando" na superfície de erro, impedindo o pouso em mínimos globais.

**Configurações de Alta Estabilidade (T94):**
1.  **Carga Controlada (LR 0.003 → 0.0008):** Redução drástica da taxa de aprendizado. Cada atualização de peso agora é cirúrgica, permitindo que a rede residual sinta a pressão do ranking sem desestabilizar os filtros já aprendidos.
2.  **Autoridade Reforçada (Ranking 10 → 12):** Aproveitamos o passo curto para exigir ainda mais precisão na ordem. Inversões de ranking no Top 20 serão punidas com rigor extremo.
3.  **Redução de Ruído (Dropout 0.4 → 0.3):** O excesso de dropout estava "cegando" a rede em janelas curtas de 30 dias. Com 0.3, mantemos a regularização sem sacrificar a captura de micro-padrões semanais.
4.  **Sinergia de Stacking:** O GAT agora entrega um "Score de Presença + Ordem" muito mais confiável, servindo como uma feature de elite para o LightGBM (Challenger) realizar a classificação final.

## 📈 Expectativa
Uma curva de aprendizado monotônica (sem os saltos bruscos da T93) e um P@20 consolidado acima de **55%** ao atingir a fase de cool-down do scheduler.
