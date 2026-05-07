# 🧠 Nota de Estratégia: Tentativa 93 - Equilíbrio Tático

## 🎯 Objetivo
Quebrar o platô de 53% observado na T92 através do reequilíbrio entre a detecção de presença (Focal) e a autoridade de ranking (MSE).

## 🛠️ Ajuste de Rota
A Tentativa 92 provou que o modelo aprende rápido a detectar hotspots, mas "preguiçoso" na ordenação fina, levando a gradientes rasos e saturação precoce.

**As mudanças para a T93 são:**
1.  **Autoridade de Ranking (1.0 → 10.0):** Devolvemos o "tempero" à Loss. O modelo agora será punido severamente por inversões de ranking dentro do Top 20. Isso deve manter a magnitude do gradiente saudável por mais tempo.
2.  **Blindagem Residual (0.2 → 0.4):** Como a janela de 30 dias é pequena, o Dropout de 0.2 facilitou o overfitting. O aumento para 0.4 força a rede residual de 2 camadas a encontrar caminhos mais genéricos de inteligência espacial.
3.  **Sinergia com o Challenger:** O GAT agora entrega um ranking mais robusto e "musculado", facilitando a vida do Challenger na classificação final.

## 📈 Expectativa
Manter o gradiente acima de 1.0 por pelo menos 20 épocas e buscar romper a barreira dos 55% de P@20 em validação cega.
