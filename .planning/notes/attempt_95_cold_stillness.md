# 🧠 Nota de Estratégia: Tentativa 95 - Cold Stillness

## 🎯 Objetivo
Estabilizar a convergência em topologias de erro íngremes (Ranking Weight 12.0) através da remoção de aceleração cinética e aumento do contexto temporal.

## 🛠️ Ajuste de Rota (Pós-Colapso T94)
A Tentativa 94 demonstrou que o `OneCycleLR` é incompatível com pesos de ranking elevados. A aceleração do scheduler gerou passos que "expulsaram" o modelo da zona de convergência.

**Estratégia de Inércia Máxima (T95):**
1.  **LR Estático (0.0002):** Remoção total de schedulers dinâmicos. O modelo agora operará em uma "velocidade de cruzeiro" baixa e constante. Isso garante que o gradiente de ranking (que é naturalmente forte) seja aplicado de forma suave e cumulativa.
2.  **Contexto Ampliado (Window 30d → 60d):** Dobramos a janela de observação. Mais contexto temporal ajuda a suavizar as variações espúrias no grafo e fornece uma base mais sólida para o cálculo de ranking, reduzindo a variância do gradiente.
3.  **Foco em Refinamento:** Sem a pressão de "correr" contra o scheduler, a rede residual tem tempo para ajustar os pesos das camadas táticas com precisão microscópica.

## 📈 Expectativa
Uma descida de loss lenta, porém ininterrupta. Esperamos que o modelo cruze a barreira dos 50% de forma tardia (Época 40-50), mas que mantenha a trajetória ascendente até o final das 120 épocas, sem o risco de colapso estrutural.
