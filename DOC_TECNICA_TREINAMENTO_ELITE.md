# 🧠 Guia Técnico: Arquitetura e Dinâmica de Treinamento (Elite P10)

Este documento detalha os conceitos fundamentais utilizados no sistema de predição de crimes **Report Preview**, com foco na arquitetura **DeepST-GAT** e no monitoramento de performance.

---

## 🚀 1. Arquitetura Neural: DeepST-GAT 64

O **DeepST-GAT (Spatial-Temporal Graph Attention Network)** é o "cérebro" do sistema. Ele não olha apenas para dados isolados, mas para a **tensão territorial** como um organismo vivo.

### Por que "Deep" (Profundo)?
Diferente de modelos lineares, o DeepST-GAT possui **3 camadas de convolução em grafos**. Isso permite que o modelo entenda:
1.  **Vizinhança Imediata:** O que acontece em um bairro e afeta o vizinho (1ª camada).
2.  **Influência Regional:** Como uma mancha de crime se espalha por um distrito (2ª camada).
3.  **Dinâmica de Cidade:** Como o crime em um polo afeta a dinâmica macro (3ª camada).

### Os "Canais" (64 Neurônios)
Cada um dos 64 canais internos do modelo atua como um **especialista**:
*   Alguns canais focam apenas em **sazonalidade** (Sextas-feiras e feriados).
*   Outros focam em **tensão de facções** (Quem domina onde).
*   Outros são "termômetros" de **supressão policial** (Onde houve apreensões).
*   A rede combina o sinal de todos esses especialistas para gerar o **Risco Consolidado**.

### Atenção Espaço-Temporal (MHA)
O modelo utiliza **Multi-Head Attention (Atenção Multicabeça)**. Imagine 8 analistas olhando para o mesmo mapa:
*   Um foca no tempo (Últimos 90 dias).
*   Outro foca no espaço (Bairros vizinhos).
*   A rede "ouve" todos e dá mais peso para quem está vendo o padrão mais crítico no momento.

---

## 📊 2. Dinâmica de Treinamento: A "Corrida"

O treinamento é o processo de ajustar os 1.2 milhões de parâmetros do modelo para que ele acerte o **Top 10** do Dashboard.

### A "Caminhada" do Gradiente (Grad Norm)
O Gradiente é a **força do ajuste** que o modelo faz nos seus pesos.
*   **Grad Baixo (0.1 - 0.5):** Ajuste fino. O modelo está "polindo" o que já sabe.
*   **Grad Moderado (0.6 - 1.5):** Aprendizado ativo. O modelo está descobrindo novos padrões.
*   **Grad Alto (> 2.0):** **Surpresa Tática.** O modelo errou feio em um padrão e está tentando mudar drasticamente para se corrigir. 
    *   *Nota:* Usamos o **Clipping (1.0)** para garantir que esses saltos não "quebrem" a rede, mantendo o aprendizado estável.

### OneCycleLR (Aceleração e Frenagem)
Usamos uma estratégia de **Taxa de Aprendizado (LR)** dinâmica:
1.  **Aquecimento (Pct Start 30%):** O modelo começa devagar para entender o terreno.
2.  **Aceleração Máxima (LR 0.03):** O modelo "corre" para aprender o máximo possível de padrões complexos.
3.  **Linha de Chegada (Final Div Factor):** O modelo reduz a velocidade drasticamente no final para "estacionar" no ponto exato de menor erro, garantindo a maior precisão possível.

---

## 🎯 3. Métricas de Sucesso (O que buscamos?)

### P@10 (Precision at 10)
É a nossa métrica principal. Se o P@10 é **60%**, significa que das 10 áreas que o modelo apontou como mais perigosas, **6 realmente tiveram crimes**. 
*   **Por que não usamos a ordem?** Para a segurança pública, saber que os 10 bairros X, Y e Z são os mais quentes é mais importante do que saber se o X é o 1º ou o 2º. O objetivo é a **alocação eficiente de equipes**.

### Loss (Erro de Tensão)
O Loss mede o quão longe a previsão do modelo está da realidade. 
*   **Top-K Set Loss:** Nossa função de perda personalizada que penaliza o modelo apenas se ele deixar um bairro perigoso fora do Top 10. Ela ignora erros de "ranking interno" para focar no que importa para o gestor.

---

## 🛡️ 4. Dropout (0.5) - A Blindagem
Configuramos um **Dropout de 50%**. Isso significa que, durante o treino, o modelo "esquece" metade do que viu aleatoriamente. 
*   **Objetivo:** Forçar o modelo a não depender de um único fator (ex: "Sempre morre gente no bairro A"). Ele é obrigado a aprender a **lógica por trás do crime** (vizinhança, tensão, calendário) para conseguir prever corretamente mesmo com "falha de memória".

---
*Documento gerado em 11 de Março de 2026 para o Sistema Report Preview.*
