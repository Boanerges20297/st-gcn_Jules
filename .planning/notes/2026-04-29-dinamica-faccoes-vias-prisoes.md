---
date: 2026-04-29
title: O Tempero Faltante - Dinâmica de Facções, Vias e Prisões
context: Exploração sobre a estagnação da performance na V4 e a intuição tática de campo.
---

## Observações Críticas

1. **Modelagem Retilínea vs Abstrata**: O modelo `DeepSTGAT` atual pode estar "pensando demais". Padrões criminais de retaliação ("sangue paga sangue") são regras rígidas. Uma rede neural mais rasa (menos neurônios, menos camadas) acoplada a matrizes precisas pode reter o sinal bruto muito melhor do que uma rede profunda que abstrai a realidade.
2. **Matriz de Inimizade Tática**: Em vez de apenas adjacência geográfica contígua, o modelo precisa de uma Matriz Tática que conecte nós com base em:
   - Mapa claro de facções rivais (quem é inimigo de quem).
   - Malha viária principal (avenidas que servem de rota de ataque/fuga motorizada).
3. **O Proxy de Fragilidade Territorial**: Para inferir "feridas no coração" de uma facção:
   - **Prisões**: Desestabilizam a liderança e a mão de obra.
   - **Apreensões de Armas**: Diminuem a capacidade de defesa/ataque imediato (Peso altíssimo na retaliação).
   - **Apreensões de Drogas**: Desestabilizam o fluxo financeiro.
4. **Comportamento Policial**: A polícia satura vias secundárias e ruelas dominadas, deixando o crime deslocar-se pelo "efeito balão", enquanto os ataques de retaliação ocorrem nas vias primárias.

## Decisão
Essas dinâmicas não devem ser diluídas como "variáveis numéricas" na input matrix (node features). Elas devem esculpir a **topologia do grafo** (edge weights) para que o modelo entenda por onde o risco caminha.
