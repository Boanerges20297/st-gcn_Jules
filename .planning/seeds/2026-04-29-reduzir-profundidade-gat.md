---
title: "Redução da Profundidade do ST-GAT (Shallow GAT)"
planted_date: 2026-04-29
trigger_condition: Quando a Matriz de Adjacência Tática (A_tactical) for aprovada no Spike e injetada no pickle.
---

# Seed Idea

Ao injetar inteligência militar diretamente na estrutura do grafo (A_tactical com inimizades, prisões e vias rápidas), o modelo não precisa mais adivinhar abstrações profundas. A rede atual (`DeepSTGAT_64`) pode estar "pensando demais" (over-smoothing) e borrando as bordas rígidas do conflito de rua.

## Ação a ser tomada no Trigger
1. Criar uma nova arquitetura `ShallowSTGAT` ou `TacticalGAT` em `architectures.py`.
2. Remover blocos residuais profundos. Reduzir as camadas espaciais e temporais para o mínimo necessário para propagar o sinal viário em 1 ou 2 hops (salto do bairro atacante pro atacado).
3. Avaliar se o P@10 aumenta com o modelo retilíneo focado nos dados táticos brutos.
