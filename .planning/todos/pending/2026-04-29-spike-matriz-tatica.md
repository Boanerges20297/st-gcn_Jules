---
created: 2026-04-29 23:26:27
title: "Spike: Matriz de Adjacência Tática (A_tactical)"
area: tooling
files:
  - data/raw/
---

## Problem
Precisamos testar a viabilidade de construir uma matriz de adjacência "A_tactical" para substituir/complementar a matriz geográfica.
Essa matriz deve incorporar o conhecimento tático:
1. Malha Viária (vias de acesso rápido).
2. Oposição de Facções (quem é vizinho inimigo).
3. Peso por apreensões (Armas e Drogas) e Prisões.

## Solution
Criar um script em `scratch/` ou `tests/Sentinela/` (Spike) que:
1. Leia o `bairros_centros_latlong.json` e o `inteligencia_faccoes.csv` (ou equivalente).
2. Construa um grafo viário conectando nós.
3. Cruza ocorrências de armas/drogas para atribuir pesos de fragilidade.
4. Visualize a matriz ou valide se os nós com mais armas/prisões ganham conexões de vulnerabilidade mais fortes.
Se der certo, integraremos isso ao `processed_fortaleza.pkl`.
