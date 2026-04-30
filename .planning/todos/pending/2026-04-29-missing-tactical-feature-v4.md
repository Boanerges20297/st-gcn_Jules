---
created: 2026-04-29 23:03:49
title: Missing tactical feature for Sentinela V4
area: modeling
files:
  - scripts/training/Active/train_all_specialists.py
---

## Problem

O desempenho do Sentinela V4 está estagnado (P@10 na casa dos 40%, P@20 perto de 53%).
Temos um bom volume de dados:
1. Volume de CVLI
2. Volume de ações policiais
3. Variáveis grandes e pequenas (clima, sazonalidade, etc.)
4. Tecnologia e entendimento da dinâmica tática.

Apesar disso, está faltando um "tempero", uma visão ou feature que ainda não estamos enxergando para que o modelo consiga "cravar" a predição.

## Solution

TBD.
- Necessário realizar uma sessão de brainstorming focado (spike ou explore) para investigar o que pode ser essa peça que falta.
- Possíveis caminhos: interações temporais não lineares, efeito de contágio geográfico que não está sendo capturado pelo GAT, ou alguma feature tática de retaliação entre facções.
