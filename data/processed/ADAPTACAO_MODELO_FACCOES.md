# ADAPTAÇÃO DE MODELO ST-GCN COM DINÂMICA DE FACÇÕES

## Arquitetura Modificada

```
INPUT: X(T, N, 7)
  ├─ [0-2] Branch 1: Crime Features
  │         ├─ Encoder(3 → hidden_dim)
  │         └─ ReLU + Dropout
  │
  └─ [3-6] Branch 2: Faction Dynamics
          ├─ Encoder(4 → hidden_dim/2)
          └─ Pad(hidden_dim/2 → hidden_dim)

FUSION LAYER:
  ├─ Multi-head Attention (crime, faction_dynamics)
  └─ Residual connection: crime + 0.3 * attention

TEMPORAL: LSTM(2 layers, hidden_dim)
SPATIAL: GraphConv(hidden_dim → hidden_dim)
OUTPUT: Decoder(hidden_dim → 1) + ReLU
AUX: Change Probability(hidden_dim → 1) + Sigmoid
```

## Features de Entrada (7 dimensões)

**0.** CVLI (homicídios)
**1.** Prisões
**2.** Apreensões
**3.** Mudança de controle territorial (0-1)
**4.** Estabilidade do controle (dias, 0-365)
**5.** Risco de conflito (0-1)
**6.** Volatilidade territorial (0-1)

## Loss Function Dinâmica

```
L_total = L_main + L_auxiliary

L_main = MSE(pred, target) * dynamic_weight
  where: dynamic_weight = 1 + (mudança * 2) + (volatilidade * 0.5)

L_auxiliary = BCE(mudança_pred, mudança_real) * 0.5
```

## Fluxo de Treinamento

1. **Encode**: Separar features de crime e facções
2. **Attend**: Multi-head attention para fusão contextual
3. **Temporal**: LSTM captura padrões históricos
4. **Spatial**: Graph convolution captura vizinhança
5. **Predict**: Decoder prediz CVLI + mudanças territoriais
6. **Loss**: Calcula weighted loss considerando dinâmica

## Benefícios da Adaptação

✅ Captura mudanças de poder territorial
✅ Aumenta peso onde há conflito (maior incerteza)
✅ Predição auxiliar de mudanças de controle
✅ Mantém signal de crime como principal
✅ Flexível para adicionar mais features de inteligência
