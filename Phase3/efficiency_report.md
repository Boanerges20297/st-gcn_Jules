# Relatório de Análise de Eficiência e Validação

## 1. Visão Geral
Este relatório apresenta a comparação de eficiência entre a arquitetura atual (ST-GCN) e o protótipo proposto (ST-GAT), considerando cenários com dados completos e cenários de escassez de dados (50% no-data).

## 2. Metodologia
- **Hardware**: CPU (Simulado) / GPU (se disponível)
- **Dados**: `data/processed/graph_data` (Reais)
- **Dimensões**: Nodes=2378, Channels=2, TimeSteps=12
- **Cenários**:
  - Full Data: Todos os nós com features completas.
  - 50% Data: 50% dos nós com features zeradas (simulando falta de dados históricos).

## 3. Resultados de Performance
| Model             |   Avg Latency (ms) |   Std Latency (ms) |   Memory Delta (MB) |
|:------------------|-------------------:|-------------------:|--------------------:|
| ST-GCN (Full)     |             446.2  |             160.06 |                0    |
| ST-GAT (Full)     |            2876.17 |             292.71 |              -12.18 |
| ST-GCN (50% Data) |             333.26 |              56.4  |               -0.09 |
| ST-GAT (50% Data) |            2922.18 |             165.76 |               16.47 |

## 4. Validação Técnica
- **ST-GCN com 50% Data**: Sucesso (Shape Saída: torch.Size([1, 2378, 1]))
- **ST-GAT com 50% Data**: Sucesso (Shape Saída: torch.Size([1, 2378, 1]))

## 5. Conclusões Preliminares
- **Latência**: O GAT tende a ser mais pesado devido ao cálculo da matriz de atenção densa (NxN), enquanto o GCN usa matrizes esparsas fixas.
- **Robustez**: Ambos os modelos processam dados esparsos tecnicamente, mas o GAT tem potencial teórico de adaptar os pesos de atenção para ignorar nós zerados (via mecanismo de atenção), enquanto o GCN propaga zeros fixamente pela topologia.

