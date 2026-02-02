# Relatório de Teste de Predições - Modelo ST-GCN 3D
**Data:** 2026-02-01
**Período de Teste:** 2025-01-01 até 2026-01-12 (377 dias)

---

## 1. Configuração do Modelo

### ✅ Remoções Implementadas
- **Modelo CVP separado**: REMOVIDO
- Variável `model_cvp`: REMOVIDA
- Constante `MODEL_CVP_PATH`: REMOVIDA
- Apenas modelo 3D unificado mantido (`stgcn_model.pth`)

### 📐 Arquitetura Atual
```
Modelo: STGCN com 3 canais
├─ Entrada: (Batch, 3, Nodes, Time)
│  ├─ Canal 0: CVLI
│  ├─ Canal 1: CVP
│  └─ Canal 2: Tension
├─ Grafos: 2 (geográfico + conflito)
├─ Janela Temporal: 7 dias
├─ Nós: 319
└─ Saída: Predição de CVLI
```

**Parâmetros:** 19,009 (modelo compacto)

---

## 2. Resultados de Teste com Dados 2025

### 📊 Dados Utilizados
- **Período:** 2025-01-01 até 2026-01-12
- **Total de dias testados:** 377
- **Predições executadas:** 377/377 (100%)
- **Total de predições:** 120,263 (377 dias × 319 nós)

### 🎯 Métricas de Performance

#### Métricas Globais
| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **MAE** | 0.9854 | Erro médio absoluto ~1 crime/dia |
| **RMSE** | 0.9964 | Erro quadrático médio |
| **R²** | -30.16 | ⚠️ Modelo pior que baseline (média) |

#### Métricas para Casos com Crime Real (actual > 0)
| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **MAE** | 0.1666 | Quando há crime, erro médio é 0.17 |
| **RMSE** | 0.3900 | Variação do erro é razoável |

#### Métricas de Classificação (threshold=0.5)
| Métrica | Valor | Análise |
|---------|-------|---------|
| **Acurácia** | 2.32% | ❌ Muito baixa |
| **Precisão** | 2.32% | ❌ 98% de falsos positivos |
| **Recall** | 100% | ✅ Detecta todos os crimes |
| **F1-Score** | 0.0454 | ❌ Performance geral muito baixa |

### 🔍 Matriz de Confusão
```
                  Predito: Crime    Predito: Sem Crime
Real: Crime           2,795                0
Real: Sem Crime     117,468                0
```

**Análise:**
- ✅ **True Positives:** 2,795 - Detectou todos os crimes reais
- ❌ **False Positives:** 117,468 - Prediz crime onde não há (98%)
- ❌ **True Negatives:** 0 - Nunca prediz "sem crime" corretamente
- ✅ **False Negatives:** 0 - Não perde nenhum crime real

---

## 3. Análise de Distribuição

### 📈 Estatísticas
- **Crimes reais (actual > 0):** 2,795 (2.32% dos casos)
- **Crimes preditos (> 0.5):** 120,263 (100% dos casos)
- **Taxa de alarmes falsos:** 97.68%

### 🗺️ Performance por Região
Todas as 319 regiões classificadas como "unknown" (falta configuração `region_type`)

**Top 10 Regiões com Maior Erro:**
1. FORQUILHA - Erro: 1.026
2. INDUSTRIAL - Erro: 1.015
3. PACUJÁ - Erro: 1.012
4. EUSÉBIO - Erro: 1.011
5. ELLERY - Erro: 1.010

---

## 4. Diagnóstico do Problema

### ⚠️ OVERFITTING SEVERO

O modelo está **superestimando** a ocorrência de crimes:

#### Causas Prováveis:
1. **Threshold inadequado** (0.5 pode ser alto demais)
2. **Desbalanceamento de classes** (98% das instâncias são "sem crime")
3. **Normalização incorreta** durante treinamento
4. **ReLU final** pode estar elevando todos os valores

#### Comportamento Observado:
```python
Média predição: ~1.0 crime/dia em TODOS os nós
Média real: ~0.02 crimes/dia (muito rara)
```

O modelo está "aprendendo" a sempre prever valores > 0, o que maximiza recall mas destrói precisão.

---

## 5. Recomendações

### 🔧 Correções Imediatas

#### 1. Ajustar Threshold de Detecção
```python
# Testar thresholds mais baixos
thresholds = [0.1, 0.05, 0.01, 0.005]
```

#### 2. Normalizar Saídas do Modelo
```python
# Aplicar sigmoid ou normalização por percentil
predictions_normalized = np.percentile_rank(predictions)
```

#### 3. Recalibrar com Class Weights
Durante treinamento, usar weights:
```python
class_weight = {
    0: 1.0,      # Sem crime
    1: 50.0      # Com crime (classe minoritária)
}
```

### 📊 Análises Adicionais Necessárias

1. **Histograma de predições** - Ver distribuição real dos valores
2. **Curva ROC/AUC** - Encontrar threshold ótimo
3. **Análise temporal** - Verificar se modelo degrada ao longo do tempo
4. **Top-K acurácia** - Avaliar se os top K nós de maior risco são corretos

### 🎯 Métricas Alternativas

Para crime (eventos raros), usar:
- **Precision@K** - Dos K nós com maior risco, quantos têm crime?
- **MAP (Mean Average Precision)** - Ranking quality
- **NDCG (Normalized Discounted Cumulative Gain)**

---

## 6. Próximos Passos

### Curto Prazo
1. ✅ **Remover modelo CVP separado** - CONCLUÍDO
2. ⏳ Ajustar threshold de detecção
3. ⏳ Implementar métricas de ranking
4. ⏳ Analisar distribuição de predições

### Médio Prazo
1. Retreinar com class balancing
2. Implementar post-processing (calibração)
3. Adicionar validação temporal (walk-forward)
4. Criar baseline comparativo (modelo naive)

### Longo Prazo
1. Experimentar loss functions alternativas (Focal Loss)
2. Ensembling com múltiplos modelos
3. Incorporar features adicionais
4. A/B testing com versão anterior

---

## 7. Conclusões

### ✅ Pontos Positivos
- Modelo 3D funcional (3 canais: CVLI, CVP, Tension)
- **Recall de 100%** - Não perde nenhum crime
- Infraestrutura de teste automatizada
- CVP confirmado como feature ativa

### ❌ Problemas Críticos
- **Precisão de 2.32%** - Inaceitável para produção
- Overfitting severo (prediz crime em 100% dos casos)
- R² negativo indica modelo pior que baseline
- Falta calibração de probabilidades

### 🎯 Status de Viabilidade
**MODELO NÃO PRONTO PARA PRODUÇÃO**

Requer:
1. Recalibração de threshold
2. Implementação de métricas de ranking
3. Possível retreinamento com class weights

---

## Arquivos Gerados

1. **Script de Teste:** `scripts/test_predictions_2025.py`
2. **Resultados JSON:** `reports/test_results_2025.json`
3. **Este Relatório:** `reports/PREDICTION_TEST_REPORT_2025.md`

**Executar testes:**
```bash
python scripts/test_predictions_2025.py
```

---

**Última Atualização:** 2026-02-01 19:40:31
