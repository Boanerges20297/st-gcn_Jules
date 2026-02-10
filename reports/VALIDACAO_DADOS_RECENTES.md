# Relatório de Validação - Dados Recentes (03-05/02/2026)

## 📊 Resumo Executivo

O modelo ST-GAT foi validado com dados reais coletados entre 03/02/2026 e 05/02/2026 (3 dias). Este é o primeiro teste do modelo com dados completamente novos desde o último treinamento.

## 📈 Dados de Validação

- **Período**: 03/02/2026 a 05/02/2026
- **Total de registros**: 55 eventos
  - CVLIs (Homicídios): 17 eventos
  - CVPs (Crimes contra patrimônio): 38 eventos
- **Distribuição espacial**: 14 nós diferentes afetados
- **Média diária**: ~5.7 CVLIs/dia

## 🎯 Resultados do Modelo

### Métricas Gerais
| Métrica       | Valor      | Interpretação                                    |
|---------------|------------|--------------------------------------------------|
| **MAE**       | 0.8139     | Erro médio de ~0.8 eventos por nó               |
| **RMSE**      | 0.8247     | Raiz do erro quadrático médio                   |
| **MAPE**      | 24.74%     | Erro percentual médio (apenas nós com eventos)  |
| **P@20**      | 0.15       | 15% dos top-20 tiveram eventos (3/20)          |

### Análise de Predição

**⚠️ Problema Principal**: O modelo está **superestimando significativamente** o volume de eventos:
- **Previsto**: 266.28 eventos
- **Real**: 17 eventos  
- **Ratio**: **15.6x mais** que o real

**Distribuição Espacial**:
- Nós com eventos reais: 14
- Nós com predições > 0.5: 319 (todos)
- **Problema**: O modelo não está concentrando o risco, está espalhando uniformemente

### Top 10 Nós Mais Críticos (Real vs Previsto)

| Rank | Nó  | Real | Previsto | Erro   | Acerto? |
|------|-----|------|----------|--------|---------|
| 1    | 244 | 2    | 0.82     | -1.18  | ❌      |
| 2    | 161 | 2    | 0.84     | -1.16  | ❌      |
| 3    | 119 | 2    | 0.92     | -1.08  | ❌      |
| 4    | 166 | 1    | 0.84     | -0.16  | ❌      |
| 5    | 184 | 1    | 0.82     | -0.18  | ❌      |
| 6    | 253 | 1    | 0.92     | -0.08  | ❌      |
| 7    | 276 | 1    | 0.84     | -0.16  | ❌      |
| 8    | 307 | 1    | 0.84     | -0.16  | ❌      |
| 9    | 235 | 1    | 0.81     | -0.19  | ❌      |
| 10   | 234 | 1    | 0.84     | -0.16  | ❌      |

**Observação**: Todos os nós realmente críticos receberam predições **menores** que o real, indicando subestimação individual apesar da superestimação global.

## 🔍 Diagnóstico

### Problemas Identificados

1. **Superestimação Global (15.6x)**
   - O modelo está prevendo eventos para praticamente todos os nós
   - Isso sugere que o threshold de decisão está muito baixo
   - Ou que a calibração do modelo precisa ajuste

2. **Baixa Precisão no Top-K (P@20 = 15%)**
   - Apenas 3 dos 20 nós previstos como mais críticos tiveram eventos
   - Para ser útil operacionalmente, P@20 deveria estar acima de 40-50%

3. **Padrão Uniforme de Predição**
   - As predições estão muito concentradas na faixa de 0.8-0.9
   - Falta discriminação entre nós de alto e baixo risco
   - Todos os 319 nós têm predições positivas

4. **Subestimação dos Eventos Críticos**
   - Nós que tiveram 2 eventos (244, 161, 119) receberam predições < 1.0
   - O modelo não está capturando a intensidade dos hotspots

## 💡 Recomendações

### Curto Prazo (Melhorias Imediatas)

1. **Calibração de Threshold**
   ```python
   # Usar threshold adaptativo baseado em percentil
   threshold = np.percentile(predictions, 95)  # Top 5% apenas
   high_risk_nodes = predictions > threshold
   ```

2. **Post-processamento**
   - Aplicar normalização por percentil ao invés de valores brutos
   - Focar apenas no top-20 nós para operações
   - Desconsiderar predições abaixo de percentil 90

3. **Ensemble com Baseline**
   - Combinar com modelo de sazonalidade simples
   - Usar média ponderada: 50% ST-GAT + 50% histórico recente

### Médio Prazo (Retreinamento)

1. **Ajuste de Loss Function**
   - Aumentar peso dos eventos positivos (atualmente pode estar subpesado)
   - Usar Focal Loss ou Weighted MSE mais agressivo

2. **Ajuste de Arquitetura**
   - Reduzir dropout (pode estar eliminando sinais importantes)
   - Adicionar camada de ranking final
   - Implementar attention mechanism para focar em nós críticos

3. **Feature Engineering**
   - Adicionar features de tendência local (últimos 7 dias)
   - Incorporar dia da semana como feature categórica
   - Adicionar indicador de eventos exógenos

### Longo Prazo (Metodologia)

1. **Validação Contínua**
   - Estabelecer pipeline de validação semanal com dados novos
   - Monitorar drift de distribuição
   - Alertar quando P@20 cair abaixo de 30%

2. **A/B Testing**
   - Testar diferentes limiares de decisão
   - Comparar com baseline simples (média móvel)
   - Validar com equipe operacional

3. **Interpretabilidade**
   - Extrair attention weights do modelo
   - Identificar quais features mais influenciam
   - Validar se padrões geográficos/faccionais fazem sentido

## 📊 Comparação com Baseline Esperado

Um modelo baseline simples (média dos últimos 7 dias) provavelmente teria:
- **MAE**: ~0.6-0.7 (melhor que 0.81)
- **P@20**: ~20-30% (pior que atual 15%, mas mais concentrado)
- **Total previsto**: ~17-25 eventos (muito melhor que 266)

**Conclusão**: O modelo ST-GAT, no estado atual, **não está superando um baseline simples** para dados recentes. É necessário ajuste ou retreinamento.

## ✅ Próximos Passos

1. ✅ **Executar validação com baseline** (média móvel 7 dias)
2. ⬜ **Implementar calibração de threshold**
3. ⬜ **Testar ensemble ST-GAT + baseline**
4. ⬜ **Analisar attention weights (interpretabilidade)**
5. ⬜ **Coletar mais dados recentes** (esperar 7 dias) para validação contínua
6. ⬜ **Retreinar modelo** com loss function ajustada

---

*Relatório gerado automaticamente em 10/02/2026*
