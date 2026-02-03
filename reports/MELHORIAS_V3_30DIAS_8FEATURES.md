# Relatório de Melhorias - ST-GCN v3

**Data**: 02/02/2026  
**Sessão**: Implementação de melhorias sem ensemble  
**Status**: ✅ Concluído

---

## 🎯 Objetivo

Melhorar a acurácia do modelo ST-GCN além de P@5 = 14.33%, implementando:
1. ✅ Aumentar janela temporal para 30 dias
2. ❌ ~~Ensemble de modelos~~ (RETRATADO pelo usuário)
3. ✅ Adicionar features temporais (dia da semana, mês)
4. ✅ Documentar diferenças entre arquiteturas (ST-GCN vs LSTM vs Transformer)

---

## 🔧 Mudanças Implementadas

### 1. Janela Temporal: 14 → 30 Dias

**Arquivo**: [src/train.py](src/train.py#L23)

```python
HISTORY_WINDOW = 30  # 30 dias - captura padrões mensais e sazonais
```

**Justificativa**:
- Crimes violentos têm padrões mensais (dia do pagamento, ciclos de disputa territorial)
- Janela de 14 dias capturava apenas padrões quinzenais
- 30 dias permite detectar sazonalidade (início/fim de mês)

---

### 2. Features Temporais: 3 → 8 Canais

**Arquivo**: [src/data_processing.py](src/data_processing.py#L361-L377)

**Canais Implementados**:

| Canal | Feature | Tipo | Descrição |
|-------|---------|------|-----------|
| 0 | CVLI | Numérico | Homicídios dolosos (0-8 eventos/dia) |
| 1 | CVP | Numérico | Crimes contra patrimônio (0-9 eventos/dia) |
| 2 | Tension Index | Numérico | Índice de tensão (rivalidade de facções) |
| 3 | Day of Week (sin) | Cíclico | Padrão semanal (segunda=0, domingo=6) |
| 4 | Day of Week (cos) | Cíclico | Padrão semanal (complemento) |
| 5 | Month (sin) | Cíclico | Padrão mensal (janeiro=1, dezembro=12) |
| 6 | Month (cos) | Cíclico | Padrão mensal (complemento) |
| 7 | Is Weekend | Binário | Fim de semana (sábado/domingo = 1) |

**Encoding Cíclico (sin/cos)**:

```python
# Dia da semana: 0 (segunda) → 6 (domingo)
day_of_week = dates.dayofweek
features[:, :, 3] = np.sin(2 * np.pi * day_of_week / 7)  # sin
features[:, :, 4] = np.cos(2 * np.pi * day_of_week / 7)  # cos

# Mês: 1 (janeiro) → 12 (dezembro)
month = dates.month
features[:, :, 5] = np.sin(2 * np.pi * month / 12)  # sin
features[:, :, 6] = np.cos(2 * np.pi * month / 12)  # cos

# Fim de semana: sábado (5) e domingo (6)
features[:, :, 7] = (day_of_week >= 5).astype(np.float32)
```

**Por que sin/cos?**
- Evita descontinuidade (domingo não é "maior" que segunda)
- Preserva distância (segunda está perto de terça E domingo)
- Permite ao modelo aprender padrões circulares

---

### 3. Regeneração do Pipeline de Dados

**Problema Identificado**:
- Arquivo pickle antigo (`processed_graph_data.pkl`) tinha apenas 3 canais
- Modelo estava ignorando as 5 features temporais já implementadas no código

**Solução**:
- Executado `python src/data_processing.py` para regenerar pickle
- Shape atualizado: `(319 nodes, 1491 days, **8 channels**)`

**Antes**:
```
Treino: (1168, 3, 319, 30) ❌
```

**Depois**:
```
Treino: (1168, 8, 319, 30) ✅
```

---

### 4. Documentação de Arquiteturas

**Arquivo**: [ARQUITETURAS_COMPARACAO.md](ARQUITETURAS_COMPARACAO.md)

Criado documento comparativo detalhado:
- **ST-GCN** (atual): Graph Convolutions + Temporal Convolutions
- **LSTM**: Memória recorrente para séries temporais
- **Transformer**: Self-attention para padrões globais

**Recomendação**: Manter ST-GCN
- ✅ Estrutura espacial rica (2 grafos: geográfico + facções)
- ✅ Dataset moderado (74k ocorrências é suficiente)
- ✅ Treinamento em CPU (~20 min)
- ✅ Interpretabilidade operacional (polícia precisa entender motivos)

**Melhorias Futuras Sugeridas**:
- ST-GAT: Graph Attention + Temporal Attention
- Permite aprender pesos dinâmicos para adjacências
- Custo computacional moderado

---

## 📊 Resultados Esperados

### Baseline (Antes das Melhorias)
- Janela: 14 dias
- Features: 3 canais (CVLI, CVP, Tension)
- **P@5**: 14.33%

### Após Melhorias (Em Treinamento)
- Janela: 30 dias
- Features: 8 canais (3 base + 5 temporais)
- **P@5**: ⏳ Aguardando resultados (~20 min)

### Meta de Sucesso
- **P@5 ≥ 18%**: Melhoria significativa (+25%)
- **P@5 ≥ 20%**: Excelente (+40% - viável para produção)
- **P@5 < 15%**: Investigar overfitting ou problema de encoding

---

## 🐛 Problemas Resolvidos

### Problema 1: Arquivo pickle desatualizado
**Erro**: 
```
Treino: (1168, 3, 319, 30)  # Apenas 3 canais!
```

**Causa**: `processed_graph_data.pkl` gerado antes das features temporais serem implementadas

**Solução**: 
```bash
python src/data_processing.py  # Regenerar com 8 canais
```

**Resultado**: `(319, 1491, 8)` ✅

---

### Problema 2: Features temporais não utilizadas
**Descoberta**: 
- Código em `build_feature_tensor` já criava 8 canais
- Comentário dizia "3 canais" mas código implementava 8
- Modelo aceitava `in_channels=num_features` (dinâmico)

**Solução**: Apenas regenerar pickle (modelo já estava preparado!)

---

## 📁 Arquivos Modificados

```
src/
  train.py              # HISTORY_WINDOW = 30
  data_processing.py    # build_feature_tensor com 8 canais

data/processed/
  processed_graph_data.pkl  # Regenerado com (319, 1491, 8)

docs/ (novos)
  ARQUITETURAS_COMPARACAO.md  # Comparativo ST-GCN/LSTM/Transformer
```

---

## 🔄 Próximos Passos

### Após Treinamento Concluir
1. **Avaliar P@5**: 
   - Se ≥ 18%: Deploy em produção
   - Se < 15%: Investigar features temporais (podem estar introduzindo ruído)

2. **Testar Previsões**:
   ```bash
   python app.py  # Verificar predições para 03/02/2026
   ```

3. **Validação Operacional**:
   - Comparar top 16 bairros previstos com ocorrências reais do dia seguinte
   - Calcular Precision@5 real (não apenas validação)

### Melhorias Futuras (Se P@5 < 20%)
1. **ST-GAT (Graph Attention)**:
   - Substituir GCN por GAT para aprender pesos de adjacência
   - Código: `src/model_gat.py`

2. **Temporal Attention**:
   - Substituir convolução 1D por self-attention
   - Captura dependências de longo prazo (>30 dias)

3. **Features Exógenas Avançadas**:
   - Clima (temperatura, chuva)
   - Eventos prisionais (fugas, rebeliões)
   - Operações policiais

4. **Loss Function Alternativa**:
   - Ranking Loss (em vez de MSE)
   - Foca em ordenação correta (top-k) em vez de valores absolutos

---

## 📝 Lições Aprendidas

1. **Sempre verificar shape dos tensores**:
   - Pipeline pode estar implementado mas pickle desatualizado
   - Regenerar dados após mudanças estruturais

2. **Comentários podem estar desatualizados**:
   - Código dizia "3 canais" mas implementava 8
   - Sempre validar no código, não nos comentários

3. **Features cíclicas (sin/cos) são essenciais**:
   - Dia da semana: segunda está perto de terça E domingo
   - Mês: dezembro está perto de janeiro

4. **Janela temporal maior ≠ sempre melhor**:
   - 7 dias: muito curto (apenas padrões semanais)
   - 30 dias: bom equilíbrio (mensal + sazonal)
   - 90 dias: pode diluir padrões recentes

---

**Autor**: GitHub Copilot (Claude Sonnet 4.5)  
**Revisão**: Boanerges  
**Próxima Atualização**: Após conclusão do treinamento (~21:50)
