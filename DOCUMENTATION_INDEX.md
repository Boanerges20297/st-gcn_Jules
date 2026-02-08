# 📚 ÍNDICE DE DOCUMENTAÇÃO - ARQUITETURA HÍBRIDA ST-GCN/ST-GAT

**Última Atualização**: 07 de Fevereiro de 2026  
**Fase**: Phase 3 - Implementação Completa ✅

---

## 🎯 DOCUMENTAÇÃO PRINCIPAL

### 1. **📖 HYBRID_ARCHITECTURE.md** - LEIA PRIMEIRO
**Local**: [`docs/HYBRID_ARCHITECTURE.md`](docs/HYBRID_ARCHITECTURE.md)

**Conteúdo:**
- Sumário executivo da arquitetura
- 6 camadas da arquitetura detalhadas
- Componentes implementados (stgat, dynamic_adjacency, mapper, fusion)
- Fluxo de integração passo-a-passo
- Exemplo de código para cada componente
- Métricas & monitoring
- Debugging & troubleshooting

**Quando ler**: Primeira visão geral técnica

---

### 2. **🏗️ PHASE3_ARCHITECTURE_REVIEW.md** - ROADMAP
**Local**: [`PHASE3_ARCHITECTURE_REVIEW.md`](PHASE3_ARCHITECTURE_REVIEW.md)

**Conteúdo:**
- Estado atual (Phase 2B) ✅
- O que precisa ser feito (Phase 3)
- Arquitetura alvo (híbrida ST-GCN/ST-GAT)
- Roadmap de conclusão (3 semanas)
- Ficheiros chave
- Próximas ações

**Quando ler**: Entender contexto e planejamento

---

### 3. **✅ PHASE3_IMPLEMENTATION_COMPLETE.md** - SUMÁRIO FINAL
**Local**: [`PHASE3_IMPLEMENTATION_COMPLETE.md`](PHASE3_IMPLEMENTATION_COMPLETE.md)

**Conteúdo:**
- Resumo do que foi realizado
- Detalhes de cada componente implementado
- Ficheiros criados/modificados
- Arquitetura final visual
- Benefícios implementados
- Próximos steps
- Checklist final
- Quick start guide

**Quando ler**: Ver o que foi entregue hoje

---

## 📝 DOCUMENTAÇÃO TÉCNICA (COMPONENTES)

### **src/stgat.py** (114 linhas - Refatorado)
**O que é**: Implementação de Graph Attention Network com temporal processing

**Classes:**
- `GraphAttentionLayer`: Single-head GAT com masking
- `MultiGraphAttention`: Multi-head aplicado a múltiplas adjacências
- `STGATLayer`: Bloco spatio-temporal (conv + GAT + norm)
- `STGAT`: Rede completa (2 layers + projection)

**Correções implementadas:**
✅ Input shape correto: (B, 26, 319, 30)
✅ Temporal convolution clara
✅ GAT aplicado por timestep
✅ Residual connections funcionando
✅ LayerNorm na dimensão correta

**Quick reference:**
```python
from src.stgat import STGAT
model = STGAT(num_nodes=319, in_channels=26, time_steps=30)
output = model(x, adj_list)  # (B, 319, 1)
```

---

### **src/dynamic_adjacency.py** (379 linhas - Novo)
**O que é**: Gerenciador de matriz de adjacência dinâmica com eventos

**Classe Principal:**
- `DynamicAdjacencyMatrix`: Aplica event-driven + temporal + decay

**Métodos:**
- `apply_event()`: Amplifica vizinhos de área com evento
- `apply_temporal_factors()`: Multipliers por hora/dia da semana
- `apply_decay()`: Exponential decay para eventos antigos
- `get_active_events()`: Lista eventos ativos com decay
- `export_state() / import_state()`: Serialização

**Quick reference:**
```python
from src.dynamic_adjacency import DynamicAdjacencyMatrix
dam = DynamicAdjacencyMatrix(original_adj, nodes_gdf, decay_hours=24)
dam.apply_event(node_idx=14, severity='HIGH', radius_km=3.0)
current_adj = dam.get_current_matrix()
```

---

### **src/architecture_mapper.py** (475 linhas - Novo)
**O que é**: Mapeia hierarquia de Bairros (polígonos) + Nós Virtuais (centróides)

**Classes:**
- `Node`: Representa um nó (real com polígono ou virtual com ponto)
- `ArchitectureMapper`: Gerencia mapeamento e atribuição espacial

**Métodos Principais:**
- `get_node_by_name() / by_idx()`: Lookup rápido
- `assign_occurrence_to_node()`: Atribui coordenada ao nó
- `assign_batch_occurrences()`: Batch assignment
- `get_neighbors()`: Encontra vizinhos por distância
- `export_to_geojson()`: Exporta para visualização
- `get_adjacency_matrix_geo()`: Constrói matrix geografia

**Quick reference:**
```python
from src.architecture_mapper import ArchitectureMapper
mapper = ArchitectureMapper(nodes_gdf, num_nodes=319)
node_idx, dist = mapper.assign_occurrence_to_node(lat=-3.5, lng=-38.5)
neighbors = mapper.get_neighbors(node_idx, distance_km=2.5)
geojson = mapper.export_to_geojson(include_risks=True)
```

---

### **src/model_fusion.py** (396 linhas - Novo)
**O que é**: Orquestrador de ensemble (ST-GCN + ST-GAT)

**Classes:**
- `ModelConfidenceTracker`: Rastreia accuracy histórica
- `AttentionFusionLayer`: Aprende ponderação dinâmica
- `ModelFusion`: Orquestrador principal

**Métodos Principais:**
- `predict()`: Faz predição com ensemble
- `train_fusion_layer()`: Treina parametricamente
- `get_ensemble_stats()`: Estatísticas do ensemble

**Estratégias de Fusion:**
- `'weighted_average'`: Rápido, baseado em histórico
- `'attention'`: Parametrizado, aprende dinamicamente

**Quick reference:**
```python
from src.model_fusion import ModelFusion
fusion = ModelFusion(stgcn, stgat, device='cpu', 
                    fusion_strategy='attention')
result = fusion.predict(x, adj_list, anomaly_flags=None)
# result['fusion']: scores finais
# result['confidence']: confiança [0, 1]
# result['weights']: pesos de fusion
```

---

## 📊 DOCUMENTAÇÃO AUXILIAR

### **docs/PHASE2B_FINAL_STATUS_REPORT.md** (545 linhas)
Relatório completo da Phase 2B com:
- Semanas 1-4 deliverables
- Métricas baseline
- Code & documentation created

---

### **docs/DYNAMIC_ADJACENCY_ANALYSIS.md** (371 linhas)
Análise teórica de adjacência dinâmica:
- O que é matriz dinâmica
- Viabilidade técnica
- Impacto computacional
- Alternativas e recomendações

---

### **docs/ARCHITECTURE_REFERENCE.md** (277 linhas)
Referência de arquitetura (Phase 1):
- Quick start
- Directory structure
- Data flow
- ST-GCN vs Ranking Model comparação

---

## 🎯 COMO USAR ESTA DOCUMENTAÇÃO

### **Para Entender a Arquitetura**
1. Leia: [`PHASE3_ARCHITECTURE_REVIEW.md`](PHASE3_ARCHITECTURE_REVIEW.md) (contexto)
2. Leia: [`docs/HYBRID_ARCHITECTURE.md`](docs/HYBRID_ARCHITECTURE.md) (detalhes técnicos)
3. Consulte: Seção de "Componentes Implementados" acima

### **Para Implementar/Integrar**
1. Veja: Quick reference de cada componente acima
2. Consulte: Exemplos em [`docs/HYBRID_ARCHITECTURE.md`](docs/HYBRID_ARCHITECTURE.md) - seção "FLUXO DE INTEGRAÇÃO"
3. Use: [`PHASE3_IMPLEMENTATION_COMPLETE.md`](PHASE3_IMPLEMENTATION_COMPLETE.md) - "Quick Start"

### **Para Debugar Problemas**
1. Vá para: [`docs/HYBRID_ARCHITECTURE.md`](docs/HYBRID_ARCHITECTURE.md) - seção "DEBUGGING & TROUBLESHOOTING"
2. Procure seu erro específico
3. Siga as soluções propostas

### **Para Monitorar Performance**
1. Veja: [`docs/HYBRID_ARCHITECTURE.md`](docs/HYBRID_ARCHITECTURE.md) - seção "MÉTRICAS & MONITORING"
2. Use: `fusion.get_ensemble_stats()` para estatísticas em tempo real
3. Implemente: Logging conforme exemplos dados

---

## 📁 FICHEIROS CRIADOS

```
✅ src/stgat.py                         [114 linhas - refatorado]
✅ src/dynamic_adjacency.py             [379 linhas - novo]
✅ src/architecture_mapper.py           [475 linhas - novo]
✅ src/model_fusion.py                  [396 linhas - novo]

✅ docs/HYBRID_ARCHITECTURE.md          [350+ linhas - novo]

✅ PHASE3_ARCHITECTURE_REVIEW.md        [200 linhas - novo]
✅ PHASE3_IMPLEMENTATION_COMPLETE.md    [320 linhas - novo]

Esta Documentação:
✅ DOCUMENTATION_INDEX.md               [este arquivo]

TOTAL: ~1400 linhas de código novo + ~900 linhas de documentação
```

---

## 🔗 LINKS RÁPIDOS

| Documento | Tamanho | Tempo Leitura |
|-----------|---------|--------------|
| [HYBRID_ARCHITECTURE.md](docs/HYBRID_ARCHITECTURE.md) | 350+ linhas | 15-20 min |
| [PHASE3_ARCHITECTURE_REVIEW.md](PHASE3_ARCHITECTURE_REVIEW.md) | 200 linhas | 10 min |
| [PHASE3_IMPLEMENTATION_COMPLETE.md](PHASE3_IMPLEMENTATION_COMPLETE.md) | 320 linhas | 15 min |
| [PHASE2B_FINAL_STATUS_REPORT.md](PHASE2B_FINAL_STATUS_REPORT.md) | 545 linhas | 20 min |
| [DYNAMIC_ADJACENCY_ANALYSIS.md](docs/DYNAMIC_ADJACENCY_ANALYSIS.md) | 371 linhas | 15 min |

---

## 📞 COMO COMEÇAR

### **Option 1: Quick Overview (5 minutos)**
1. Leia sumário executivo de [`PHASE3_IMPLEMENTATION_COMPLETE.md`](PHASE3_IMPLEMENTATION_COMPLETE.md)
2. Veja diagrama de arquitetura final

### **Option 2: Technical Deep Dive (30 minutos)**
1. Leia [`PHASE3_ARCHITECTURE_REVIEW.md`](PHASE3_ARCHITECTURE_REVIEW.md)
2. Estude [`docs/HYBRID_ARCHITECTURE.md`](docs/HYBRID_ARCHITECTURE.md)
3. Consulte quick references dos componentes

### **Option 3: Implementation Ready (1 hora)**
1. Leia toda a documentação acima
2. Copie exemplos de "Quick Start"
3. Teste cada componente isoladamente
4. Integre seguindo "FLUXO DE INTEGRAÇÃO"

---

## ✅ PRÓXIMOS STEPS

- [x] Arquitetura híbrida implementada
- [x] Documentação técnica completa
- [ ] Phase 3.2: Training & Integration Tests
- [ ] Phase 3.3: API Endpoints & Dashboard
- [ ] Phase 3.4: Production Deployment

---

**Última atualização**: 07 de Fevereiro de 2026  
**Status**: ✅ Phase 3 - Implementação Completa  
**Próxima fase**: Treinamento & Testes (previsto para 10-14 de Fevereiro)

🎉 **Bem-vindo à Era da Crime Prediction Inteligente!**
