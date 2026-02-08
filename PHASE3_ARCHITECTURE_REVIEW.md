# 🏗️ FASE 3: REVISÃO ARQUITETURA HÍBRIDA ST-GCN/ST-GAT

**Data**: 07 de Fevereiro de 2026  
**Status**: Phase 2B Completo ✅ | Phase 3 Iniciando 🚀  
**Objetivo**: Integrar arquitetura híbrida (Bairros + Nós Dinâmicos Virtuais)

---

## 📊 RESUMO EXECUTIVO

### Estado Atual (Phase 2B ✅)
```
✅ Semana 1-4 Completa:
   • Métricas (P@K, NDCG@K, Recall@K)
   • Integração Eventos Exógenos
   • Otimização Long-tail (P@20)
   • Explainability Layer (LLM-based)

✅ Modelos em Produção:
   • ST-GCN v2 (26 canais | 319 nós | 30-day window)
   • ST-GAT v1 (basic, precisa integração)
   • Ranking Model (validador de top-5)
   • Enhanced Model (com anomaly awareness)

✅ Infraestrutura:
   • API Flask (<200ms latência)
   • Dashboard com visualização híbrida (polígonos + pontos)
   • Event Anomaly Detector (14+ eventos)
   • Explanation Generator (46 padrões heurísticos)

✅ Métricas Baseline:
   • P@5 = 0.80 (alvo: ≥0.78)
   • P@20 = 0.50→0.55 (otimizado)
   • NDCG@20 = 0.75 (alvo: ≥0.76)
   • Confiança Top-5: 100%
```

---

## 🎯 O QUE PRÉCISA SER FEITO (Phase 3)

### ⚙️ Arquitetura Alvo: Híbrida ST-GCN/ST-GAT

```
INPUT LAYER (26 canais)
   ├─ CVLI (homicídios)
   ├─ CVP (crimes patrimônio)
   ├─ Tension (estática)
   ├─ One-hot (dia semana, mês)
   └─ Exogenous Flags

   ↓ (NOVO)
────────────────────────────────

SPATIAL MODELING LAYER (Híbrido)
   ├─ ST-GCN (Convolução clássica)
   │  └─ Estático, rápido, baseline
   │
   └─ ST-GAT (Atenção - NOVO)
      ├─ Aprende pesos de adjacência
      ├─ Dinâmico por dia/contexto
      └─ Multi-head attention

FUSION LAYER (NOVO)
   ├─ Combina outputs: GCN + GAT
   │  └─ Weighted average ou concatenação
   └─ Adaptativo por confiança

   ↓
────────────────────────────────

TEMPORAL MODELING (Melhorado)
   ├─ Convolução 1D (baseline)
   ├─ Self-Attention Temporal (NOVO)
   └─ Pattern Recognition

   ↓
────────────────────────────────

DYNAMIC ADJACENCY (NOVO)
   ├─ Event-driven multipliers
   │  └─ Amplifica vizinhos afetados
   ├─ Temporal factors
   │  └─ Reduz pesos em horas silenciosas
   └─ Decaying influence

   ↓
────────────────────────────────

OUTPUT & POST-PROCESSING
   ├─ Node Rankings (319 nós)
   ├─ Confidence Scores (anomaly-adjusted)
   ├─ Explanations (por fator)
   ├─ Metrics (P@5-20, NDCG@K)
   └─ API JSON
```

---

## 📋 COMPONENTES PARA INTEGRAÇÃO

### 1. ✅ Arquivo Existente: `src/stgat.py`

**Status**: Básico, pronto para melhorias
- GraphAttentionLayer (OK)
- MultiGraphAttention (OK)
- STGATLayer (precisa debug)
- STGAT main (skeleton)

**Problemas encontrados**:
```python
# ATUAL (linha 77):
x_in = x.unsqueeze(2)  # (N, C, 1, T)
# ISSUE: Assume N=batch, mas N=num_nodes!
# CORREÇÃO: Reshape apropriado para tratar nodes como features
```

**Ação**: Refatorar para receber (batch, channels, nodes, time)

---

### 2. 🆕 ArchitectureMapper: Polígonos + Nós Dinâmicos

**Status**: Parcialmente implementado em app.py (linhas 1413-1460)

**Componentes Existentes**:
- ✅ `/api/polygons` - retorna polígonos + pontos dinâmicos
- ✅ `nodes_gdf` - GeoDataFrame com geometrias
- ✅ `nodes_centroids_proj` - centróides para K-tree

**Falta**:
- Mapeamento explícito: Bairro (polígono) → Nós Dinâmicos (pontos internos)
- Atribuição de features: Ocorrência → Polígono → Nó Dinâmico
- Propagação de risco: Polígono → Vizinhos Virtuais

**Ação**: Criar classe `ArchitectureMapper`

---

### 3. 🆕 DynamicAdjacencyManager

**Status**: Proposta em `docs/DYNAMIC_ADJACENCY_ANALYSIS.md` (viável ✅)

**Implementação Recomendada**:
```python
class DynamicAdjacencyMatrix:
    """
    Gerencia matriz de adjacência com:
    • Event-driven changes (+severidade)
    • Temporal multipliers (hora, semana)
    • Decaying influence (t > 24h)
    """
    
    def __init__(self, original_adj, nodes_gdf):
        self.base = original_adj.copy()
        self.current = original_adj.copy()
        self.nodes = nodes_gdf
    
    def apply_event(self, event, severity, radius_m):
        """Amplifica vizinhos do evento"""
        # factor = {HIGH: 1.3, MEDIUM: 1.15, LOW: 1.05}
        
    def apply_temporal_factors(self, hour, day_of_week):
        """Reduz pesos em horas silenciosas"""
        
    def apply_decay(self, hours_since_event):
        """Decai influência exponencialmente"""
        
    def get_normalized(self):
        """Retorna matriz normalizada via random walk"""
```

**Ação**: Implementar com event log persistente

---

### 4. 🔄 Model Ensemble & Fusion

**Status**: Código pronto em `app.py` (linhas 1199-1240)

**Modelos Carregados**:
- ✅ `model_cvli` (ST-GCN v2 - 26 canais)
- ✅ `model_stgat` (ST-GAT v1 - em produção mas básico)
- ✅ `ranking_validator` (Validation Model)

**Falta**: Lógica de fusion

```python
# PSEUDOCÓDIGO
def predict_hybrid(x, adj_list, anomaly_flags):
    # ST-GCN: scores_gcn = model_cvli(x, adj_list)
    # ST-GAT: scores_gat = model_stgat(x, adj_list, learn_attention=True)
    
    # Fusion (NOVO):
    # confidence_gcn = reliability_score(scores_gcn)  # histórico
    # confidence_gat = reliability_score(scores_gat)  # dinâmico
    
    # weights = softmax([confidence_gcn, confidence_gat])
    # scores = weights[0] * scores_gcn + weights[1] * scores_gat
    
    # Anomaly adjustment (EXISTENTE):
    # scores *= (1 - anomaly_level * 0.3)
    
    return scores
```

**Ação**: Implementar fusion logic em `src/model_fusion.py`

---

## 📈 ROADMAP DE CONCLUSÃO (Phase 3)

### Semana 1: Foundação & Debugging
```
[ ] 1.1: Debug ST-GAT architecture (src/stgat.py)
   └─ Reshape input: (batch, 26, 319, 30) → (batch, 319, 26, 30)
   └─ Testar forward pass
   └─ Validar outputs

[ ] 1.2: Refatorar Data Processing
   └─ Explicit mapping: Bairro (polygon) → Virtual Nodes
   └─ Feature assignment via polygon intersection
   └─ Criar ArchitectureMapper class

[ ] 1.3: Implement DynamicAdjacencyManager
   └─ Event log persistence
   └─ Real-time matrix updates
   └─ Decay mechanism
```

### Semana 2: Integration & Testing
```
[ ] 2.1: Implement Model Fusion
   └─ Load GCN + GAT
   └─ Confidence weighting scheme
   └─ Anomaly-adjusted outputs
   └─ Unit tests

[ ] 2.2: Extend API Endpoints
   └─ /predict/hybrid (fusion outputs)
   └─ /predict/components (GCN vs GAT breakdown)
   └─ /adjacency/status (event-driven state)

[ ] 2.3: Dashboard Updates
   └─ Visualize polygon coloring
   └─ Dynamic node indicators
   └─ Confidence visualization
```

### Semana 3: Production Validation
```
[ ] 3.1: Training Pipeline
   └─ Train ST-GAT on full data (2 epochs)
   └─ Train fusion layer (5 epochs)
   └─ A/B test: GCN vs GAT vs Fusion

[ ] 3.2: Performance Benchmarking
   └─ Latency benchmarks
   └─ Memory profiling
   └─ Accuracy metrics (P@5-20)

[ ] 3.3: Production Deployment
   └─ Model versioning
   └─ Rollback strategy
   └─ Monitoring setup
```

---

## 📊 MÉTRICAS DE SUCESSO

### Phase 2B Baseline
```
P@5:   0.80
P@20:  0.55
NDCG@5:  0.92
NDCG@20: 0.77
Latency: <200ms
CPU: Suporta
```

### Phase 3 Targets (30 dias)
```
P@5:   0.82+ (GAT adaptativo)
P@20:  0.60+ (dinâmica melhorada)
NDCG@5:  0.94+ (fusion benefits)
NDCG@20: 0.80+ (melhor cobertura)
Latency: <250ms (acceptable)
CPU: Still supported
```

---

## 🔧 FICHEIROS CHAVE

| Arquivo | Status | Ação |
|---------|--------|------|
| `src/stgat.py` | Básico | Refatorar input shapes + debug forward |
| `src/model_fusion.py` | 🆕 | Criar com fusion logic |
| `src/dynamic_adjacency.py` | 🆕 | Criar com event handling |
| `app.py` | Parcial | Integrar endpoints novos |
| `templates/index.html` | Parcial | Visualização dinâmica de confiança |
| `docs/HYBRID_ARCHITECTURE.md` | 🆕 | Documento técnico detalhado |

---

## ✅ PRÓXIMAS AÇÕES (HOJE)

1. **Debugar ST-GAT** (2h)
   - Testar shapes de input/output
   - Validar forward pass
   
2. **Criar Architecture Mapper** (2h)
   - Mapeamento Bairro → Nós
   - Feature assignment
   
3. **Implement Dynamic Adjacency** (2h)
   - Event log
   - Matrix updates
   
4. **Criar documento técnico** (1h)
   - Detalhar arquitetura híbrida
   - Exemplos de uso

---

## 📚 DOCUMENTAÇÃO RELACIONADA

- ✅ [PHASE2B_FINAL_STATUS_REPORT.md](PHASE2B_FINAL_STATUS_REPORT.md)
- ✅ [DYNAMIC_ADJACENCY_ANALYSIS.md](docs/DYNAMIC_ADJACENCY_ANALYSIS.md)
- ✅ [docs/ARCHITECTURE_REFERENCE.md](docs/ARCHITECTURE_REFERENCE.md)
- 🆕 HYBRID_ARCHITECTURE.md (criar hoje)
- 🆕 TRAINING_GUIDE_PHASE3.md (criar hoje)

---

**Ready to proceed with Phase 3! 🚀**
