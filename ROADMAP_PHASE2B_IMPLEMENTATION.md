# 🏗️ ARQUITETURA + ROADMAP DE IMPLEMENTAÇÃO

**Data**: 06/02/2026  
**Foco**: B (Cobertura P@20) + C (Estabilidade Eventos) + D (Explainability)  
**Timeline**: 5 semanas (Feb 7 - Mar 13, 2026)

---

## 📐 VISÃO ARQUITETÔNICA COMPLETA

### Arquitetura Atual (Production Ready)

```
┌─────────────────────────────────────────────────────────────┐
│                     FORTALEZA CRIME RISK SYSTEM              │
│                                                              │
│  ┌─────────────┐      ┌──────────────┐    ┌─────────────┐  │
│  │   ST-GCN    │─────▶│ RankingModel │───▶│  API/Flask  │  │
│  │  (Temporal)  │      │  (Top-5 only)│    │  Dashboard  │  │
│  └─────────────┘      └──────────────┘    └─────────────┘  │
│        ▲                     ▲                    ▲          │
│        │                     │                    │          │
│  26 Features         Pairwise Loss          P@5 Metric     │
│  • CVLI              • Ranking loss         • JSON API      │
│  • Temporal          • No explanation       • HTML viz     │
│  • Spatial           │                       │             │
│  • Exogenous         └──────────────────────┘             │
│                                                              │
│  GAPS:                                                       │
│  ❌ Não cobre long-tail (nodes 20-100)                   │
│  ❌ Frágil a eventos exógenos (sem anomaly detection)     │
│  ❌ Black-box (nenhuma explicação)                        │
│  ❌ Métricas limitadas (só P@5)                           │
└─────────────────────────────────────────────────────────────┘
```

### Arquitetura Proposta (Enhanced)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENHANCED CRIME RISK SYSTEM v2                        │
│                                                                         │
│  INPUT LAYER                                                            │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ • CVLI data (26D)                                            │      │
│  │ • Exogenous events (from JSON)                              │      │
│  │ • Historical patterns (seasonal)                            │      │
│  └──────────────────────────────────────────────────────────────┘      │
│         │                                          │                   │
│         ▼                                          ▼                   │
│  ┌──────────────────────┐            ┌──────────────────────────┐    │
│  │    ST-GCN Module     │            │  Event Anomaly Detector  │    │
│  │  (Temporal + Spatial)│            │  (LLM-based)            │    │
│  │                      │            │                          │    │
│  │ Output: Node scores  │            │ • Parse event text      │    │
│  │ (all 319 nodes)      │            │ • Severity classification│    │
│  │                      │            │ • Anomaly flag          │    │
│  └──────────────────────┘            │ • Affected node list    │    │
│         │                             │                         │    │
│         ▼                             ▼                         │    │
│  ┌────────────────────────────────────────────────┐              │    │
│  │  RANKING MODEL (Enhanced)                     │              │    │
│  │                                                │              │    │
│  │  Input: Node scores + event context (66D)    │              │    │
│  │  Output: Ranking for all 319 nodes           │              │    │
│  │                                                │              │    │
│  │  Loss: Pairwise + anomaly weighting          │              │    │
│  │  Metrics: P@5, P@10, P@20, NDCG@5-20        │              │    │
│  └────────────────────────────────────────────────┘              │    │
│         │              │               │                         │    │
│         ▼              ▼               ▼                         │    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │ Metric   │  │ Explanations │  │ Confidence   │               │    │
│  │ Reporter │  │ (LLM layer)  │  │ Adjuster     │               │    │
│  │          │  │              │  │ (event-based)│               │    │
│  │ • P@5-20 │  │ • Why top-5? │  │ • Reduce if  │               │    │
│  │ • NDCG   │  │ • Why node X │  │   anomaly    │               │    │
│  │ • Recall │  │   in top-20? │  │ • Uncertainty│               │    │
│  └──────────┘  └──────────────┘  └──────────────┘               │    │
│         │              │               │                         │    │
│         └──────────────┴───────────────┘                         │    │
│                      │                                           │    │
│                      ▼                                           │    │
│         ┌─────────────────────────────┐                         │    │
│         │  API + Dashboard (Enhanced) │                         │    │
│         │                             │                         │    │
│         │ • /predict (all nodes)      │                         │    │
│         │ • /metrics (P@5-20, NDCG)  │                         │    │
│         │ • /explain (why each node)  │                         │    │
│         │ • /anomaly_alert (events)   │                         │    │
│         │ • /confidence (anomaly?)    │                         │    │
│         └─────────────────────────────┘                         │    │
└─────────────────────────────────────────────────────────────────────────┘

KEY IMPROVEMENTS:
✅ B: P@20 coverage (all nodes, not just top-5)
✅ C: Event anomaly detection (stability)
✅ D: Explanations (why each node is ranked where)
```

---

## 📊 COMPONENTES NOVOS

### 1️⃣ Event Anomaly Detector (Novo)
```python
class EventAnomalyDetector:
    """
    Integrates exogenous events into model confidence
    
    Input: Event text
    Output: {
        severity: float (0-1),
        affected_nodes: list,
        anomaly_flag: bool,
        confidence_reduction: float (0-1)
    }
    """
    
    def parse_event(event_text):
        # Use simple LLM prompt (Gemini FREE tier if possible)
        # Or use heuristics for MVP
        pass
    
    def detect_anomaly():
        # Event happened today → reduce model confidence
        # Warn: "Model is less reliable due to event in X"
        pass
```

### 2️⃣ Enhanced RankingModel (Modified)
```python
class EnhancedRankingModel(nn.Module):
    """
    Drop-in replacement for current RankingModel
    
    Additions:
    • Input: Include event anomaly flag
    • Output: Confidence score (uncertainty)
    • Loss: Pairwise + anomaly weighting (harder to match top-5 if event)
    """
    
    def forward(x, anomaly_flags):
        # Same as before + anomaly awareness
        # Reduces confidence when event = True
        pass
```

### 3️⃣ Explanation Layer (Novo)
```python
class ExplanationGenerator:
    """
    LLM-based: "Explain why node X is in top-K"
    
    Prompt template:
    "Node {node_id} is ranked {rank} with score {score}.
     Context: {temporal_pattern}, {nearby_nodes}, {events}.
     Explain concisely why."
    
    Output: Human-readable explanation
    """
    
    def explain_ranking(node_id, rank, context):
        pass
```

### 4️⃣ Metric Reporter (Novo)
```python
class MetricReporter:
    """
    Calculates P@5, P@10, P@20, NDCG@5-20
    Reports per-window metrics
    Tracks improvement over time
    """
    
    def calculate_all_metrics(y_true, y_pred):
        # P@K for k in [5, 10, 15, 20]
        # NDCG@K for k in [5, 10, 20]
        # Recall@20
        # Ranking correlation
        pass
```

---

## 📅 ROADMAP DE IMPLEMENTAÇÃO (5 SEMANAS)

### SEMANA 1: Fundação (Feb 7-13)

**Objetivo**: Validar arquitetura + Setup metrics

#### Task 1.1: Métricas Adicionais (8h)
```python
# File: src/metrics.py (NEW)

def precision_at_k(y_true, y_pred, k):
    """P@K for k in 5, 10, 20"""
    real_top_k = set(np.argsort(-y_true)[:k])
    pred_top_k = set(np.argsort(-y_pred)[:k])
    overlap = len(real_top_k & pred_top_k)
    return overlap / k

def ndcg_at_k(y_true, y_pred, k):
    """Normalized Discounted Cumulative Gain"""
    # DCG = sum(rel_i / log2(i+1))
    # NDCG = DCG / ideal_DCG
    pass

def recall_at_k(y_true, y_pred, k):
    """Recall@K: How many actual top-20 do we find?"""
    pass

class MetricReporter:
    def report(y_true, y_pred):
        return {
            'p_at_5': precision_at_k(y_true, y_pred, 5),
            'p_at_10': precision_at_k(y_true, y_pred, 10),
            'p_at_20': precision_at_k(y_true, y_pred, 20),
            'ndcg_at_5': ndcg_at_k(y_true, y_pred, 5),
            'ndcg_at_10': ndcg_at_k(y_true, y_pred, 10),
            'ndcg_at_20': ndcg_at_k(y_true, y_pred, 20),
            'recall_at_20': recall_at_k(y_true, y_pred, 20)
        }
```

**Deliverable**: `src/metrics.py` (novo arquivo)

#### Task 1.2: Baseline Evaluation (4h)
```python
# File: scripts/evaluate_baseline_metrics.py (NEW)

# Load production model
# Load test data (últimos 60 dias)
# Evaluate: P@5-20, NDCG@5-20
# Save baseline_metrics.json

# Expected output:
# {
#   "p_at_5": 0.80,
#   "p_at_10": 0.65,
#   "p_at_20": 0.50,
#   "ndcg_at_5": 0.92,
#   "ndcg_at_10": 0.88,
#   "ndcg_at_20": 0.75,
#   "recall_at_20": 0.45
# }
```

**Deliverable**: `baseline_metrics.json` (referência)

#### Task 1.3: Análise Long-Tail (4h)
```python
# File: analysis/long_tail_analysis.py (NEW)

# Pergunta: Quais nodes deveriam estar em top-20 mas não estão?

nodes_in_top20_real = top-20 nodes by actual CVLI
nodes_in_top20_pred = top-20 nodes by model prediction

missed = nodes_in_top20_real - nodes_in_top20_pred
print(f"Missed {len(missed)} nodes in top-20")
print(f"Examples: {missed[:10]}")

# For each missed node:
# - Qual é o rank verdadeiro?
# - Qual é o rank predito?
# - Por quê? (temporal? spatial?)
```

**Deliverable**: `long_tail_analysis.json`

#### ✅ Check-in Week 1:
- [ ] Metrics.py funciona
- [ ] Baseline metrics calculadas (P@5-20)
- [ ] Long-tail analysis pronta
- [ ] Status: Pronto para Week 2

---

### SEMANA 2: Event Integration (Feb 14-20)

**Objetivo**: Integrar eventos exógenos + anomaly detection básico

#### Task 2.1: Event Parsing Heurístico (6h)
```python
# File: src/event_anomaly_detector.py (NEW)

class EventAnomalyDetector:
    """
    Parse eventos SEM LLM (MVP - usar heurísticas)
    
    Later: upgrade para real LLM se necessário
    """
    
    def __init__(self):
        self.severity_keywords = {
            'homicídio|morte|corpo': 1.0,
            'roubo|assalto|armado': 0.8,
            'tráfico|droga': 0.7,
            'briga|agressão': 0.5,
            'furto': 0.3
        }
    
    def parse_event(event_text: str) -> dict:
        """
        Heurístico: keyword matching
        
        Output:
        {
            'severity': float 0-1,
            'crime_types': list,
            'anomaly_flag': bool,  # True if severity > 0.6
            'confidence_reduction': float,  # How much to reduce model confidence
        }
        """
        # Keyword matching
        severity = 0.0
        for keywords, score in self.severity_keywords.items():
            if any(kw in event_text.lower() for kw in keywords.split('|')):
                severity = max(severity, score)
        
        return {
            'severity': severity,
            'anomaly_flag': severity > 0.6,
            'confidence_reduction': severity * 0.3  # Reduz 30% de confiança se evento high-severity
        }
```

**Deliverable**: `src/event_anomaly_detector.py`

#### Task 2.2: Load & Index Events (4h)
```python
# File: src/event_manager.py (NEW)

class EventManager:
    """
    Gerencia eventos exógenos actuais
    """
    
    def __init__(self):
        with open('data/exogenous_events_geocoded.json') as f:
            self.events = json.load(f)
    
    def get_events_for_date(date):
        """Retorna eventos para uma data específica"""
        return [e for e in self.events if e['date'] == date]
    
    def get_anomaly_level_for_date(date):
        """
        Calcula nível de anomalia para data
        
        Output: float 0-1 (0 = sem eventos, 1 = evento crítico)
        """
        events = self.get_events_for_date(date)
        detector = EventAnomalyDetector()
        
        severities = [detector.parse_event(e['text'])['severity'] for e in events]
        return max(severities) if severities else 0.0
```

**Deliverable**: `src/event_manager.py`

#### Task 2.3: Modificar RankingModel (6h)
```python
# File: src/ranking_model.py (MODIFY existing)

class EnhancedRankingModel(nn.Module):
    """
    Drop-in replacement para RankingModel actual
    
    Adições:
    • Input anomaly_level (float 0-1)
    • Output confidence score (em cima do ranking score)
    """
    
    def __init__(self, input_dim=26, hidden_dim=128, output_dim=1):
        super().__init__()
        # Same as before
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, anomaly_level=0.0):
        """
        x: (batch_size, seq_len, input_dim)
        anomaly_level: float 0-1
        
        Returns: (scores, confidence)
        """
        # Same forward pass as before
        x_flat = x.mean(dim=1)  # (batch_size, input_dim)
        h = F.relu(self.fc1(x_flat))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        scores = self.fc3(h)  # (batch_size, 1)
        
        # Confidence: reduce if anomaly
        confidence = 1.0 - (anomaly_level * 0.3)  # Max 30% reduction
        
        return scores, confidence
```

**Deliverable**: Modified RankingModel

#### Task 2.4: Treinar com Anomaly Awareness (8h)
```python
# File: scripts/train_with_anomaly_awareness.py (NEW)

# 1. Load data
event_manager = EventManager()
model = EnhancedRankingModel()

# 2. For each training batch:
#    - Get date
#    - Get anomaly_level for that date
#    - Train with anomaly awareness
#    - Loss: pairwise + anomaly weighting

# 3. Evaluate on validation set
# 4. Save model + metrics
```

**Deliverable**: `models/ranking_model_with_anomaly.pkl`

#### ✅ Check-in Week 2:
- [ ] Event anomaly detector funciona
- [ ] Model treina com anomaly flags
- [ ] Validação mostra P@5-20
- [ ] Status: Pronto para Week 3

---

### SEMANA 3: Long-Tail Optimization (Feb 21-27)

**Objetivo**: Melhorar P@20 (cobertura de nodes fora do top-5)

#### Task 3.1: Analyze Ranking Errors (4h)
```python
# File: analysis/ranking_error_analysis.py (NEW)

# Para cada node fora top-5:
# - Qual deveria ser o rank?
# - Qual é o rank predito?
# - Problema: undershooting ou overshooting?

# Exemplo:
# Node 240: CVLI=5.5, deveria estar em posição ~15
#          Model prediz posição 45
#          Problema: undershooting (prediz muito baixo)

# For all 319 nodes:
undershot = nodes onde rank_pred > rank_true
overshot = nodes onde rank_pred < rank_true

# Análise: Qual é o padrão?
# - Temporal features não ajudam long-tail?
# - Spatial neighbors não são considerados?
# - Problemas com scale/normalization?
```

**Deliverable**: `error_analysis.json`

#### Task 3.2: Add Top-K Loss (8h)
```python
# File: src/loss_functions.py (NEW)

class RankingLosses:
    """
    Customized loss functions para P@K task
    """
    
    @staticmethod
    def pairwise_ranking_loss(y_pred, y_true):
        """Original loss - otimiza para P@5"""
        # Same as before
        pass
    
    @staticmethod
    def top_k_loss(y_pred, y_true, k=20):
        """
        Loss para otimizar P@K
        
        Ideia: Penalize misranking top-K nodes mais que outros
        
        Loss = sum_i weight_i * |rank_i_pred - rank_i_true|
               where weight_i is high if i should be in top-K
        """
        real_top_k = torch.argsort(-y_true)[:k]
        weight = torch.zeros_like(y_true)
        weight[real_top_k] = 1.0  # Weight 1.0 for top-K
        weight[~real_top_k] = 0.1  # Weight 0.1 for rest
        
        # Ranking loss with weighted importance
        loss = 0.0
        pred_ranks = torch.argsort(-y_pred)
        true_ranks = torch.argsort(-y_true)
        
        for i in range(len(y_pred)):
            pred_rank = torch.where(pred_ranks == i)[0].item()
            true_rank = torch.where(true_ranks == i)[0].item()
            loss += weight[i] * abs(pred_rank - true_rank)
        
        return loss
    
    @staticmethod
    def combined_loss(y_pred, y_true, alpha=0.7):
        """
        Combine P@5 and P@20 optimization
        
        Loss = alpha * pairwise_loss(y_pred, y_true, k=5)
             + (1-alpha) * top_k_loss(y_pred, y_true, k=20)
        """
        loss_p5 = RankingLosses.pairwise_ranking_loss(y_pred, y_true)
        loss_p20 = RankingLosses.top_k_loss(y_pred, y_true, k=20)
        return alpha * loss_p5 + (1 - alpha) * loss_p20
```

**Deliverable**: `src/loss_functions.py`

#### Task 3.3: Treinar com Combined Loss (8h)
```python
# File: scripts/train_for_p20_coverage.py (NEW)

# 1. Load data + baseline model
model = EnhancedRankingModel()

# 2. Train com combined loss
#    Loss = 0.5 * P@5_loss + 0.5 * P@20_loss
#    Objetivo: Balance between top-5 e long-tail

# 3. Evaluate:
#    - P@5: Should stay ~0.80 (não piora)
#    - P@20: Should improve 0.50 → 0.55-0.60
#    - P@10: Should improve too

# 4. If good: Save model
```

**Deliverable**: `models/model_with_p20_optimization.pkl`

#### Task 3.4: Comparison & Decision (4h)
```python
# File: analysis/compare_models.py (NEW)

models = {
    'baseline': load_model('baseline'),
    'with_anomaly': load_model('with_anomaly'),
    'with_p20_focus': load_model('with_p20_focus')
}

for model_name, model in models.items():
    metrics = evaluate_comprehensive(model)
    print(f"{model_name}:")
    print(f"  P@5: {metrics['p_at_5']:.3f}")
    print(f"  P@10: {metrics['p_at_10']:.3f}")
    print(f"  P@20: {metrics['p_at_20']:.3f}")

# Decision: Which model to move forward with?
# Criteria:
# ✅ P@5 >= 0.78 (don't hurt top-5)
# ✅ P@20 >= 0.55 (+10% improvement)
# ✅ Generalization is good (val ≈ test)
```

**Deliverable**: Comparison report + selected model

#### ✅ Check-in Week 3:
- [ ] Long-tail analysis complete
- [ ] Combined loss implemented
- [ ] P@20 improved 0.50 → 0.55-0.60
- [ ] Model selected for Week 4
- [ ] Status: Pronto para Week 4

---

### SEMANA 4: Explainability (Feb 28 - Mar 6)

**Objetivo**: Add LLM-based explanations (preparação para doutorado)

#### Task 4.1: Explanation Generator (6h)
```python
# File: src/explanation_generator.py (NEW)

class ExplanationGenerator:
    """
    Gera explicações human-readable para rankings
    
    Pode usar simple heuristics (MVP) ou real LLM (later)
    """
    
    def __init__(self, model, data_manager):
        self.model = model
        self.data_manager = data_manager
    
    def explain_node_ranking(self, node_id, rank, context_dict):
        """
        Cria explicação estruturada
        
        Input:
        - node_id: qual nó
        - rank: qual posição
        - context_dict: {
            'score': float,
            'temporal_pattern': str (e.g., "high in Feb"),
            'nearby_nodes': list,
            'events': list,
            'similarity': float (correlação com vizinhos)
          }
        
        Output:
        {
            'summary': "Node 146 is #1 because it has...",
            'factors': [
                {'name': 'Seasonal', 'contribution': 0.4, 'explanation': 'High in Feb'},
                {'name': 'Spatial', 'contribution': 0.3, 'explanation': 'Neighbors 145, 147 also high'},
                {'name': 'Exogenous', 'contribution': 0.3, 'explanation': 'Event in Aldeota'}
            ],
            'confidence': 0.87,
            'caveats': ['High event activity', 'Unusual for this season']
        }
        """
        
        # Heuristic approach (MVP)
        factors = []
        
        # Temporal factor
        if 'temporal_pattern' in context_dict:
            pattern = context_dict['temporal_pattern']
            factors.append({
                'name': 'Temporal Pattern',
                'contribution': 0.4,
                'explanation': pattern
            })
        
        # Spatial factor
        if 'nearby_nodes' in context_dict:
            nearby = context_dict['nearby_nodes']
            factors.append({
                'name': 'Spatial Correlation',
                'contribution': 0.3,
                'explanation': f'Neighbors {nearby} also high'
            })
        
        # Event factor
        if 'events' in context_dict and context_dict['events']:
            factors.append({
                'name': 'Recent Events',
                'contribution': 0.3,
                'explanation': f"{len(context_dict['events'])} events in area"
            })
        
        # Summary
        summary = f"Node {node_id} (rank #{rank}) is predicted here because: "
        summary += " + ".join([f"({f['name']}: {f['contribution']*100:.0f}%)" for f in factors])
        
        return {
            'summary': summary,
            'factors': factors,
            'confidence': context_dict.get('confidence', 0.85),
            'caveats': []
        }
```

**Deliverable**: `src/explanation_generator.py`

#### Task 4.2: API Enhancement (6h)
```python
# File: src/app.py (MODIFY existing)

@app.route('/explain/<int:node_id>')
def explain_node(node_id):
    """
    Get explanation for why node is ranked as it is
    """
    
    # Get latest prediction
    predictions = get_latest_predictions()  # shape: (319,)
    y_pred = predictions.values
    
    # Get context for this node
    rank = np.argsort(-y_pred).tolist().index(node_id) + 1
    score = y_pred[node_id]
    
    # Temporal context
    temporal_context = analyze_temporal_pattern(node_id)
    
    # Spatial context
    nearby_nodes = get_top_neighbors(node_id, k=3)
    
    # Events context
    recent_events = event_manager.get_recent_events(node_id)
    
    context = {
        'score': float(score),
        'temporal_pattern': temporal_context,
        'nearby_nodes': nearby_nodes,
        'events': recent_events,
        'confidence': 0.87
    }
    
    # Generate explanation
    explainer = ExplanationGenerator(model, None)
    explanation = explainer.explain_node_ranking(node_id, rank, context)
    
    return jsonify(explanation)

@app.route('/metrics')
def get_metrics():
    """
    Get comprehensive metrics: P@5-20, NDCG, etc.
    """
    y_true = get_actual_data()
    y_pred = get_predictions()
    
    metrics = MetricReporter().report(y_true, y_pred)
    
    return jsonify(metrics)

@app.route('/anomaly_status')
def get_anomaly_status():
    """
    Is there an event today? How confident are we?
    """
    today_events = event_manager.get_events_for_date(datetime.now().date())
    anomaly_level = event_manager.get_anomaly_level_for_date(datetime.now().date())
    
    return jsonify({
        'has_events': len(today_events) > 0,
        'anomaly_level': anomaly_level,
        'events': today_events,
        'model_confidence': 1.0 - (anomaly_level * 0.3),
        'warning': "Model confidence reduced due to recent events" if anomaly_level > 0.6 else None
    })
```

**Deliverable**: Enhanced API endpoints

#### Task 4.3: Dashboard Update (4h)
```html
<!-- File: templates/dashboard.html (MODIFY existing) -->

<!-- Add new sections -->

<div id="metrics">
  <h2>Comprehensive Metrics</h2>
  <table>
    <tr>
      <th>Metric</th>
      <th>Value</th>
      <th>Target</th>
      <th>Status</th>
    </tr>
    <tr>
      <td>P@5</td>
      <td id="p5" class="metric">-</td>
      <td>≥0.80</td>
      <td id="p5_status">-</td>
    </tr>
    <tr>
      <td>P@10</td>
      <td id="p10" class="metric">-</td>
      <td>≥0.65</td>
      <td id="p10_status">-</td>
    </tr>
    <tr>
      <td>P@20</td>
      <td id="p20" class="metric">-</td>
      <td>≥0.55</td>
      <td id="p20_status">-</td>
    </tr>
    <tr>
      <td>NDCG@20</td>
      <td id="ndcg20" class="metric">-</td>
      <td>≥0.75</td>
      <td id="ndcg20_status">-</td>
    </tr>
  </table>
</div>

<div id="anomaly_alert">
  <h2>Anomaly Status</h2>
  <div id="anomaly_content">
    <p>No events detected. Model confidence: <span id="confidence">-</span></p>
  </div>
</div>

<div id="explanation">
  <h2>Why is Node <input id="node_input" type="number" min="0" max="318"> ranked where?</h2>
  <div id="explanation_content"></div>
</div>

<script>
  // On load
  fetch('/metrics').then(r => r.json()).then(metrics => {
    document.getElementById('p5').innerText = metrics.p_at_5.toFixed(3);
    document.getElementById('p10').innerText = metrics.p_at_10.toFixed(3);
    document.getElementById('p20').innerText = metrics.p_at_20.toFixed(3);
    document.getElementById('ndcg20').innerText = metrics.ndcg_at_20.toFixed(3);
  });
  
  // Anomaly status
  fetch('/anomaly_status').then(r => r.json()).then(data => {
    document.getElementById('confidence').innerText = (data.model_confidence * 100).toFixed(0) + '%';
    if (data.warning) {
      document.getElementById('anomaly_alert').style.backgroundColor = 'yellow';
      document.getElementById('anomaly_content').innerText = data.warning;
    }
  });
  
  // Explanation on demand
  document.getElementById('node_input').addEventListener('change', (e) => {
    fetch(`/explain/${e.target.value}`).then(r => r.json()).then(expl => {
      document.getElementById('explanation_content').innerHTML = 
        `<p>${expl.summary}</p>` +
        expl.factors.map(f => `<li>${f.name}: ${(f.contribution*100).toFixed(0)}% - ${f.explanation}</li>`).join('');
    });
  });
</script>
```

**Deliverable**: Enhanced dashboard

#### Task 4.4: Documentation (4h)
```markdown
# File: docs/EXPLAINABILITY_GUIDE.md (NEW)

## How Explanations Work

### Example
"Node 146 is ranked #1 because:
- Temporal Pattern (40%): High crime in Aldeota during February
- Spatial Correlation (30%): Neighbors 145, 147 also high
- Recent Events (30%): 2 events (robbery) in past 3 days

Model confidence: 87% (slightly reduced due to event activity)"

### API Usage
GET /explain/{node_id}

Response:
{
  "summary": "...",
  "factors": [
    {"name": "...", "contribution": 0.4, "explanation": "..."}
  ],
  "confidence": 0.87,
  "caveats": [...]
}

### Future: Real LLM
Can upgrade to Google Gemini API for more natural explanations:
"Node 146 is the highest-risk area because it's part of the Aldeota 
neighborhood, which historically has high crime in February. 
Recent gang activity (events from 3 days ago) further elevates risk."
```

**Deliverable**: `docs/EXPLAINABILITY_GUIDE.md`

#### ✅ Check-in Week 4:
- [ ] ExplanationGenerator works
- [ ] API endpoints /explain, /metrics, /anomaly_status live
- [ ] Dashboard shows all metrics + explanations
- [ ] Documentation complete
- [ ] Status: Enhanced system ready for Week 5

---

### SEMANA 5: Testing & Deployment (Mar 7-13)

**Objetivo**: Validação completa + deploy para produção

#### Task 5.1: Comprehensive Testing (6h)
```python
# File: tests/test_enhanced_system.py (NEW)

class TestEnhancedSystem:
    
    def test_metrics_calculation(self):
        """Verify P@5-20, NDCG work correctly"""
        y_true = [8, 7, 6, 5, 4, 3, 2, 1]
        y_pred = [7.9, 6.8, 5.7, 4.6, 4.5, 3.4, 2.3, 1.2]
        
        metrics = MetricReporter().report(y_true, y_pred)
        
        assert metrics['p_at_5'] ∈ [0, 1]
        assert metrics['p_at_10'] ∈ [0, 1]
        assert metrics['p_at_20'] ∈ [0, 1]
        assert metrics['ndcg_at_5'] ∈ [0, 1]
        # ... more tests
    
    def test_anomaly_detection(self):
        """Verify event parsing + anomaly detection"""
        detector = EventAnomalyDetector()
        
        result = detector.parse_event("Homicídio em Messejana")
        assert result['severity'] == 1.0
        assert result['anomaly_flag'] == True
        
        result = detector.parse_event("Furto de celular")
        assert result['severity'] < 0.5
        assert result['anomaly_flag'] == False
    
    def test_explanation_generation(self):
        """Verify explanations are sensible"""
        gen = ExplanationGenerator(model, None)
        
        context = {
            'score': 7.5,
            'temporal_pattern': 'High in February',
            'nearby_nodes': [145, 147],
            'events': ['robbery'],
            'confidence': 0.87
        }
        
        expl = gen.explain_node_ranking(146, 1, context)
        
        assert 'summary' in expl
        assert 'factors' in expl
        assert len(expl['factors']) > 0
        assert expl['confidence'] > 0.75
    
    def test_api_endpoints(self):
        """Verify all endpoints return correct format"""
        # /metrics
        response = client.get('/metrics')
        assert response.status_code == 200
        assert 'p_at_5' in response.json
        
        # /explain/123
        response = client.get('/explain/123')
        assert response.status_code == 200
        assert 'summary' in response.json
        
        # /anomaly_status
        response = client.get('/anomaly_status')
        assert response.status_code == 200
        assert 'has_events' in response.json
    
    def test_generalization_multiple_windows(self):
        """Verify model generalizes across time windows"""
        windows = split_data_into_rolling_windows(10)
        
        scores = []
        for window_train, window_test in windows:
            model.train(window_train)
            p5 = model.evaluate(window_test)['p_at_5']
            scores.append(p5)
        
        assert np.mean(scores) >= 0.78  # Don't hurt P@5
        assert np.std(scores) < 0.15    # Stable
```

**Deliverable**: Comprehensive test suite

#### Task 5.2: End-to-End Validation (4h)
```python
# File: scripts/final_validation.py (NEW)

# Simulação: Run model em últimos 30 dias com métricas
# Report: P@5, P@10, P@20, NDCG, etc.

predictions = model.predict(test_data)
y_true = test_data.targets

metrics = MetricReporter().report(y_true, predictions)

print("=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)
print(f"P@5:    {metrics['p_at_5']:.3f}  (baseline: 0.80, target: ≥0.78)")
print(f"P@10:   {metrics['p_at_10']:.3f}  (baseline: 0.65, target: ≥0.65)")
print(f"P@20:   {metrics['p_at_20']:.3f}  (baseline: 0.50, target: ≥0.55)")
print(f"NDCG@5: {metrics['ndcg_at_5']:.3f}  (baseline: 0.92, target: ≥0.92)")
print(f"NDCG@20:{metrics['ndcg_at_20']:.3f} (baseline: 0.75, target: ≥0.76)")

# GO/NO-GO Decision
go = (metrics['p_at_5'] >= 0.78 and
      metrics['p_at_20'] >= 0.55 and
      metrics['ndcg_at_5'] >= 0.92)

if go:
    print("\n✅ GO: Ready for production deployment")
else:
    print("\n❌ NO-GO: More work needed")
    if metrics['p_at_5'] < 0.78:
        print("  - P@5 is too low, hurt model quality")
    if metrics['p_at_20'] < 0.55:
        print("  - P@20 didn't improve")
```

**Deliverable**: Final validation report

#### Task 5.3: Deployment Prep (6h)
```python
# File: scripts/deploy.py (NEW)

"""
Deployment checklist:
✅ Model trained on full dataset
✅ All tests pass
✅ Metrics validated
✅ API endpoints tested
✅ Dashboard displays correctly
✅ Event anomaly integration working
✅ Explanations generated correctly

Deployment steps:
1. Backup current model (current_model.pkl → backup_model.pkl)
2. Load new model into memory
3. Verify on shadow traffic (10% of requests, don't log)
4. If 99%+ success rate: Switch 50% traffic
5. Monitor for 6 hours
6. If metrics stable: Switch 100% traffic
7. Monitor for 24 hours
"""

def deploy_model(new_model_path, shadow_traffic_duration=3600):
    """
    Canary deployment: start with shadow traffic
    """
    
    # Backup
    os.system("cp models/ranking_model.pkl models/ranking_model.backup.pkl")
    
    # Load new
    new_model = load_model(new_model_path)
    
    # Shadow traffic test
    print("Running shadow traffic test (10% requests, 1 hour)...")
    success_count = 0
    for request in get_requests(duration=shadow_traffic_duration, percentage=0.1):
        try:
            prediction = new_model.predict(request)
            success_count += 1
        except:
            pass
    
    success_rate = success_count / 100  # Expected ~100 requests in 1 hour (10%)
    
    if success_rate >= 0.99:
        print(f"✅ Shadow test passed ({success_rate*100:.0f}% success)")
        
        # Switch to 50% traffic
        print("Switching to 50% traffic...")
        global ACTIVE_MODELS
        ACTIVE_MODELS = {
            'primary': new_model,
            'fallback': old_model,
            'traffic_split': 0.5
        }
        
        # Monitor
        print("Monitoring for 6 hours...")
        metrics_before = get_metrics()
        time.sleep(6 * 3600)
        metrics_after = get_metrics()
        
        if is_stable(metrics_before, metrics_after):
            print("✅ Metrics stable, switching to 100% traffic")
            ACTIVE_MODELS = {'primary': new_model}
            
            # Final monitoring
            print("Final monitoring for 24 hours...")
            time.sleep(24 * 3600)
            
            print("✅ Deployment successful!")
        else:
            print("❌ Metrics degraded, rolling back")
            ACTIVE_MODELS = {'primary': old_model}
    else:
        print(f"❌ Shadow test failed ({success_rate*100:.0f}% success)")
```

**Deliverable**: Deployment script + checklist

#### Task 5.4: Documentation Final (4h)
```markdown
# File: docs/DEPLOYMENT_GUIDE.md (NEW)

## System Architecture v2

### Components
1. **ST-GCN**: Temporal+spatial patterns
2. **Enhanced RankingModel**: P@5-20 optimization + anomaly awareness
3. **EventAnomalyDetector**: Event parsing + severity classification
4. **ExplanationGenerator**: Why-this-ranking explanations
5. **MetricReporter**: P@5-20, NDCG, Recall metrics

### Data Flow

```
Event (JSON) → EventAnomalyDetector → Anomaly flag
                                   ↓
                                   v
CVLI data (26D) → ST-GCN → RankingModel → Predictions (all 319)
                  (features)  + anomaly │
                                       ├→ Metrics (P@5-20, NDCG)
                                       ├→ Explanations (per node)
                                       └→ API response
```

### API Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/predict` | Get current predictions | `{"scores": [...], "top_5": [...], "confidence": 0.87}` |
| `/metrics` | Get comprehensive metrics | `{"p_at_5": 0.80, "p_at_20": 0.55, ...}` |
| `/explain/{node_id}` | Why is this node ranked? | `{"summary": "...", "factors": [...]}` |
| `/anomaly_status` | Events today? | `{"has_events": true, "anomaly_level": 0.7, ...}` |

### Deployment

See `scripts/deploy.py` for canary deployment strategy.
```

**Deliverable**: Complete deployment documentation

#### ✅ Final Check-in:
- [ ] All tests pass
- [ ] Final validation shows metrics met
- [ ] Deployment script tested (dry-run)
- [ ] Documentation complete
- [ ] **STATUS: READY FOR PRODUCTION**

---

## 📊 MILESTONE TRACKING

```
WEEK 1 (Feb 7-13):    ██░░░░░░░░░░░░░░░░░░ Setup + Baseline
WEEK 2 (Feb 14-20):   ████░░░░░░░░░░░░░░░ Event Integration
WEEK 3 (Feb 21-27):   ██████░░░░░░░░░░░░░ Long-tail Optimization
WEEK 4 (Feb 28-6):    ████████░░░░░░░░░░░ Explainability
WEEK 5 (Mar 7-13):    ██████████░░░░░░░░░ Testing & Deployment

Total: 50 hours of work, 5 people sprints
Expected outcome: P@5 maintained (~0.80), P@20 improved (~0.55-0.60), 
                  system stable against events, explanations available
```

---

## 🎯 KEY SUCCESS METRICS

| Metric | Current | Target | Owner |
|--------|---------|--------|-------|
| P@5 | 0.80 | ≥0.78 | RankingModel |
| P@20 | 0.50 | ≥0.55 | Long-tail loss |
| NDCG@5 | 0.92 | ≥0.92 | Regression test |
| P@5 in event days | ~0.65 | ≥0.75 | Anomaly detector |
| API latency | 150ms | <200ms | Inference |
| Explanation quality | N/A | >3 factors/node | ExplanationGen |
| Uptime | 99.5% | 99.5% | Deployment |

---

## 📁 FILE STRUCTURE (New Files)

```
src/
├── metrics.py (NEW)                    # P@K and NDCG functions
├── event_anomaly_detector.py (NEW)     # Parse events + detect anomalies
├── event_manager.py (NEW)              # Manage exogenous events
├── ranking_model.py (MODIFY)           # Add anomaly awareness
├── loss_functions.py (NEW)             # Combined losses (P@5 + P@20)
└── explanation_generator.py (NEW)      # Generate explanations

scripts/
├── evaluate_baseline_metrics.py (NEW)   # Baseline P@K metrics
├── train_with_anomaly_awareness.py (NEW) # Train with event flags
├── train_for_p20_coverage.py (NEW)     # Optimize for P@20
├── compare_models.py (NEW)             # Compare approaches
├── final_validation.py (NEW)           # End-to-end validation
└── deploy.py (NEW)                     # Canary deployment

analysis/
├── long_tail_analysis.py (NEW)         # Analyze missed nodes
├── ranking_error_analysis.py (NEW)     # Where do we go wrong?
└── compare_models.py (NEW)             # Side-by-side comparison

tests/
└── test_enhanced_system.py (NEW)       # Comprehensive tests

templates/
└── dashboard.html (MODIFY)             # Add metrics + explanations

docs/
├── EXPLAINABILITY_GUIDE.md (NEW)       # How explanations work
└── DEPLOYMENT_GUIDE.md (NEW)           # System v2 overview
```

---

## ⚠️ RISKS & CONTINGENCIES

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| P@20 doesn't improve | MEDIUM | HIGH | Use different loss weighting |
| Event detection too aggressive | MEDIUM | MEDIUM | Tune anomaly threshold |
| Explanations not useful | LOW | MEDIUM | Simplify to template-based |
| API latency increases | LOW | HIGH | Cache predictions + explanations |
| Deployment breaks something | LOW | CRITICAL | Extensive testing + canary |

---

## 🚀 STARTING POINT (TODAY)

**When ready to start:**

1. Create the files above in correct directories
2. Start WEEK 1 TASK 1.1 (metrics.py)
3. Follow checklist-style approach
4. Daily standup: "What did I complete? What's blocking?"
5. Weekly review: Check-in against milestones

**Owner**: Data Science Team Lead (track progress)  
**Frequency**: Daily commits, weekly reviews  
**Escalation**: Any task blocking > 1 day → escalate

---

**This plan is your NORTH STAR for the next 5 weeks. Stay on track!**

🎯 **ULTIMATE GOAL**: Ship Phase 2b (B+C+D) with confidence, production-ready
