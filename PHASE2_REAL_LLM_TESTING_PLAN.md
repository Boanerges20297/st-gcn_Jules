# 🚀 PHASE 2.5: REAL LLM TESTING PLAN

**Data**: 06/02/2026  
**Predecessor**: Phase 2 Mock Tests (Completed ✅)  
**Goal**: Validate Approach 3 (Severity Detection) com REAL LLM  
**Timeline**: 2-3 dias

---

## 🎯 Objetivo

Confirmar que LLM features trazem **real improvement** em P@5:
```
Current Production:    P@5 = 0.80 (baseline realista)
Phase 2 Mock Tests:    P@5 = 0.86 (artificial, deterministic)
Goal Real Tests:       P@5 ≥ 0.82 (ganho real +2.5%)
```

---

## 📋 Tarefas Sequenciais

### TASK 1: Setup LLM API (3h)

#### 1.1 Escolher Provider ⭐ RECOMENDADO
```
┌────────────────────────────────────────────┐
│ GOOGLE GEMINI API 1.5 FLASH               │
├────────────────────────────────────────────┤
│ ✅ Custo: $0.075 / 1M input tokens        │
│ ✅ Velocidade: ~100-200ms                  │
│ ✅ Token limit: 1M (suficiente)           │
│ ✅ JSON mode: SIM                         │
│ ✅ Via google.generativeai library        │
│                                            │
│ 50 eventos × 500 tokens avg × $0.075     │
│ = ~$1.87 total cost                       │
└────────────────────────────────────────────┘
```

#### 1.2 Setup Credenciais
```python
# File: .env (GITIGNORE!)
GOOGLE_API_KEY=sk-xxxxxxxxxxxxx

# Installation
pip install google-generativeai python-dotenv
```

#### 1.3 Test API Connection
```python
import google.generativeai as genai
import os

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Test")
print(response.text)  # Should work
```

---

### TASK 2: Create LLM Event Parser (4h)

#### 2.1 Design Prompt Template
```
Você é um especialista em análise de segurança pública. 
Analise o seguinte evento de crime/segurança:

EVENTO:
{event_text}

Forneça a seguinte análise em JSON (APENAS JSON, sem texto adicional):
{{
  "event_id": "string (gerado ou fornecido)",
  "severity_level": "CRITICAL|HIGH|MEDIUM|LOW",
  "severity_score": float (0.0-1.0),
  "crime_types": ["list", "of", "crime", "types"],
  "affected_neighborhoods": ["list", "of", "bairros"],
  "police_response_priority": "immediate|fast|moderate|slow|delayed",
  "estimated_affected_nodes": [list, of, node, ids],
  "context": "brief explanation of why this assessment",
  "confidence": float (0.0-1.0)
}}

Ser conservador em severity (não super-estimar).
Basear em fatos, não em sensacionalismo.
```

#### 2.2 Load Real Events
```python
import json

with open('data/exogenous_events_geocoded.json') as f:
    events = json.load(f)

# Should have 20-50 events
print(f"Loaded {len(events)} events")
```

#### 2.3 Parse with Real LLM
```python
def parse_event_with_llm(event_text: str) -> dict:
    """Parse event using Google Gemini"""
    
    prompt = f"""
    [PROMPT_TEMPLATE]
    
    EVENT:
    {event_text}
    """
    
    response = model.generate_content(prompt)
    
    try:
        result = json.loads(response.text)
        return result
    except:
        print(f"  ⚠️ Failed to parse: {event_text[:50]}")
        return None

# Batch parse
parsed_events = []
for i, event in enumerate(events[:30]):
    print(f"  {i+1}/30: Parsing...")
    parsed = parse_event_with_llm(event['text'])
    if parsed:
        parsed_events.append(parsed)
    time.sleep(0.5)  # Rate limiting

# Save
with open('parsed_events_real_llm.json', 'w') as f:
    json.dump(parsed_events, f, indent=2)
```

---

### TASK 3: Feature Engineering with Real Data (2h)

#### 3.1 Create Features
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_severity_features(parsed_events: list, num_nodes: int = 319) -> np.ndarray:
    """
    Create 40D feature matrix from parsed events
    
    Returns: (319, 40) matrix
    """
    
    # Feature slots
    # 10D: severity one-hot
    # 15D: crime_type one-hot
    # 5D: police_response one-hot
    # 10D: aggregations (weighted sums)
    
    features = np.zeros((num_nodes, 40))
    
    crime_taxonomy = {
        'homicide': 1.0, 'robbery': 0.8, 'assault': 0.6,
        # ... complete taxonomy
    }
    
    for event in parsed_events:
        severity_score = event['severity_score']
        crime_types = event['crime_types']
        affected_nodes = event['estimated_affected_nodes']
        police_priority = event['police_response_priority']
        
        # Average crime importance
        crime_importance = np.mean([
            crime_taxonomy.get(ct, 0.5) for ct in crime_types
        ])
        
        # Weighted score
        combined_score = severity_score * crime_importance
        
        # Apply to affected nodes
        for node_id in affected_nodes:
            if 0 <= node_id < num_nodes:
                features[node_id] += combined_score
    
    # Scale
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled

severity_features = engineer_severity_features(parsed_events)
print(severity_features.shape)  # (319, 40)
```

---

### TASK 4: Train RankingModel with Real Features (3h)

#### 4.1 Load Production Data
```python
import pickle
import torch

# Load real CVLI data from production
with open('data/processed_graph_data.pkl', 'rb') as f:
    graph_data = pickle.load(f)

# Extract 26 original features + 40 new severity features = 66 total
node_features = graph_data['node_features']  # (319, N_days, 26)
cvli_data = graph_data['cvli_data']  # (319, N_days)

print(f"Original shape: {node_features.shape}")
# (319, 1491, 26)
```

#### 4.2 Combine with Severity Features
```python
# Append severity features to each day
N_nodes, N_days, N_original_features = node_features.shape

combined_features = np.zeros((N_nodes, N_days, 66))
combined_features[:, :, :26] = node_features  # Original 26
combined_features[:, :, 26:66] = severity_features[:, np.newaxis, :]  # Broadcast severity

print(f"Combined shape: {combined_features.shape}")
# (319, 1491, 66)
```

#### 4.3 Train on Test Period
```python
# Use SAME validation/test split as Phase 1
# Train: days 0-1200
# Val: days 1200-1350
# Test: days 1350-1491 (last 141 days)

X_train = combined_features[:, :1200, :]  # (319, 1200, 66)
X_val = combined_features[:, 1200:1350, :]  # (319, 150, 66)
X_test = combined_features[:, 1350:, :]  # (319, 141, 66)

y_train = cvli_data[:, :1200]
y_val = cvli_data[:, 1200:1350]
y_test = cvli_data[:, 1350:]

# Train RankingModel with 66D input
model_v2 = RankingModel(input_dim=66, hidden_dim=128, output_dim=1)

# Training loop (reuse from Phase 1 code)
# ... standard PyTorch training ...
```

#### 4.4 Evaluate
```python
# RankingModel validation
val_results = evaluate_ranking(model_v2, X_val, y_val, top_k=5)
print(f"Validation P@5: {val_results['p_at_5']:.3f}")

# Test set (generalization)
test_results = evaluate_ranking(model_v2, X_test, y_test, top_k=5)
print(f"Test P@5: {test_results['p_at_5']:.3f}")

# Compare vs baseline
baseline_p5 = 0.80  # From production
improvement = test_results['p_at_5'] - baseline_p5
print(f"Improvement: {improvement:+.3f} ({improvement/baseline_p5*100:+.1f}%)")

# GO/NO-GO decision
if test_results['p_at_5'] >= 0.82:
    print("✅ GO: Ready for production!")
elif test_results['p_at_5'] >= 0.80:
    print("⚠️  MAYBE: Borderline, iterate")
else:
    print("❌ NO-GO: Features don't help")
```

---

### TASK 5: Ablation & Feature Importance (2h)

#### 5.1 Drop Features by Group
```python
def ablation_test(model, X_test, y_test):
    """Test importance of each feature group"""
    
    results = {}
    baseline_p5 = evaluate_ranking(model, X_test, y_test)['p_at_5']
    
    # Drop original 26 features
    X_no_original = X_test.copy()
    X_no_original[:, :, :26] = 0
    p5_no_original = evaluate_ranking(model, X_no_original, y_test)['p_at_5']
    results['original_26'] = p5_no_original
    
    # Drop new severity features
    X_no_severity = X_test.copy()
    X_no_severity[:, :, 26:66] = 0
    p5_no_severity = evaluate_ranking(model, X_no_severity, y_test)['p_at_5']
    results['severity_40'] = p5_no_severity
    
    print(f"Baseline P@5: {baseline_p5:.3f}")
    print(f"Without original 26: {p5_no_original:.3f} (Δ{p5_no_original-baseline_p5:+.3f})")
    print(f"Without severity 40: {p5_no_severity:.3f} (Δ{p5_no_severity-baseline_p5:+.3f})")
    
    return results
```

---

## 📅 GANTT Timeline

```
Feb 6 (Today):
└─ ✅ Phase 2 Mock tests completed

Feb 7:
├─ [3h] TASK 1: Setup LLM API
├─ [4h] TASK 2: Parse real events
└─ Status: Ready for training

Feb 8:
├─ [3h] TASK 3: Feature engineering
├─ [3h] TASK 4: Train & evaluate
└─ Status: Results in hand

Feb 9:
├─ [2h] TASK 5: Ablation analysis
├─ [2h] Write report
└─ Final Decision: Go/No-Go
```

---

## 💰 Costs

| Item | Cost | Notes |
|------|------|-------|
| Google Gemini API | ~$2 | 50 events × 500 tokens |
| Engineering time | ~16h | 1-2 data scientists |
| Total | ~$2 | Very cheap for potential 2.5% gain |

---

## 🎯 Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Real P@5 ≥ 0.82 | ✅ Go | TBD |
| Test generalization | ✅ Similar to val | TBD |
| Ablation shows signal | ✅ Severity helps | TBD |
| No latency increase > 30ms | ✅ <200ms total | TBD |

---

## 📝 Deliverables

```
reports/
├── phase2_real_llm_results.json          # P@5, metrics by window
├── ablation_importance.json              # Feature importance
├── parsed_events_validation.json         # QA check
└── PHASE2_FINAL_REPORT.md                # Recommendation
```

---

## ⚠️ Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| LLM parsing fails | LOW | Manual validation + fallback |
| Features don't help | MEDIUM | Use mock results to understand why |
| Latency increases too much | LOW | Use batch inference, cache |
| API costs spike | LOW | Rate limit, cache results |

---

## ✅ Next Steps (TODAY)

1. Ler este plano completamente
2. Setup Google API key
3. Decidir: Aprovar Real LLM testing?
4. If YES: Start TASK 1 tomorrow

---

**Status**: 🟢 READY  
**Owner**: Data Science Team  
**Approval**: Required before starting
