# ST-GCN Crime Prediction - Architecture Reference

## Project Status
✅ **Phase 1 Complete**: Hyperparameter optimization achieved NDCG@5=0.9995 (near-perfect ranking)
📊 **Active Model**: Pairwise Ranking Loss (ranking_model_v2.py)
🔄 **Next Phase**: LLM Semantic Features Integration

---

## 🚀 Quick Start

### Active Production Code
```bash
# Train the best-performing ranking model
python train_ranking_v2.py

# Evaluate with rigorous metrics (NDCG@5, Spearman, P@5)
python eval_ranking_models.py

# Run Flask API
python app.py
```

### Best Trained Model
- **File**: `models/ranking_model_best_Config_01_Small.pkl`
- **Performance**: NDCG@5=0.9995, P@5=1.0000, Spearman=0.9766
- **Config**: batch_size=4, lr=0.001, hidden_dim=64
- **Improvement vs ST-GCN**: +566% (0.15 → 1.0)

---

## 📁 Directory Structure - Active Components

### Core Model Code (`/src/`)

| File | Purpose | Status |
|------|---------|--------|
| `ranking_model_v2.py` | Pairwise Ranking Loss + MLP | ✅ ACTIVE |
| `ranking_features.py` | 26-channel feature extraction | ✅ ACTIVE |
| `data_processing.py` | Data pipeline (ST-GCN original) | ✅ ACTIVE |
| `model.py` | ST-GCN architecture | 📚 REFERENCE |
| `train.py` | ST-GCN training (original) | 📚 REFERENCE |
| `llm_service.py` | Google Generative AI integration | 🔄 PHASE 2 |
| `validate_predictions.py` | Post-training validation | ⚙️ UTILITY |
| `analyze_models.py` | Comparative model analysis | ⚙️ UTILITY |

### Training Scripts (Root)

| File | Purpose | Status |
|------|---------|--------|
| `train_ranking_v2.py` | Best ranking trainer | ✅ ACTIVE |
| `hyperparam_search.py` | Grid search (12 configs) | ✅ COMPLETED |
| `eval_ranking_models.py` | Rigorous evaluation | ✅ COMPLETED |
| `app.py` | Flask API server | ✅ ACTIVE |

### Deprecated/Removed Files
- ❌ `src/ranking_model.py` - Old ListNet v1 (superseded)
- ❌ `train_ranking.py` - Old v1 training (superseded)
- ❌ `test_categorical_features.py` - Debug script
- ❌ `debug_train.py` - Debugging
- ❌ `analyze_tuning.py` - Old hyperparameter analysis
- ❌ `hybrid_analysis.py` - Speculative LLM analysis
- ❌ `prove_obvious_patterns.py` - POC script

---

## 🏗️ Active Architecture

### Data Flow

```
Raw Crime Data (JSON)
    ↓
data_processing.py (load + normalize)
    ↓
ranking_features.py (26-channel extraction)
    ├─ Channels 0-2: CVLI, CVP, Tension metrics
    ├─ Channels 3-9: Day-of-week (one-hot)
    ├─ Channels 10-21: Month (one-hot)
    ├─ Channels 22: Weekend flag
    └─ Channels 23-25: Reserved
    ↓
processed_graph_data.pkl (319 nodes × 1491 timesteps × 26 channels)
    ↓
train_ranking_v2.py (training pipeline)
    ├─ PairwiseLoss (optimizes ranking order)
    ├─ Hyperparameters: batch=4, lr=0.001, hidden=64
    └─ Best model: ranking_model_best_Config_01_Small.pkl
    ↓
eval_ranking_models.py (rigorous evaluation)
    ├─ NDCG@5, Spearman correlation
    ├─ P@5 (top-5 accuracy)
    └─ vs random baseline (+827%)
    ↓
app.py (Flask API)
    └─ Predictions + visualizations
```

### Model Architecture (Ranking)

```
RankingModel (PyTorch MLP)
├─ Input: 26D features
├─ Layer 1: Linear(26 → 64) + ReLU
├─ Layer 2: Linear(64 → 64) + ReLU
├─ Layer 3: Linear(64 → 1) + Sigmoid
└─ Loss: PairwiseLoss (ranking optimization)

Training Config (Best):
├─ Batch Size: 4
├─ Learning Rate: 0.001
├─ Hidden Dim: 64
├─ Epochs: ~9 (fast convergence)
└─ Runtime: 1.6s for full train
```

---

## 📊 Comparison: ST-GCN vs Ranking Model

| Metric | ST-GCN | Ranking v2 | Improvement |
|--------|--------|-----------|------------|
| P@5 | 0.15 | 1.00 | +566% |
| NDCG@5 | 0.22 | 0.9995 | +354% |
| Spearman ρ | 0.35 | 0.9766 | +179% |
| Training Time | 60 epochs | 9 epochs | 7× faster |
| Optimization | MSE (value) | Pairwise (rank) | Direct ranking |
| Loss Type | Regression | Classification | Ranking-aware |

**Key Insight**: ST-GCN optimizes value prediction (MSE), but crime ranking doesn't need exact values—only relative order. Ranking loss directly optimizes what matters.

---

## 🎯 Phase 1 Results (Hyperparameter Search)

### Configuration Performance
- **Config_01_Small** (WINNER): batch=4, lr=0.001, hidden=64
  - NDCG@5: 0.9995 (99.95% of ideal)
  - P@5: 1.0000 (100% top-5 accuracy)
  - Training: 1.6s

- **Config_02-05, 07-12** (7 others): batch=[8,16,32], lr=0.001-0.005, hidden=[128,256]
  - All achieved P@5 ≥ 0.8-1.0
  - Faster training (0.1-0.5s) but same ranking quality

- **Config_06** (lr too high): batch=16, lr=0.01, hidden=128
  - P@5: 0.8000 (learning rate too aggressive)

### Key Finding
11 out of 12 configs converged to near-perfect performance. **Architecture is robust**, not sensitive to hyperparameters (unlike ST-GCN which required careful tuning).

---

## 💾 Model Files

### Production Models
- ✅ `models/ranking_model_best_Config_01_Small.pkl` - Best (NDCG@5=0.9995)
- ✅ `models/ranking_model_best_Config_02_SmallLR.pkl` - Alt (same quality, faster)
- ✅ `models/stgcn_model_v2.pth` - Original ST-GCN (reference)

### Deprecated Models
- ❌ `models/ranking_model_v1.pkl` - ListNet v1 (deleted, P@5≈0)
- ❌ `models/ranking_model_v2.pkl` - Old v2 checkpoint (replaced by config version)
- ❌ `models/stgcn_model.pth` - Very old (deleted)
- ❌ `models/stgcn_model_v3.pth` - Intermediate (deleted)

---

## 📈 Data Structure (Phase 1)

### Dataset
- **Neighborhoods**: 319 total
- **Active (CVLI > 0)**: ~15-20 neighborhoods
- **Timesteps**: 1491 days (2022-01-01 to 2026-01-21)
- **Features**: 26 channels per node per timestep

### Feature Channels
```
Channels 0-2:   Crime metrics (CVLI, CVP, Tension)
Channels 3-9:   Day of week (7D one-hot)
Channels 10-21: Month of year (12D one-hot)
Channel 22:     Is-weekend flag
Channels 23-25: Reserved for future expansion
```

### Example Record
```json
{
  "node_id": 146,
  "timestep": 500,
  "features": [
    42.5,  // CVLI
    28.3,  // CVP  
    15.2,  // Tension
    // One-hot day-of-week (Sunday = 1)
    1, 0, 0, 0, 0, 0, 0,
    // One-hot month (January = 1)
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Weekend flag
    1,
    // Reserved
    0, 0, 0
  ]
}
```

---

## 🔄 Phase 2 Preview: LLM Semantic Features

**Goal**: Add semantic embeddings from neighborhood descriptions (e.g., location, demographics, facilities) to improve generalization.

**Planned Additions**:
- Google Generative AI embeddings for each neighborhood (using `llm_service.py`)
- Merge embeddings with existing 26D features (→ 26+384=410D)
- Retrain with combined features
- Expected: Maintain P@5≈1.0 with better cross-validation

**Timeline**: 3-4 days

---

## 🔧 Utility Scripts

### In `/scripts/` (mostly legacy, kept for reference)
- `check_new_data.py` - Data freshness checks
- `merge_and_retrain.py` - Batch processing pipeline
- `test_cvli_accuracy.py` - Data validation
- Others for specific analyses during development

### In `/tests/`
- `test_model_viability.py` - Model validation
- `test_simulation.py` - Prediction simulation
- `test_api.py` - API endpoint testing

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `RANKING_PROOF_OF_CONCEPT.md` | Initial POC showing RankingLoss beats ST-GCN |
| `PHASE1_FINAL_REPORT.md` | Grid search results + insights |
| `PHASE1_PROGRESS.md` | Timeline and expectations |
| `ARCHITECTURE_REFERENCE.md` | This file - current state |

---

## ✅ Cleanup Checklist (Completed)

- ✅ Deleted `src/ranking_model.py` (v1 ListNet)
- ✅ Deleted `train_ranking.py` (v1 training)
- ✅ Deleted `models/ranking_model_v1.pkl` (old model)
- ✅ Deleted debug scripts: `test_categorical_features.py`, `debug_train.py`, `analyze_tuning.py`, `hybrid_analysis.py`, `prove_obvious_patterns.py`
- ✅ Deleted old models: `stgcn_model.pth`, `stgcn_model_v3.pth`, `ranking_model_best.pth`, `ranking_model_v2_best.pth`
- ✅ Created clean ARCHITECTURE_REFERENCE.md (this file)

---

## 🚦 Next Steps

1. **Immediate**: Use this architecture for Phase 2 development (LLM features)
2. **Reference**: Link to this doc in all new code
3. **Maintenance**: Update this file when adding new components
4. **Deprecation**: Move legacy scripts to `/deprecated/` if needed later

---

## 📞 Key Contacts / References

- **Best Model**: `models/ranking_model_best_Config_01_Small.pkl`
- **Training Script**: `train_ranking_v2.py`
- **Evaluation**: `eval_ranking_models.py`
- **API**: `app.py` (runs on port 5000)
- **Data**: `data/processed/processed_graph_data.pkl`

