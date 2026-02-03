# ST-GCN Crime Ranking - Crime Prediction via Ranking Loss

**Status**: 🚀 Fase 1 Completo | 📊 Fase 2 Em Progresso
**Data Atualização**: 03/02/2026

---

## 📈 Arquitetura Atual

### Fase 1 ✅ Completa: Pairwise Ranking (NDCG@5=0.9995)

**Problema Identificado**: ST-GCN com MSE otimiza valor de crime, não ranking.
**Solução**: Implementar Pairwise Ranking Loss - direto otimiza order.
**Resultado**: +566% melhoria vs ST-GCN original

| Métrica | ST-GCN (MSE) | Ranking v2 | Melhoria |
|---------|-------------|-----------|---------|
| P@5 | 0.15 | 1.00 | +566% |
| NDCG@5 | 0.22 | 0.9995 | +354% |
| Spearman ρ | 0.35 | 0.9766 | +179% |
| Training | 60 epochs | 9 epochs | 7× faster |

**Arquivos Principais**:
- `src/ranking_model_v2.py` - PairwiseLoss + MLP
- `train_ranking_v2.py` - Training pipeline
- `eval_ranking_models.py` - Rigorous evaluation
- `models/ranking_model_best_Config_01_Small.pkl` - Best model

---

### Fase 2 🚀 Em Progresso: Semantic Embeddings (410D)

**Objetivo**: Expandir features com embeddings semânticos dos bairros.
- Adicionar 384D embeddings do Google Generative AI
- Combinar com 26D features existentes → 410D total
- Validar: Manter P@5≈1.0 + Melhor generalização

**Estrutura**:
- `FASE2_INTEGRACAO_SEMANTICA.md` - Arquitetura integrada com LLM
- `FASE2_CHECKLIST.md` - Checklist detalhado de implementação
- **⚠️ INTEGRADO**: Função em `llm_service.py` (não solta no ar)

**Timeline**: 3-4 dias
**Próximo Check**: Validar com dados Jan/2026 (fora do período de treino)

---

## 🚀 Quick Start

### Treinar Melhor Modelo (Fase 1)
```bash
python train_ranking_v2.py
```

### Avaliar com Métricas Rigorosas
```bash
python eval_ranking_models.py
```

### API Flask
```bash
python app.py  # http://localhost:5000
```

---

## 📁 Estrutura Organizada

**Production Code** (`/src`):
```
ranking_model_v2.py       - PairwiseLoss architecture
ranking_features.py       - 26D feature extraction
data_processing.py        - Data pipeline
llm_service.py           - LLM integration (CIOPS + Semantic)
```

**Training Scripts** (root):
```
train_ranking_v2.py      - Main training
hyperparam_search.py     - Grid search (12 configs)
eval_ranking_models.py   - Evaluation with NDCG@5
app.py                   - Flask API
```

**Documentation**:
```
ARCHITECTURE_REFERENCE.md       - System architecture
PHASE1_FINAL_REPORT.md         - Phase 1 results
FASE2_INTEGRACAO_SEMANTICA.md  - Phase 2 design
FASE2_CHECKLIST.md             - Phase 2 tasks
```

---

## 🔄 Data Flow

```
Raw Crime Data
    ↓
data_processing.py (normalize)
    ↓
ranking_features.py (26D features)
    ├─ CVLI, CVP, Tension
    ├─ Day-of-week, Month
    └─ Weekend flag
    ↓
[PHASE 2] llm_service.py (384D embeddings)
    ↓
[PHASE 2] expand 26D → 410D
    ↓
train_ranking_v2.py (PairwiseLoss)
    ↓
ranking_model_best.pkl
    ↓
eval_ranking_models.py
    ↓
app.py (API)
```

---

## 💾 Best Model

- **File**: `models/ranking_model_best_Config_01_Small.pkl`
- **NDCG@5**: 0.9995 (99.95% ideal)
- **P@5**: 1.0000 (100% top-5 accuracy)
- **Spearman**: 0.9766 (excellent ranking correlation)

---

## 📚 Documentação

- [ARCHITECTURE_REFERENCE.md](ARCHITECTURE_REFERENCE.md) - System overview
- [PHASE1_FINAL_REPORT.md](PHASE1_FINAL_REPORT.md) - Grid search results
- [FASE2_INTEGRACAO_SEMANTICA.md](FASE2_INTEGRACAO_SEMANTICA.md) - Phase 2 technical
- [FASE2_CHECKLIST.md](FASE2_CHECKLIST.md) - Implementation checklist

