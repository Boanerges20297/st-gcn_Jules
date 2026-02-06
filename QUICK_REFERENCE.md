# 🗂️ QUICK REFERENCE - PHASE 2B IMPLEMENTATION

**Keep this document open while coding. It's your cheat-sheet.**

---

## 📅 TIMELINE AT A GLANCE

```
WEEK 1  (Feb 7-13):   [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] Metrics Setup
WEEK 2  (Feb 14-20):  [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] Events Integration
WEEK 3  (Feb 21-27):  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] Long-tail Optimization
WEEK 4  (Feb 28-6):   [████████████████░░░░░░░░░░░░░░░░░░░░░░░░] Explainability
WEEK 5  (Mar 7-13):   [████████████████████░░░░░░░░░░░░░░░░░░░░] Testing & Deploy
```

---

## 🎯 YOUR MISSION THIS WEEK

### WEEK 1 (Starting Feb 7)

**Goal**: Validate metrics + understand current gaps

| Task | Hours | File | Done? |
|------|-------|------|-------|
| 1.1: Implement P@5-20 metrics | 8h | `src/metrics.py` | ⬜ |
| 1.2: Calculate baseline | 4h | `baseline_metrics.json` | ⬜ |
| 1.3: Analyze long-tail gaps | 4h | `long_tail_analysis.json` | ⬜ |
| **TOTAL WEEK 1** | **16h** | | |

**Checkpoint (Fri Feb 13):**
- [ ] metrics.py is tested and works
- [ ] baseline_metrics.json exists with P@5-20 values
- [ ] long_tail_analysis.json shows gaps
- [ ] You understand where improvement can come

**If stuck:**
- metrics.py? → Check `src/ranking_pure_seasonality.py` for inspiration
- baseline? → Run on test set (days 1350-1491)
- analysis? → Use `np.argsort(-y_pred)` to get ranks

---

### WEEK 2 (Feb 14-20)

**Goal**: Integrate events + make model aware

| Task | Hours | File | Done? |
|------|-------|------|-------|
| 2.1: Event anomaly detector | 6h | `src/event_anomaly_detector.py` | ⬜ |
| 2.2: Event manager (load + index) | 4h | `src/event_manager.py` | ⬜ |
| 2.3: Modify RankingModel | 6h | `src/ranking_model.py` | ⬜ |
| 2.4: Train with anomaly awareness | 8h | `scripts/train_with_anomaly_awareness.py` | ⬜ |
| **TOTAL WEEK 2** | **24h** | | |

**Checkpoint (Fri Feb 20):**
- [ ] EventAnomalyDetector parses events correctly (test on 5 examples)
- [ ] EventManager loads `exogenous_events_geocoded.json`
- [ ] RankingModel accepts anomaly_level parameter
- [ ] Model trains without errors
- [ ] Model metrics (P@5-20) calculated

**Output**: `models/ranking_model_with_anomaly.pkl`

---

### WEEK 3 (Feb 21-27)

**Goal**: Improve P@20 coverage

| Task | Hours | File | Done? |
|------|-------|------|-------|
| 3.1: Error analysis (why missed?) | 4h | `error_analysis.json` | ⬜ |
| 3.2: Implement loss functions | 8h | `src/loss_functions.py` | ⬜ |
| 3.3: Train with combined loss | 8h | `scripts/train_for_p20_coverage.py` | ⬜ |
| 3.4: Compare 3 models | 4h | `comparison_report.json` | ⬜ |
| **TOTAL WEEK 3** | **24h** | | |

**Checkpoint (Fri Feb 27):**
- [ ] Error analysis shows why long-tail is missed
- [ ] Combined loss (0.5 P@5 + 0.5 P@20) implemented
- [ ] Model trains with new loss
- [ ] **DECISION**: Which model variant to move forward?
  - Baseline? (safest)
  - With anomaly? (stable)
  - With P@20 focus? (ambitious)

**Output**: 
- 3 trained models
- Comparison metrics
- Decision on which to deploy

---

### WEEK 4 (Feb 28 - Mar 6)

**Goal**: Add explanations (prep for doutorado)

| Task | Hours | File | Done? |
|------|-------|------|-------|
| 4.1: Explanation generator | 6h | `src/explanation_generator.py` | ⬜ |
| 4.2: API endpoints | 6h | `src/app.py` (MODIFY) | ⬜ |
| 4.3: Dashboard update | 4h | `templates/dashboard.html` | ⬜ |
| 4.4: Documentation | 4h | `docs/EXPLAINABILITY_GUIDE.md` | ⬜ |
| **TOTAL WEEK 4** | **20h** | | |

**Checkpoint (Wed Mar 6):**
- [ ] ExplanationGenerator creates sensible explanations
- [ ] `/explain/{node_id}` API endpoint works
- [ ] `/metrics` endpoint returns P@5-20, NDCG
- [ ] `/anomaly_status` endpoint works
- [ ] Dashboard shows new sections

**Output**: Enhanced system with explanations

---

### WEEK 5 (Mar 7-13)

**Goal**: Test everything + deploy

| Task | Hours | File | Done? |
|------|-------|------|-------|
| 5.1: Comprehensive tests | 6h | `tests/test_enhanced_system.py` | ⬜ |
| 5.2: End-to-end validation | 4h | `scripts/final_validation.py` | ⬜ |
| 5.3: Deployment prep | 6h | `scripts/deploy.py` | ⬜ |
| 5.4: Documentation | 4h | `docs/DEPLOYMENT_GUIDE.md` | ⬜ |
| **TOTAL WEEK 5** | **20h** | | |

**Final Checkpoint (Thu Mar 13):**
- [ ] All tests pass
- [ ] End-to-end validation shows:
  - P@5 ≥ 0.78 ✅
  - P@20 ≥ 0.55 ✅
  - NDCG@5 ≥ 0.92 ✅
- [ ] Deployment script tested on dry-run
- [ ] **GO/NO-GO DECISION**

**Output**: Production-ready system or rollback plan

---

## 🔑 KEY CONCEPTS TO REMEMBER

### P@K (Precision@K)
```python
# How many of top-K predicted are actually top-K real?
real_top_k = set(np.argsort(-y_true)[:k])
pred_top_k = set(np.argsort(-y_pred)[:k])
p_at_k = len(real_top_k & pred_top_k) / k
# Range: 0.0 to 1.0 (1.0 = perfect)
```

### NDCG@K
```python
# How good is the ranking quality?
# Rewards correct ordering, not just correctness
# Example: Rank [1,2,3,4,5] vs [5,4,3,2,1]
# P@5 = 1.0 for both, but NDCG @5 = 1.0 vs 0.1
```

### Anomaly Flag
```python
# Event happening today → anomaly_flag = True
# Model confidence drops: 1.0 - (anomaly_level * 0.3)
# Example: High event (severity 0.8) → confidence 0.76
```

---

## 📝 DAILY CODING CHECKLIST

When starting each day:

1. **What did I complete yesterday?**
   - [ ] Update ROADMAP status
   - [ ] Commit code: `git add . && git commit -m "Week X TaskX: [done]"`

2. **What am I working on today?**
   - [ ] Pick ONE task
   - [ ] Estimate hours (realistic!)
   - [ ] Create branch: `git checkout -b feature/week_X_task_Y`

3. **Am I blocked?**
   - [ ] Any errors? Search `tests/` or run existing examples
   - [ ] Missing data? Check `data/` structure
   - [ ] Design question? Review docs first

4. **Before pushing:**
   - [ ] Code runs without errors
   - [ ] New file is documented (docstrings)
   - [ ] Tested on sample data
   - [ ] Committed with clear message

---

## 🚨 IF YOU GET STUCK

### Metrics not calculating?
→ Look at `src/ranking_pure_seasonality.py` line 110-150  
→ Copy-paste the pattern, adapt to your metrics

### Event parsing fails?
→ Start with simple keyword matching (heuristics)  
→ DON'T jump to LLM yet
→ Test on 5 real events from `data/exogenous_events_geocoded.json`

### Model training is slow?
→ Use smaller dataset first (last 200 days instead of 1491)
→ Reduce model size (128 hidden → 64)
→ Use GPU if available: `device = torch.device('cuda')`

### Metrics don't improve?
→ Check if P@5 is regressing (you hurt it!)
→ Review error_analysis.json - where are failures?
→ Pivot: Maybe P@20 can't improve, focus on stability instead

### Test failures?
→ Run tests individually: `python -m pytest tests/test_enhanced_system.py::TestEnhancedSystem::test_metrics_calculation`
→ Use `-v` flag for verbose: `pytest -v`

---

## 📊 QUICK METRICS REFERENCE

| Metric | Current | Target | If Miss |
|--------|---------|--------|---------|
| P@5 | 0.80 | ≥0.78 | Kill the change! |
| P@10 | 0.65 | ≥0.65 | OK if P@5 stable |
| P@20 | 0.50 | ≥0.55 | Main goal of Week 3 |
| NDCG@5 | 0.92 | ≥0.92 | Quality metric |
| NDCG@20 | 0.75 | ≥0.76 | Stretch goal |

---

## 🎯 DEFINITION OF DONE (Per Task)

Task is DONE when:
```
✅ Code is written + tested on sample data
✅ No console errors (all imports work)
✅ Produces output file/metrics as expected
✅ Docstring explains what it does
✅ Committed to git with clear message
✅ Checked into team (peer review if possible)
```

---

## 🚀 DEPLOYMENT READINESS

Before deploying (Week 5):

- [ ] All 5 weeks complete
- [ ] metrics: P@5 ≥ 0.78, P@20 ≥ 0.55
- [ ] Test suite passes 100%
- [ ] No console warnings
- [ ] Documentation reviewed
- [ ] Stakeholder approval obtained
- [ ] Rollback plan documented

---

## 💾 GIT WORKFLOW

```bash
# Start feature
git checkout -b feature/week1_task1

# During development
git add .
git commit -m "Week 1 Task 1.1: Implement P@K metrics"

# When group (ready to merge)
git push origin feature/week1_task1
git pull request
# (Code review + merge)

# Go back to main
git checkout main
git pull
# (Start next feature branch)
```

---

## 📞 WHO TO ASK

| Issue | Who | Where |
|-------|-----|-------|
| Architecture question | Tech Lead | ROADMAP doc section1 |
| Data question | Data owner | `data/` directory + docs |
| Metrics formula | ML engineer | `src/ranking_pure_seasonality.py` |
| API design | Backend | `src/app.py` existing endpoints |
| Deployment | DevOps | `scripts/deploy.py` template |

---

## 🎓 LEARNING RESOURCES (Inside repo)

```
docs/README.md                          ← Full system understanding
docs/TECHNICAL_SUMMARY.md               ← ST-GCN + RankingModel details
src/ranking_pure_seasonality.py         ← Example metric implementation
src/train.py                            ← Training loop pattern
scripts/                                ← Examples of evaluation scripts
```

**REMEMBER**: 80% of answers are in existing code. Check there FIRST!

---

## 📝 YOUR WEEKLY PROGRESS TRACKER

Save this to `PROGRESS.md` in repo root. Update every Friday.

```markdown
## WEEK 1 Progress
- [x] metrics.py written
- [x] baseline calculated  
- [x] long_tail analysis done
- [ ] Code reviewed
- Status: On track / Behind / Blocked

## WEEK 2 Progress  
(Same structure)

## WEEK 3 Progress
(Same structure)

## WEEK 4 Progress
(Same structure)

## WEEK 5 Progress
(Same structure)
```

---

**FINAL THOUGHT**: You've got 5 weeks and a solid plan. Focus on ONE task at a time. Each task is concrete and achievable. Trust the process!

💪 **You've got this!**
