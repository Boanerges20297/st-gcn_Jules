---
name: numpy-pickle-compat
description: >
  Use this skill when Python pickle/joblib/scikit-learn model artifacts in this
  project fail to load with "No module named 'numpy._core...'", especially for
  regional Poisson payloads or processed model files after changing NumPy
  versions. Diagnose whether the failing artifact is data or model payload, then
  patch the shared loader with a NumPy module alias instead of regenerating all
  artifacts first.
license: MIT
metadata:
  author: codex
  version: "1.0"
---

# NumPy Pickle Compatibility

This captures the project path for artifacts pickled with NumPy 2 module names
and loaded under NumPy 1.x.

**Failure pattern:** `ModuleNotFoundError: No module named 'numpy._core.numeric'`
while loading regional artifacts such as `fortaleza`, `rmf`, or `interior`.
**Verified by:** `load_payload()` loaded all three Poisson regressors and
`StateOrchestrator(str(Path.cwd()))` initialized all Poisson and ST-GAT
specialists.

## When To Use This

- A model/data load prints `Erro ao carregar <region>: No module named 'numpy._core...`.
- The failing path involves `models/active/production/poisson/*.pkl` or
  `data/processed/processed_*.pkl`.

## Procedure

- [ ] 1. Reproduce separately for data and model payloads. In this project,
      `_load_pickle_safe()` already handles processed data in
      `src/core/orchestrator.py`; the Poisson models route through
      `load_payload()` in `src/core/fortaleza_poisson_backend.py`.
- [ ] 2. If the model payload fails, patch the shared loader, not each regional
      caller. Use a custom `pickle.Unpickler.find_class()` that rewrites
      `numpy._core` to `numpy.core`.
- [ ] 3. Verify with all regional payloads and then instantiate
      `StateOrchestrator`. On Windows PowerShell, set
      `$env:PYTHONIOENCODING='utf-8'` before printing orchestrator logs.

### Example

```powershell
$env:PYTHONIOENCODING='utf-8'
@'
from pathlib import Path
from src.core.orchestrator import StateOrchestrator

orch = StateOrchestrator(str(Path.cwd()))
print(sorted(orch.specialists))
'@ | .\.venv\Scripts\python.exe -
```

Expected: `['fortaleza', 'interior', 'rmf']`.

## Gotchas

- The processed data can load successfully while the Poisson model payloads fail;
  test both before editing.
- scikit-learn may warn about version mismatch after the NumPy alias fix. That is
  a separate compatibility risk, not the `numpy._core.numeric` load failure.

## What Didn't Work

- Forcing the repo back to `origin/main` did not fix this because the committed
  Poisson `.pkl` artifacts still reference NumPy 2 module paths.
- Testing only `data/processed/processed_*.pkl` was a dead end; those files
  already passed through `_load_pickle_safe()`.
