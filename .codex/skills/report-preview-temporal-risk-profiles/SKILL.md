---
name: report-preview-temporal-risk-profiles
description: >
  Use this skill when changing Report Preview risk prediction, dashboard, or
  explainability code that surfaces CVLI peak weekday/hour patterns. It preserves
  live prediction-time temporal calculation, startup/export side effects, and
  encoding correctness without hiding emoji or removing logs.
license: MIT
metadata:
  author: codex
  version: "1.0"
---

# Report Preview Temporal Risk Profiles

Use this for the "dia da semana e faixa de horario de pico" workflow in Report
Preview. The correct path is to calculate temporal CVLI profiles from real
`data/hora/bairro` data when risk is predicted, then propagate the fields through
API, explainability, and dashboard.

**Failure pattern:** treating the temporal profile as a startup cache or fixing
terminal mojibake by stripping emoji/ASCII-sanitizing logs hides the real issue
and can remove required startup checks/exports.
**Verified by:** `python -m py_compile app.py src\explanation_generator.py`,
HTML script parse with Node, and direct `_build_predictive_temporal_profiles`
assertions showing profiles with weekday, hour range, and sample size.

## When To Use This

- The user asks for peak day/hour, horario de risco, temporal pattern, or more
  specific predictive risk timing in Report Preview.
- The change touches `/api/risk`, `/api/explain/<node_id>`, `templates/index.html`,
  startup validation, or temporal CVLI logic.
- Encoding/log output looks wrong during this workflow.

## Procedure

- [ ] 1. Read the current flow before editing: `_compute_peak_hours_cache`,
      `/api/risk`, `/api/explain/<node_id>`, and dashboard consumers of
      `peak_hours`.
- [ ] 2. Keep startup validation/export threads in place. If "no cache" is
      required, make startup a validation/warmup only; make `/api/risk` compute
      the temporal profile during the request.
- [ ] 3. Build temporal profiles from the raw enriched CSV using real `data`,
      `hora`, `bairro`, and `tipo == cvli`. Do not invent fallback hours.
- [ ] 4. Return `peak_hours`, `peak_weekday`, `peak_time_label`, hour bounds,
      shares, sample size, reference date, and horizon days when there is enough
      local support; otherwise omit the temporal factor.
- [ ] 5. Propagate the same fields to node metrics, top10 metadata, explain
      payload, and dashboard labels.
- [ ] 6. Validate with the smallest checks:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py src\explanation_generator.py
@'
const fs = require('fs');
const html = fs.readFileSync('templates/index.html', 'utf8');
const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/gi)].map(m => m[1]).filter(s => s.trim());
for (let i = 0; i < scripts.length; i++) new Function(scripts[i]);
console.log('OK', scripts.length);
'@ | node -
```

## Gotchas

- Do not solve emoji/mojibake by stripping emoji or forcing ASCII logs. Emoji is
  valid; validate terminal encoding and only change encoding at the source.
- Do not remove startup checks, validation logs, Crime-Predict export, or
  background jobs while changing temporal risk logic.
- Keep historical feature windows such as 7d/14d/30d distinct from predictive
  horizon. Rename only what the code truly changes.

## What Didn't Work

- Replacing logs with ASCII-safe text removed symptoms but not the encoding
  cause.
- Removing the startup temporal cache thread entirely violated the required
  startup verification/export behavior.
- Broad patches against mojibake-heavy comments are brittle; use structural
  anchors and immediately compile-check.
