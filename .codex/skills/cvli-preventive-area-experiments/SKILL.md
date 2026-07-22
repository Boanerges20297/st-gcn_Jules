---
name: cvli-preventive-area-experiments
description: >
  Use this skill when working on Report Preview CVLI research experiments for Fortaleza: comparing bairro rankings, spatial greedy, honeycomb/hex/circle overlays, GA optimization, dashboard wording, and dissertation notes. Use it especially when the user wants operational prevention areas, not deterministic policing orders.
license: MIT
metadata:
  author: codex
  version: "1.0"
---

# CVLI Preventive Area Experiments

This captures the proven workflow for CVLI spatial experiments in this repo: keep bairro prediction as the strategic layer, test spatial refinements empirically, and document both gains and useful negative results.

**Failure pattern:** visual/geometric refinements such as hexagons can look precise but fail to improve future CVLI capture, creating false operational confidence.
**Verified by:** repeated runs of `scripts/experiments/fortaleza_hybrid_capture_spike.py` with `py_compile`, endpoint checks, JS parse checks, and generated summaries under `outputs/experiments/*_latest_*.csv`.

## When to use this

- The user asks to improve where to allocate preventive policing for CVLI while preserving academic rigor.
- The user proposes bairros, hexagons, circles, GA, greedy spatial selection, capture/area trade-offs, or dashboard map overlays.
- The user criticizes a visual layer as operationally misleading or too deterministic.
- The task needs dissertation-ready documentation of empirical gains or negative results.

## Procedure

- [ ] 1. Keep the frame non-deterministic: write “areas of higher estimated CVLI risk” or “evidence for operational assessment”, not “determine where to police”.
- [ ] 2. Treat bairro ranking as the strategic baseline. Always compare against top-k bairro capture before promoting any spatial overlay.
- [ ] 3. Use the existing experiment script first:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\fortaleza_hybrid_capture_spike.py --horizon 30 --k 20 30 --eligible-top 30 --spatial-cells 20 30 40 --spatial-radius-km 1.0 1.5 2.0 --run-ga --ga-objective all --zone-shape hex
```

- [ ] 4. Keep generated outputs under `outputs/experiments/`; this path is ignored by Git. Do not add timestamped output spam unless `--keep-history` is explicitly needed.
- [ ] 5. Judge methods by future capture and operational area:
  - bairro top-k: strongest strategic baseline;
  - spatial greedy: may capture well but can require very large areas;
  - honeycomb/hex GA: only promote if it beats bairro or gives a clearly superior capture/area trade-off;
  - circle overlay: use as historical evidence inside predicted bairros, not as a claim of exact prediction.
- [ ] 6. Document results in `research/dissertacao_descobertas/` only after checking numbers. Document negative results too when they change the methodological direction.
- [ ] 7. For dashboard wording, prefer “Bairros de maior risco estimado” and “Evidência histórica local”. Avoid “Zonas GA” in operational UI.

## Checks

Run the smallest relevant checks:

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\experiments\fortaleza_hybrid_capture_spike.py app.py
```

For dashboard JS edits:

```powershell
@'
const fs = require('fs');
const html = fs.readFileSync('templates/index.html', 'utf8');
const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/gi)].map(m => m[1]).filter(s => s.trim());
for (let i = 0; i < scripts.length; i++) new Function(scripts[i]);
console.log('OK', scripts.length);
'@ | node -
```

For API payload:

```powershell
@'
from app import app
with app.test_client() as c:
    r = c.get('/api/ga_operational_zones')
    data = r.get_json()
    print(r.status_code, len(data.get('features', [])))
'@ | .\.venv\Scripts\python.exe -
```

## Gotchas

- Do not present hexagons as a gain because they look clean. In this session, bairro top 30 reached about 89% capture while honeycomb GA variants stayed far lower.
- Do not let historical CVLI points directly define the predicted area unless the layer is explicitly framed as historical evidence.
- A full honeycomb must use fixed resolution per layer. Variable per-cell hex radius creates gaps and is not a real closed honeycomb.
- The user values negative results for the dissertation when they are empirically grounded.

## What didn't work

- Hexagon/honeycomb GA as a replacement for bairro ranking: visually cleaner, but no real capture gain.
- Adaptive hex radius by bairro size: improved visual logic but reduced capture.
- Score inherited only from bairro: gives the GA little real local signal.
- Overlapping circles/hexes: operationally confusing, even when capture improves.
