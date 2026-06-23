"""Retroativamente enriquece os archives de eventos exogenos com campos shock_*."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.enrichment import _compute_shock_fields  # noqa: E402

archives = sorted((ROOT / "data" / "archives").glob("exogenous_events_*.json"))
total_updated = 0

for f in archives:
    try:
        events = json.loads(f.read_text(encoding="utf-8"))
        changed = False
        for ev in events:
            if "shock_is_conflict" not in ev:
                ev.update(_compute_shock_fields(ev))
                changed = True
        if changed:
            f.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
            total_updated += len(events)
    except Exception as e:
        print(f"Erro em {f.name}: {e}")

print(f"Eventos enriquecidos: {total_updated} em {len(archives)} arquivos")

# Amostra do resultado
sample_f = archives[4]
sample = json.loads(sample_f.read_text(encoding="utf-8"))
ev = sample[0]
shock_fields = {k: v for k, v in ev.items() if k.startswith("shock_")}
print(f"\nExemplo: natureza={ev.get('natureza')} | severity={ev.get('conflict_severity')}")
print(f"shock_fields: {shock_fields}")
