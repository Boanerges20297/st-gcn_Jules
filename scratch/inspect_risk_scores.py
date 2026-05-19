import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import app
from src.core.orchestrator import normalize_name

app.load_data_and_models()
exogenous_shocks, _ = app.build_current_exogenous_shocks()
scores_map, _ = app.orchestrator.get_combined_risk(exogenous_shocks, return_trends=True)

print("Total risk scores keys:", len(scores_map))
caninde_keys = [k for k in scores_map.keys() if 'CANINDE' in k.upper()]
print("Keys with 'CANINDE' in risk_scores:", caninde_keys)

# Check if there is CANINDE in nodes_gdf for any specialist
for reg_key, spec in app.orchestrator.specialists.items():
    print(f"\nSpecialist: {reg_key}")
    nodes_gdf = spec['data']['nodes_gdf']
    matches = nodes_gdf[nodes_gdf['name'].str.upper().str.contains('CANINDE', na=False)]
    if len(matches) > 0:
        print(matches[['name', 'regiao']])
    else:
        print("No matches")
