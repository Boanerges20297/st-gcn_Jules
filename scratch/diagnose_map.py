"""Diagnose: quantos poligonos do GeoJSON conseguem ser mapeados para dados de risco do backend."""
import json, os, sys, unicodedata, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

def normalize_name(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
    return re.sub(r'\s*[-\u2013(]?\s*AIS.*$', '', text).strip()

# 1) Nomes dos poligonos no GeoJSON
geojson_path = os.path.join(BASE, 'data', 'static', 'AIS - CAPITAL.geojson')
with open(geojson_path, encoding='utf-8') as f:
    geo = json.load(f)

poly_names = []
for feat in geo['features']:
    raw = feat['properties'].get('Name', '')
    norm = normalize_name(raw)
    poly_names.append((raw, norm))

print(f"=== GeoJSON AIS-CAPITAL: {len(poly_names)} poligonos ===")
unique_norms = set(n for _, n in poly_names)
print(f"Nomes unicos (normalizados): {len(unique_norms)}")
print(f"Amostras: {sorted(list(unique_norms))[:10]}")

# 2) Nomes dos nos do orchestrator
import pickle
pkl_path = os.path.join(BASE, 'data', 'processed', 'processed_fortaleza.pkl')
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data.get('nodes_gdf')
if nodes_gdf is not None:
    backend_names = set()
    for _, row in nodes_gdf.iterrows():
        backend_names.add(normalize_name(str(row['name'])))
    print(f"\n=== Backend nodes_gdf (Fortaleza): {len(nodes_gdf)} nos, {len(backend_names)} unicos ===")
    print(f"Amostras: {sorted(list(backend_names))[:10]}")
    
    # 3) Intersecao
    matched = unique_norms & backend_names
    unmatched_geo = unique_norms - backend_names
    unmatched_backend = backend_names - unique_norms
    
    print(f"\n[OK] Match: {len(matched)} poligonos com dados de risco")
    print(f"[FAIL] SEM match (poligono existe mas sem dados): {len(unmatched_geo)}")
    if unmatched_geo:
        for name in sorted(unmatched_geo)[:30]:
            print(f"   - {name}")
    
    print(f"\n[WARN] Backend tem dados mas sem poligono: {len(unmatched_backend)}")
    if unmatched_backend:
        for name in sorted(unmatched_backend)[:30]:
            print(f"   - {name}")
else:
    print("ERROR: nodes_gdf not found in pkl!")
    print("Keys:", list(data.keys()))
