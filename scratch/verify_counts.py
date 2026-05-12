"""Verify: quantos bairros unicos de Fortaleza existem no pkl e quantos caem em cada faixa de risco."""
import json, os, sys, unicodedata, re, pickle, numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

def normalize_name(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
    return re.sub(r'\s*[-\u2013(]?\s*AIS.*$', '', text).strip()

def normalize_risk_score(score):
    return max(0.0, min(100.0, float(score)))

pkl_path = os.path.join(BASE, 'data', 'processed', 'processed_fortaleza.pkl')
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data['nodes_gdf']
scores = data.get('scores', None)

print(f"Total nos: {len(nodes_gdf)}")

# Deduplicar por nome normalizado e contar
seen = {}
for i, row in nodes_gdf.iterrows():
    name = normalize_name(str(row['name']))
    if name not in seen:
        if scores is not None:
            try:
                score = float(scores[i]) if i < len(scores) else 20.0
            except:
                score = 20.0
        else:
            score = 20.0
        seen[name] = normalize_risk_score(score)

print(f"Nos unicos (por nome normalizado): {len(seen)}")

critico = sum(1 for s in seen.values() if s >= 71)
alto = sum(1 for s in seen.values() if 51 <= s < 71)
moderado = sum(1 for s in seen.values() if 31 <= s < 51)
baixo = sum(1 for s in seen.values() if s < 31)

print(f"\nDistribuicao de risco (deduplicado):")
print(f"  Critico (>=71): {critico}")
print(f"  Alto (51-70):   {alto}")
print(f"  Moderado (31-50): {moderado}")
print(f"  Baixo (<31):    {baixo}")
print(f"  Total:          {critico + alto + moderado + baixo}")

# Mostrar os moderados
print(f"\nBairros MODERADOS:")
for name, score in sorted(seen.items(), key=lambda x: -x[1]):
    if 31 <= score < 51:
        print(f"  {name}: {score:.1f}")
