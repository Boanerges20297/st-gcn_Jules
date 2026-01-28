import pickle
import numpy as np
from pathlib import Path
p = Path('data/processed/processed_graph_data.pkl')
if not p.exists():
    raise SystemExit('processed_graph_data.pkl not found')
with p.open('rb') as f:
    data = pickle.load(f)
adj_geo = np.array(data['adj_geo'])
adj_faction = np.array(data['adj_faction'])

def summarize(adj, name):
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    edges = int((adj>0).sum()//2)
    density = edges / (n*(n-1)/2)
    print(f"{name} summary:")
    print(f"  nodes: {n}")
    print(f"  edges (undirected): {edges}")
    print(f"  degree mean: {deg.mean():.2f}")
    print(f"  degree median: {np.median(deg):.2f}")
    print(f"  degree max: {deg.max():.0f}")
    print(f"  density: {density:.6f}")
    print('')

summarize(adj_geo, 'adj_geo')
summarize(adj_faction, 'adj_faction')

# basic overlap: fraction of geo edges that are also faction edges
geo_mask = (adj_geo>0)
fac_mask = (adj_faction>0)
common = np.logical_and(geo_mask, fac_mask).sum()//2
print(f'common edges (geo & faction): {common}')
print(f'fraction of geo edges that are also faction edges: {common / ( (adj_geo>0).sum()//2 ): .4f}')
