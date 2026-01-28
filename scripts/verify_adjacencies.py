import pickle
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

p = Path('data/processed/processed_graph_data.pkl')
if not p.exists():
    raise SystemExit('processed_graph_data.pkl not found')

with p.open('rb') as f:
    data = pickle.load(f)

nodes = data.get('nodes_gdf')
adj_geo = np.array(data.get('adj_geo'))
adj_faction = np.array(data.get('adj_faction'))

if nodes is None:
    raise SystemExit('nodes_gdf missing in data pack')

# Project centroids to metric CRS (EPSG:3857) for distance checks
nodes_proj = nodes.to_crs(epsg=3857)
centroids = nodes_proj.geometry.centroid
xs = centroids.x.values
ys = centroids.y.values
coords = np.column_stack((xs, ys))

# Get list of geo edges (undirected, i<j)
i_idx, j_idx = np.where(np.triu(adj_geo > 0, k=1))
edges = list(zip(i_idx.tolist(), j_idx.tolist()))

def dist_m(i, j):
    dx = coords[i,0] - coords[j,0]
    dy = coords[i,1] - coords[j,1]
    return np.hypot(dx, dy)

# Sample up to 2000 random geo edges to check distances
import random
random.seed(42)
N = len(edges)
sample_n = min(2000, N)
if N == 0:
    print('No geo edges found.')
    raise SystemExit
sampled = random.sample(edges, sample_n)
dists = [dist_m(u,v) for u,v in sampled]

# Fraction of sampled edges within 2000 m
within_2000 = sum(1 for d in dists if d <= 2000)
frac_within = within_2000 / len(dists)

# Also compute max distance among all geo edges (careful: could be many)
# We'll compute for all but be mindful of memory/time; do vectorized approach
all_i, all_j = i_idx, j_idx
if len(all_i) > 0:
    dx_all = coords[all_i,0] - coords[all_j,0]
    dy_all = coords[all_i,1] - coords[all_j,1]
    d_all = np.hypot(dx_all, dy_all)
    max_dist = float(d_all.max())
    pct_le_2000 = float((d_all <= 2000).sum()) / len(d_all)
else:
    max_dist = 0.0
    pct_le_2000 = 0.0

# Faction influence checks
# For a few random nodes, list how many faction-neighbors they have and whether those neighbors are connected in adj_faction
unique_factions = nodes['faction'].values if 'faction' in nodes.columns else np.array(['']*len(nodes))
# Pick up to 10 example nodes that have non-empty faction
candidates = [i for i,f in enumerate(unique_factions) if f]
examples = random.sample(candidates, min(10, len(candidates)))
example_info = []
for i in examples:
    fac = unique_factions[i]
    same_idxs = [idx for idx,x in enumerate(unique_factions) if x == fac and idx != i]
    num_same = len(same_idxs)
    # Check adj_faction connectivity to these indices
    connected = 0
    for j in same_idxs[:200]:
        if adj_faction[i,j] > 0:
            connected += 1
    example_info.append({'node': i, 'faction': fac, 'same_count': num_same, 'connected_checked': min(200, num_same), 'connected': connected})

# Overlap stats
geo_edges = (adj_geo > 0).sum()//2
faction_edges = (adj_faction > 0).sum()//2
common = np.logical_and(adj_geo>0, adj_faction>0).sum()//2

print('Geo edges (undirected):', int(geo_edges))
print('Faction edges (undirected):', int(faction_edges))
print('Common edges:', int(common))
print('Sampled fraction of geo edges within 2000m: {:.3f} (sample_size={})'.format(frac_within, len(dists)))
print('All geo edges: max distance (m): {:.1f}, pct <=2000: {:.3f}'.format(max_dist, pct_le_2000))
print('\nExample faction connectivity (per node):')
for info in example_info:
    print(info)

# Quick assertion-like statements
print('\nChecks:')
print(' - Majority of geo edges within 2000m sample:', frac_within > 0.98)
print(' - Most geo edges overall <=2000m:', pct_le_2000 > 0.98)
print(' - Faction edges present (dense):', faction_edges > geo_edges)
print(' - Fraction of geo edges also faction edges:', common / max(1, geo_edges))
