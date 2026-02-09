#!/usr/bin/env python
import pickle

data = pickle.load(open('data/processed/processed_graph_data.pkl', 'rb'))
nodes_gdf = data['nodes_gdf']

print("=" * 80)
print("📊 ESTATÍSTICAS ATUALIZADAS DO GRAFO")
print("=" * 80)
print(f"\nTotal de nós: {len(nodes_gdf)}")
print(f"\nDistribuição por tipo:")
type_counts = nodes_gdf['node_type'].value_counts()
for ntype, count in type_counts.items():
    print(f"   - {ntype}: {count}")

print(f"\nProcurando por nós importantes:")
important = ['MORRO', 'FAVELA', 'COMUNIDADE', 'BECO', 'OURO']
for term in important:
    mask = nodes_gdf['name'].str.upper().str.contains(term, na=False)
    count = mask.sum()
    if count > 0:
        print(f"   ✅ {term}: {count} nós encontrados")
        samples = nodes_gdf[mask]['name'].head(3).tolist()
        for name in samples:
            print(f"      - {name}")

print("\n" + "=" * 80)
