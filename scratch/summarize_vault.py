import json
import os

path = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview\data\training_vault\vault_state_e1.json"
with open(path, 'r') as f:
    data = json.load(f)

hits = data['hits']
long_term = data['long_term']

print(f"Total nodes: {len(hits)}")
print(f"Nodes with > 0 hits: {len([h for h in hits if h > 0])}")
print(f"Max hits: {max(hits)} at index {hits.index(max(hits))}")
print(f"Avg hits (non-zero): {sum(hits)/len([h for h in hits if h > 0]):.2f}")
print(f"Max long_term: {max(long_term)} at index {long_term.index(max(long_term))}")

# Check top 5 indices
top_hits = sorted(range(len(hits)), key=lambda i: hits[i], reverse=True)[:5]
for i in top_hits:
    print(f"Index {i}: {hits[i]} hits, {long_term[i]:.2f} long_term")
