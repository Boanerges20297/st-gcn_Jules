import json

with open('data/raw/dados_status_020226.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Type: {type(d)}")
print(f"Len: {len(d)}")
print("\nFirst 5 items:")
for i in range(min(5, len(d))):
    print(f"{i}: {type(d[i])}")
    if isinstance(d[i], dict):
        print(f"   Keys: {list(d[i].keys())[:10]}")
