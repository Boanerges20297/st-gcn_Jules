import json

with open('data/raw/dados_status_ocorrencias_gerais.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total eventos: {len(data)}")
print(f"\nPrimeiros 10 tipos de eventos:")
for i, e in enumerate(data[:10]):
    print(f"  {i+1}. {e.get('tipo_evento', 'N/A')}")
