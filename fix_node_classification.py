import json

# Cidades da RMF (região metropolitana, NÃO bairros de Fortaleza)
RMF_CITIES = {
    'trairi', 'eusebio', 'aquiraz', 'caucaia', 'maracanau', 'pacajus',
    'horizonte', 'itaitinga', 'chorozinho', 'guaiuba', 'pacatuba',
    'saogoncalo', 'iguatu', 'russas', 'quixada', 'sobral',
    'crateus', 'quixeramobim', 'morada nova', 'limoeiro'
}

# Carregar dados
with open('data/raw/bairros_centros_latlong.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

fixed_count = 0

for key in list(data.keys()):
    entry = data[key]
    name_normalized = key.lower().strip()
    
    # Se é uma entrada de RMF
    if entry.get('regiao') == 'rmf':
        # Se o nome está na lista de cidades RMF, garantir que seja classificado como cidade
        if name_normalized in RMF_CITIES:
            # Adicionar campo para indicar que é uma cidade, não bairro
            entry['node_type'] = 'cidade'
            entry['eh_cidade'] = True
            fixed_count += 1
            print(f"✓ Fixed RMF City: {key} -> node_type: 'cidade'")

# Salvar versão corrigida
backup_file = 'data/raw/bairros_centros_latlong_backup.json'
with open(backup_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"\n✓ Backup salvo: {backup_file}")

# Salvar versão original corrigida
with open('data/raw/bairros_centros_latlong.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✓ Updated: data/raw/bairros_centros_latlong.json")
print(f"Total entries fixed: {fixed_count}")
