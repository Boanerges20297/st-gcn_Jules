"""
Força o merge dos dados de 2026 ao arquivo principal
"""
import json
import pandas as pd
from datetime import datetime

print("="*60)
print("MERGE FORÇADO - DADOS 2026")
print("="*60)

# Carregar arquivo com dados novos
print("\n1. Carregando dados_status_020226.json...")
with open('data/raw/dados_status_020226.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Arquivo é export PHPMyAdmin - dados estão em raw_data[2]['data']
new_records = []
for item in raw_data:
    if isinstance(item, dict) and item.get('type') == 'table':
        table_data = item.get('data', [])
        new_records.extend(table_data)

print(f"   ✓ {len(new_records)} registros encontrados")

# Verificar período
df_new = pd.DataFrame(new_records)
df_new['data_dt'] = pd.to_datetime(df_new['data'])
print(f"   - Período: {df_new['data_dt'].min()} até {df_new['data_dt'].max()}")

# Carregar base existente
print("\n2. Carregando dados_status_ocorrencias_gerais.json...")
try:
    with open('data/raw/dados_status_ocorrencias_gerais.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    # Pode ser que seja export PHPMyAdmin também
    if isinstance(existing_data, list) and len(existing_data) > 0:
        if isinstance(existing_data[0], dict) and existing_data[0].get('type') in ['header', 'database', 'table']:
            # É export PHPMyAdmin
            real_data = []
            for item in existing_data:
                if isinstance(item, dict) and item.get('type') == 'table':
                    real_data.extend(item.get('data', []))
            existing_data = real_data
    
    print(f"   ✓ {len(existing_data)} registros existentes")
    
    if len(existing_data) > 0:
        df_old = pd.DataFrame(existing_data)
        if 'data' in df_old.columns:
            # Converter para string primeiro (pode ter listas)
            df_old['data'] = df_old['data'].astype(str)
            df_old['data_dt'] = pd.to_datetime(df_old['data'], errors='coerce')
            print(f"   - Período: {df_old['data_dt'].min()} até {df_old['data_dt'].max()}")
except FileNotFoundError:
    existing_data = []
    print("   ⚠️  Arquivo não encontrado, criando novo")

# Backup
print("\n3. Criando backup...")
backup_file = f'data/raw/backups/dados_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
import os
os.makedirs('data/raw/backups', exist_ok=True)
with open(backup_file, 'w', encoding='utf-8') as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=2)
print(f"   ✓ Backup: {backup_file}")

# Merge baseado em (id, data, hora) para evitar duplicatas reais
print("\n4. Mergeando dados...")
existing_keys = {(r.get('id'), r.get('data'), r.get('hora')) for r in existing_data}
truly_new = []

for rec in new_records:
    key = (rec.get('id'), rec.get('data'), rec.get('hora'))
    if key not in existing_keys:
        truly_new.append(rec)

print(f"   ✓ {len(truly_new)} registros realmente novos (não duplicados)")

# Combinar
merged = existing_data + truly_new

# Ordenar por data
try:
    merged.sort(key=lambda x: (x.get('data', ''), x.get('hora', '')))
except:
    pass

# Salvar
print("\n5. Salvando arquivo mergeado...")
with open('data/raw/dados_status_ocorrencias_gerais.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"   ✓ Total de registros: {len(merged)}")

# Verificar resultado
df_final = pd.DataFrame(merged)
df_final['data_dt'] = pd.to_datetime(df_final['data'])
print(f"\n📊 RESULTADO FINAL:")
print(f"   - Total: {len(df_final)} registros")
print(f"   - Período: {df_final['data_dt'].min()} até {df_final['data_dt'].max()}")
print(f"   - Registros 2026: {len(df_final[df_final['data_dt'] >= '2026-01-01'])}")
print(f"   - CVLI total: {len(df_final[df_final['tipo'] == 'cvli'])}")
print(f"   - CVP total: {len(df_final[df_final['tipo'] == 'cvp'])}")

print(f"\n{'='*60}")
print("✅ MERGE CONCLUÍDO!")
print("="*60)
print("\nPróximos passos:")
print("  1. python src/data_processing.py")
print("  2. python src/train.py")
