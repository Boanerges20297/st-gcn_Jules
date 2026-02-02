#!/usr/bin/env python3
"""Filtrar dados_merged.json para 2024/2025"""
import json
import pandas as pd

print("Filtrando dados_merged.json para 2024/2025...")

with open('data/raw/dados_merged.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)

df = pd.DataFrame(all_data)
df['data'] = pd.to_datetime(df['data'], errors='coerce')

# Filtrar 2024/2025
filtered_df = df[(df['data'] >= '2024-01-01') & (df['data'] <= '2025-12-31')]

print(f"Total original: {len(df)}")
print(f"Total 2024/2025: {len(filtered_df)}")
print(f"Intervalo: {filtered_df['data'].min()} a {filtered_df['data'].max()}")

# Converter datas para string antes de salvar
filtered_df['data'] = filtered_df['data'].dt.strftime('%Y-%m-%d')

# Salvar como dados_merged_2024_2025.json
output_file = 'data/raw/dados_merged_2024_2025.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_df.to_dict('records'), f, ensure_ascii=False)

print(f"\n✓ Salvo em: {output_file}")
print(f"\nCVLI: {len(filtered_df[filtered_df['tipo_evento'] == 'HOMICIDIO DOLOSO'])}")
print(f"CVP (ROUBO): {len(filtered_df[filtered_df['tipo_evento'].str.contains('ROUBO', na=False)])}")
