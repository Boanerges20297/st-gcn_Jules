#!/usr/bin/env python3
"""Compare dados_merged vs dados_brutos para validação"""
import json
import pandas as pd

print("=" * 80)
print("ANÁLISE: dados_merged.json vs dados_status_ocorrencias_gerais.json")
print("=" * 80)

# Carregando merged
print("\n[1] Carregando dados_merged.json...")
with open('data/raw/dados_merged.json', 'r', encoding='utf-8') as f:
    merged_data = json.load(f)

if isinstance(merged_data, list):
    merged_df = pd.DataFrame(merged_data)
else:
    merged_df = pd.DataFrame([merged_data])

print(f"  Total de registros: {len(merged_df)}")
print(f"  Colunas: {list(merged_df.columns)[:10]}")
print(f"  Intervalo de datas: {merged_df['data'].min()} a {merged_df['data'].max()}")
print(f"\n  Tipos de eventos:")
print(merged_df['tipo_evento'].value_counts().head(10))

# Carregando brutos
print("\n[2] Carregando dados_status_ocorrencias_gerais.json...")
with open('data/raw/dados_status_ocorrencias_gerais.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Filtrar headers
raw_events = [r for r in raw_data if isinstance(r, dict) and 'tipo_evento' in r]
raw_df = pd.DataFrame(raw_events)

print(f"  Total de registros: {len(raw_df)}")
print(f"  Colunas: {list(raw_df.columns)[:10]}")
print(f"  Intervalo de datas: {raw_df['data'].min()} a {raw_df['data'].max()}")
print(f"\n  Tipos de eventos:")
print(raw_df['tipo_evento'].value_counts().head(10))

# Comparação
print("\n[3] COMPARAÇÃO:")
print(f"  Merged tem {len(merged_df)} registros")
print(f"  Brutos tem {len(raw_df)} registros")
print(f"  Diferença: {abs(len(merged_df) - len(raw_df))} registros")

# CVLI
cvli_types = ['HOMICIDIO DOLOSO', 'FEMINICÍDIO', 'ROUBO SEGUIDO DE MORTE (LATROCINIO)', 'LESAO CORPORAL SEGUIDA DE MORTE']
cvli_merged = len(merged_df[merged_df['tipo_evento'].isin(cvli_types)])
cvli_raw = len(raw_df[raw_df['tipo_evento'].isin(cvli_types)])

print(f"\n  CVLI (homicídios):")
print(f"    Merged: {cvli_merged}")
print(f"    Brutos: {cvli_raw}")

# Verificar se dados_merged é subset dos brutos
print(f"\n  ⚠️  ISSUE: dados_merged é SUBSET dos brutos? {len(merged_df) < len(raw_df)}")
print(f"  ⚠️  Datas: merged começa em 2022, brutos em 2026")
