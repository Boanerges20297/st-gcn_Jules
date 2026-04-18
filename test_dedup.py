import pandas as pd
import numpy as np

file_path = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'

# Carrega a base checando tipos (low_memory=False para evitar warnings com tipos variados)
df = pd.read_csv(file_path, low_memory=False)
total_inicial = len(df)
print(f"Total inicial de registros: {total_inicial}")

# Deduplicação Exata (todas as colunas iguais)
df_dedup = df.drop_duplicates()
print(f"Registros após drop_duplicates(exact): {len(df_dedup)} (Removidos: {total_inicial - len(df_dedup)})")

# Deduplicação Lógica: Mesma ocorrência no mesmo dia, hora, bairro e tipo
cols_chave = ['data', 'hora', 'bairro', 'tipo', 'cidade']
df_dedup2 = df_dedup.drop_duplicates(subset=[c for c in cols_chave if c in df_dedup.columns])
print(f"Registros após drop_duplicates(logico): {len(df_dedup2)} (Removidos adicionais: {len(df_dedup) - len(df_dedup2)})")

# Opcional: deduplicação por id_evento caso exista e não seja NaN
if 'id_evento' in df.columns:
    df_with_id = df_dedup2.dropna(subset=['id_evento'])
    dups_id = len(df_with_id) - len(df_with_id.drop_duplicates(subset=['id_evento']))
    print(f"IDs de evento duplicados encontrados (mas não limpos globalmente pq ids as vezes sao null): {dups_id}")

# Salva a base apenas com a deduplicação exata (a mais segura para não perder métricas parecidas se ocorreram juntas)
df_dedup.to_csv(file_path, index=False)
print(f"Base salva com {len(df_dedup)} registros definitivos.")
