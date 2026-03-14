import pandas as pd

df = pd.read_csv('data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv', low_memory=False)
print(f'Total de registros: {len(df)}')
print('\n--- Primeira 20 RUAS preenchidas ---')
filled = df[df['name'].notna()]
if len(filled) > 0:
    print(filled[['name', 'bairro', 'latitude', 'longitude']].head(20).to_string())
else:
    print("Nenhuma rua preenchida!")
    
print(f'\nRuas preenchidas: {df["name"].notna().sum()} / {len(df)}')
print(f'Bairros preenchidos: {df["bairro"].notna().sum()} / {len(df)}')
