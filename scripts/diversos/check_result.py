import pandas as pd
df = pd.read_csv('data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv', nrows=1000, low_memory=False)
print(f'Ruas preenchidas: {df["name"].notna().sum()} / {len(df)}')
print(f'Bairros preenchidos: {df["bairro"].notna().sum()} / {len(df)}')
print('\n--- Amostra com ruas ---')
sample = df[df['name'].notna()][['name', 'bairro']].head(10)
if len(sample) > 0:
    print(sample.to_string())
else:
    print("Nenhuma rua encontrada na amostra")
