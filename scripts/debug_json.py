import sys
sys.path.append('.')
from scripts.merge_new_data import robust_load_any

df = robust_load_any('data/raw/dados_status.json')
print("Columns:", df.columns.tolist())

if 'tipo_evento' in df.columns:
    print("\nTop 20 eventos:")
    print(df['tipo_evento'].value_counts().head(20))
    
    cvlis = df[df['tipo_evento'].str.contains('HOMICIDIO|CVLI|LATROCINIO|LESAO CORPORAL SEGUIDA DE MORTE|FEMINICIDIO', case=False, na=False)]
    print("\nPossible CVLIs count:", len(cvlis))
    print(cvlis[['data', 'tipo_evento', 'bairro', 'latitude', 'longitude']])
else:
    print("No tipo_evento column found.")
