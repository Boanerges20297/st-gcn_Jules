import pandas as pd

df = pd.read_csv('outputs/occurrences_with_bairro_geo.csv')
print(f"Total eventos: {len(df)}")
print(f"Colunas: {df.columns.tolist()}")
print(f"\nTipos de evento únicos: {df['tipo_evento'].nunique()}")

tipos = df['tipo_evento'].value_counts().head(30)
print(f"\nTop 30 tipos:\n{tipos}")

# Verificar CVP
cvp_keywords = ['ROUBO', 'FURTO']
df_cvp = df[df['tipo_evento'].str.upper().str.contains('|'.join(cvp_keywords), na=False)]
print(f"\n\nTotal eventos com ROUBO/FURTO: {len(df_cvp)}")

# Verificar veículos
veiculo_keywords = ['VEÍCULO', 'VEICULO', 'MOTO', 'CARRO']
df_veiculo = df_cvp[df_cvp['tipo_evento'].str.upper().str.contains('|'.join(veiculo_keywords), na=False)]
print(f"Total com VEÍCULO/MOTO/CARRO: {len(df_veiculo)}")

print(f"\nExemplos de eventos CVP+Veículo:")
for i, tipo in enumerate(df_veiculo['tipo_evento'].unique()[:10], 1):
    print(f"  {i}. {tipo}")
