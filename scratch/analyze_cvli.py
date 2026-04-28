import pandas as pd
import os

path = r'c:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
if os.path.exists(path):
    df = pd.read_csv(path, low_memory=False)
    cvli = df[df['tipo'] == 'cvli']
    # Group by id_evento (or date/hour/bairro if id_evento is missing)
    # But let's look for id_evento first
    if 'id_evento' in cvli.columns:
        counts = cvli.groupby('id_evento').size()
        multi = counts[counts > 1]
        print(f"Total CVLI rows: {len(cvli)}")
        print(f"Occurrences with multiple victims (by id_evento): {len(multi)}")
        print(multi.value_counts())
        
        # Look at one example
        if len(multi) > 0:
            example_id = multi.index[0]
            print(f"\nExample id_evento: {example_id}")
            print(cvli[cvli['id_evento'] == example_id][['data', 'hora', 'bairro', 'nome_vitima']])
    else:
        print("Column id_evento not found in CVLI data")
else:
    print("File not found")
