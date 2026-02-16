import json
import pandas as pd
import os

def debug_dates():
    path = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clean_records = []
    for item in data:
        if isinstance(item, dict):
            clean_item = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in item.items()}
            clean_records.append(clean_item)
            
    df = pd.DataFrame(clean_records)
    df['data_dt'] = pd.to_datetime(df['data'].astype(str), errors='coerce')
    
    print("Data Maxima no JSON: " + str(df['data_dt'].max()))
    print("Total registros: " + str(len(df)))
    
    print("\nUltimos 5 registros do JSON:")
    print(df[['id', 'data', 'tipo', 'cidade']].tail(5).to_string())
    
    feb_records = df[df['data_dt'] >= '2026-02-01']
    print("\nRegistros em Fevereiro/2026: " + str(len(feb_records)))
    
    if len(feb_records) > 0:
        print("\nExemplo de registro de Fevereiro:")
        print(feb_records[['id', 'data', 'tipo', 'cidade']].head(3).to_string())

if __name__ == "__main__":
    debug_dates()
