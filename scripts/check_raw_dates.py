import json
import pandas as pd
import os

def check_raw_file():
    path = 'data/raw/dados_status_ocorrencias_gerais.json'
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten if necessary
    clean_records = []
    for item in data:
        if not isinstance(item, dict): continue
        if 'data' in item and isinstance(item['data'], dict):
            item = item['data']
        clean_records.append(item)
            
    df = pd.DataFrame(clean_records)
    df['data_dt'] = pd.to_datetime(df['data'].astype(str), errors='coerce')
    
    print(f"File: {path}")
    print(f"Data Maxima: {df['data_dt'].max()}")
    print(f"Total registros: {len(df)}")
    print(f"Registros apos 30/01/2026: {len(df[df['data_dt'] > '2026-01-30'])}")

if __name__ == "__main__":
    check_raw_file()
