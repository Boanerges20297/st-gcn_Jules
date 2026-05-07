import pandas as pd
import numpy as np
import json
import unicodedata
from scipy.spatial import KDTree

def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def repair_bairros():
    CSV_ENRICH = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    LATLON_FILE = 'data/raw/bairros_centros_latlong.json'
    
    print("Carregando dados...")
    df = pd.read_csv(CSV_ENRICH, low_memory=False)
    
    with open(LATLON_FILE, encoding="utf-8") as f:
        raw_ll = json.load(f)
    
    # Filtrar apenas bairros de Fortaleza para o mapeamento
    fortaleza_bairros = {norm(k): v for k, v in raw_ll.items() if v.get('regiao') == 'fortaleza'}
    
    names = list(fortaleza_bairros.keys())
    coords = np.array([[fortaleza_bairros[n]['lat'], fortaleza_bairros[n]['long']] for n in names])
    tree = KDTree(coords)
    
    mask_null = df['bairro'].isna() | (df['bairro'].apply(norm) == 'DESCONHECIDO')
    mask_gps = df['latitude'].notna() & df['longitude'].notna()
    mask_to_repair = mask_null & mask_gps
    
    print(f"Total de registros: {len(df)}")
    print(f"Registros com bairro DESCONHECIDO/Nulo: {mask_null.sum()}")
    print(f"Registros passíveis de reparo (com GPS): {mask_to_repair.sum()}")
    
    if mask_to_repair.sum() > 0:
        points = df.loc[mask_to_repair, ['latitude', 'longitude']].values
        dist, idx = tree.query(points)
        
        # Limite de 5km (aprox 0.045 graus)
        THRESHOLD = 0.045 
        
        recovered = 0
        for i, (d, ix) in enumerate(zip(dist, idx)):
            if d < THRESHOLD:
                real_idx = df.index[mask_to_repair][i]
                df.at[real_idx, 'bairro'] = names[ix]
                df.at[real_idx, 'cidade'] = 'FORTALEZA' # Se está perto de um bairro de Fortaleza, é Fortaleza
                recovered += 1
                
        print(f"Sucesso: {recovered} registros recuperados e atribuídos a bairros reais.")
        
        # Mostrar exemplos
        print("\nExemplos de recuperados:")
        print(df.loc[mask_to_repair][['bairro', 'latitude', 'longitude', 'cidade']].head(10))
    
    # Salvar uma versão temporária para validar
    # df.to_csv('data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO_REPAIRED.csv', index=False)

if __name__ == "__main__":
    repair_bairros()
