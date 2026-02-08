import pickle
import os
import pandas as pd
import geopandas as gpd
import json
import sys

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
NODES_PKL_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
NODES_PKL = os.path.join(NODES_PKL_DIR, 'nodes_gdf.pkl')
BAIRROS_JSON = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')

def recreate_nodes():
    print(f"--- Iniciando recriação de nodes_gdf.pkl ---")
    
    nodes_gdf = None
    data_pack = None
    
    # 1. Tentar carregar do pickle principal
    if os.path.exists(DATA_FILE):
        print(f"Lendo {DATA_FILE}...")
        try:
            with open(DATA_FILE, 'rb') as f:
                data_pack = pickle.load(f)
            nodes_gdf = data_pack.get('nodes_gdf')
            if nodes_gdf is not None:
                print("nodes_gdf encontrado no pickle principal.")
            else:
                print("nodes_gdf NÃO encontrado no pickle principal.")
        except Exception as e:
            print(f"Erro ao ler pickle: {e}")
    else:
        print(f"Arquivo {DATA_FILE} não encontrado.")
    
    # 2. Se não existir, regenerar do JSON
    if nodes_gdf is None:
        print("nodes_gdf não encontrado ou nulo. Regenerando a partir do JSON...")
        if os.path.exists(BAIRROS_JSON):
            try:
                with open(BAIRROS_JSON, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                records = []
                for name, info in data.items():
                    if name in ["Nome", "null", "None", ""] or name is None:
                        continue
                    
                    regiao = info.get('regiao', 'desconhecido').lower()
                    
                    # Lógica de tipo de nó (igual ao app.py)
                    if 'node_type' in info:
                        node_type = info['node_type']
                    elif 'eh_cidade' in info and info['eh_cidade']:
                        node_type = 'cidade'
                    else:
                        node_type = 'bairro' if regiao == 'fortaleza' else 'cidade'
                    
                    records.append({
                        'name': name,
                        'latitude': info['lat'],
                        'longitude': info['long'],
                        'regiao': regiao,
                        'node_type': node_type
                    })
                
                if records:
                    df = pd.DataFrame(records)
                    nodes_gdf = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df.longitude, df.latitude),
                        crs="EPSG:4326"
                    )
                    print(f"Regenerado com sucesso: {len(nodes_gdf)} nós.")
                    
                    # Atualizar o pickle principal para corrigir o problema na raiz
                    if data_pack is not None:
                        print("Atualizando processed_graph_data.pkl com nodes_gdf...")
                        data_pack['nodes_gdf'] = nodes_gdf
                        with open(DATA_FILE, 'wb') as f:
                            pickle.dump(data_pack, f)
            except Exception as e:
                print(f"Erro ao processar JSON: {e}")
        else:
            print(f"ERRO: Arquivo JSON não encontrado: {BAIRROS_JSON}")
            return

    # 3. Salvar nodes_gdf.pkl
    if nodes_gdf is not None:
        os.makedirs(NODES_PKL_DIR, exist_ok=True)
        try:
            with open(NODES_PKL, 'wb') as f:
                pickle.dump(nodes_gdf, f)
            print(f"Sucesso! Arquivo salvo em: {NODES_PKL}")
            print(f"Total de nós: {len(nodes_gdf)}")
            
            # Verificar se há geometrias válidas
            valid_geoms = nodes_gdf.geometry.notna().sum()
            print(f"Geometrias válidas: {valid_geoms}/{len(nodes_gdf)}")
            
        except Exception as e:
            print(f"Erro ao salvar nodes_gdf.pkl: {e}")
    else:
        print("Falha crítica: Não foi possível obter ou gerar nodes_gdf.")

if __name__ == "__main__":
    recreate_nodes()