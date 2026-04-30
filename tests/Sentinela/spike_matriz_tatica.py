import pandas as pd
import json
import numpy as np
from math import radians, cos, sin, asin, sqrt
import networkx as nx
import os
import unicodedata

def normalize_text(text):
    if pd.isna(text): return text
    text = str(text).upper().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

# --- Configurations ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
BAIRROS_JSON = os.path.join(DATA_DIR, "bairros_centros_latlong.json")
FACCOES_CSV = os.path.join(DATA_DIR, "inteligencia_faccoes.csv")
OCORRENCIAS_CSV = os.path.join(DATA_DIR, "ocorrencias_tropa_limpo_fortaleza.csv")

# Constants for Haversine distance
def haversine(lon1, lat1, long2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, long2, lat2 = map(radians, [lon1, lat1, long2, lat2])

    # haversine formula 
    dlon = long2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def run_spike():
    print("Iniciando Spike: Construção da Matriz Tática...")

    # 1. Carregar Dados de Bairros (Coordenadas)
    print("Carregando coordenadas dos bairros...")
    with open(BAIRROS_JSON, "r", encoding="utf-8") as f:
        bairros_data = json.load(f)
    
    bairros_list = []
    for nome, info in bairros_data.items():
        if info['regiao'] == 'fortaleza': # filtrar apenas fortaleza
            bairros_list.append({
                'bairro': normalize_text(nome),
                'lat': info['lat'],
                'long': info['long']
            })
    df_bairros = pd.DataFrame(bairros_list)
    print(f"Total de bairros carregados: {len(df_bairros)}")

    # 2. Carregar Dados de Facções
    print("Carregando inteligência de facções...")
    df_faccoes = pd.read_csv(FACCOES_CSV)
    df_faccoes['bairro'] = df_faccoes['local'].apply(normalize_text)
    df_bairros = pd.merge(df_bairros, df_faccoes[['bairro', 'faccao_predominante', 'grau_dominio']], on='bairro', how='left')
    df_bairros['faccao_predominante'] = df_bairros['faccao_predominante'].fillna('NEUTRO')
    df_bairros['grau_dominio'] = df_bairros['grau_dominio'].fillna(0.0)

    # 3. Carregar Ocorrências Policiais (Fragilidade)
    print("Carregando ocorrências e calculando índice de fragilidade...")
    df_ocorrencias = pd.read_csv(OCORRENCIAS_CSV)
    df_ocorrencias['bairro'] = df_ocorrencias['bairro'].apply(normalize_text)
    
    # Agrupar por bairro para obter total de armas e drogas (proxy de ação policial / vulnerabilidade)
    fragilidade = df_ocorrencias.groupby('bairro').agg({
        'qtd_armas': 'sum',
        'qtd_drogas': 'sum',
        'natureza': 'count' # total de ocorrencias
    }).reset_index()
    fragilidade.rename(columns={'natureza': 'total_acoes_policiais'}, inplace=True)
    
    # Criar um score de fragilidade: armas pesam mais.
    fragilidade['score_fragilidade'] = fragilidade['qtd_armas'] * 15.0 + fragilidade['qtd_drogas'] * 1.0 + fragilidade['total_acoes_policiais'] * 0.5
    # Normalizar (Min-Max)
    max_frag = fragilidade['score_fragilidade'].max() if fragilidade['score_fragilidade'].max() > 0 else 1
    fragilidade['fragilidade_norm'] = fragilidade['score_fragilidade'] / max_frag

    df_bairros = pd.merge(df_bairros, fragilidade[['bairro', 'fragilidade_norm', 'qtd_armas', 'qtd_drogas']], on='bairro', how='left')
    df_bairros['fragilidade_norm'] = df_bairros['fragilidade_norm'].fillna(0.0)
    df_bairros['qtd_armas'] = df_bairros['qtd_armas'].fillna(0)
    
    # 4. Construir Matriz de Adjacência Tática
    print("Construindo Matriz de Adjacência Tática...")
    nodes = df_bairros['bairro'].tolist()
    n = len(nodes)
    A_tactical = np.zeros((n, n))
    
    # Parâmetros de calibração tática
    DIST_THRESHOLD_KM = 3.0 # Considerar adjacência viária se < 3km (proxy de conexão fácil)
    RIVALRY_MULTIPLIER = 2.0 # Se facções diferentes, maior tensão
    
    edges_info = []

    for i in range(n):
        for j in range(n):
            if i != j:
                dist = haversine(df_bairros.loc[i, 'long'], df_bairros.loc[i, 'lat'], 
                                 df_bairros.loc[j, 'long'], df_bairros.loc[j, 'lat'])
                
                # Regra Retilínea (Topologia Bruta)
                if dist <= DIST_THRESHOLD_KM:
                    # Peso base: inverso da distância
                    weight = 1.0 / (dist + 0.1) 
                    
                    fac_i = df_bairros.loc[i, 'faccao_predominante']
                    fac_j = df_bairros.loc[j, 'faccao_predominante']
                    
                    # Conflito de Facções
                    is_enemy = (fac_i != fac_j) and (fac_i != 'NEUTRO') and (fac_j != 'NEUTRO')
                    if is_enemy:
                        weight *= RIVALRY_MULTIPLIER
                    
                    # Fragilidade: Se o bairro J foi alvo de muitas ações policiais,
                    # ele está "ferido" e mais vulnerável a ataques do bairro I (inimigo).
                    # Adicionamos um bônus no peso da aresta I -> J (direcionado)
                    frag_j = df_bairros.loc[j, 'fragilidade_norm']
                    if is_enemy and frag_j > 0.1:
                        weight *= (1.0 + frag_j) # Aumenta peso de ataque para J
                        
                    A_tactical[i, j] = weight
                    
                    edges_info.append({
                        'origem': nodes[i],
                        'destino': nodes[j],
                        'fac_origem': fac_i,
                        'fac_destino': fac_j,
                        'dist_km': dist,
                        'weight': weight,
                        'fragilidade_destino': frag_j,
                        'is_enemy': is_enemy
                    })

    # 5. Avaliar e validar
    print("\nResumo da Matriz:")
    print(f"Shape: {A_tactical.shape}")
    print(f"Densidade: {(np.count_nonzero(A_tactical) / (n*n)):.4f}")
    
    df_edges = pd.DataFrame(edges_info)
    df_enemies = df_edges[df_edges['is_enemy'] == True]
    
    print("\nTop 10 Rotas de Conflito Tático (Maior Risco de Ataque devido à Vulnerabilidade):")
    top_risks = df_enemies.sort_values(by='weight', ascending=False).head(10)
    for idx, row in top_risks.iterrows():
        print(f" -> {row['origem']} ({row['fac_origem']}) -> {row['destino']} ({row['fac_destino']}) | Peso: {row['weight']:.2f} | Dist: {row['dist_km']:.2f}km | Fragilidade Dest: {row['fragilidade_destino']:.2f}")

    # Salvar matriz para inspeção
    out_file = "matriz_tatica_spike.npy"
    np.save(out_file, A_tactical)
    print(f"\nMatriz salva em {out_file}")
    print("Spike finalizado.")

if __name__ == "__main__":
    run_spike()
