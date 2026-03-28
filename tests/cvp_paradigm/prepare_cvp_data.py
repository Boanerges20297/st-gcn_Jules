import json
import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import re
import unicodedata
import logging
import sys

# --- CONFIGURAÇÃO ---
DATA_DIR = 'data/raw'
BAIRROS_FILE = os.path.join(DATA_DIR, 'bairros_centros_latlong.json')
FACCOES_FILE = os.path.join(DATA_DIR, 'inteligencia_faccoes.csv')
OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
OUTPUT_DIR = 'tests/cvp_paradigm'
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])

NON_OFFICIAL_RESIDUES = [
    'TABAPUÁ', 'PARQUE LEBLON', 'IPARANA', 'GUADALAJARA', 'URUCUTUBA', 
    'PARQUE ALBANO', 'PARQUE SOLEDADE', 'INDUSTRIAL', 'GUAJERU', 'DIF III',
    'MARECHAL RONDON', 'POTIRA', 'JUREMA', 'METRÓPOLE'
]

def normalize_text(text):
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def clean_name(n):
    n = normalize_text(n)
    merges = ['CONJUNTO CEARA', 'PRAIA DO FUTURO', 'VILA MANOEL SATIRO', 'ALTO ALEGRE', 'EDSON QUEIROZ', 'JOSE WALTER']
    for m in merges:
        if m in n: return m
    return n.strip()

def prepare_cvp_paradigm_v5():
    logging.info("🚀 Preparando STGCN v5: Canais de Inteligência e Sazonalidade Ativados")
    
    # 1. Carregar Inteligência
    faccoes_dict = {}
    if os.path.exists(FACCOES_FILE):
        df_fac = pd.read_csv(FACCOES_FILE)
        for _, row in df_fac.iterrows():
            loc = clean_name(str(row['local']))
            faccoes_dict[loc] = {'faction': str(row['faccao_predominante']).upper(), 'grau': float(row.get('grau_dominio', 0.5))}
    
    # 2. Bairros
    with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
        nodes_raw = json.load(f)
    
    fort_nodes = []
    for name, info in nodes_raw.items():
        c_name = clean_name(name)
        if info.get('regiao') == 'fortaleza' and not any(res in normalize_text(name) for res in NON_OFFICIAL_RESIDUES):
            intel = faccoes_dict.get(c_name, {})
            fort_nodes.append({'name': c_name, 'lat': info['lat'], 'long': info['long'], 'faction': intel.get('faction', 'NEUTRO'), 'grau': intel.get('grau', 0.0)})
    
    nodes_df = pd.DataFrame(fort_nodes).drop_duplicates(subset=['name']).reset_index(drop=True)
    N = len(nodes_df)

    # 3. Adjacência Estrita (5km)
    dist_mat = cdist(nodes_df[['lat', 'long']].values, nodes_df[['lat', 'long']].values, 'euclidean')
    adj_geo = (dist_mat < 0.045).astype(float)
    
    # 4. Processar Ocorrências
    occ_raw = pd.read_csv(OCORRENCIAS_FILE, low_memory=False)
    occ_raw['data'] = pd.to_datetime(occ_raw['data'], errors='coerce')
    occ_raw = occ_raw.dropna(subset=['data'])
    
    # Flags para canais de inteligência
    occ_raw['is_veiculo'] = occ_raw['tipo_evento'].str.contains('ROUBO.*VEICULO|FURTO.*VEICULO|CARRO|MOTO', regex=True, na=False)
    
    date_range = pd.date_range(occ_raw['data'].min(), occ_raw['data'].max())
    T = len(date_range)
    date_map = {d: i for i, d in enumerate(date_range)}
    
    features = np.zeros((N, T, 29), dtype=np.float32)
    node_map = {row['name']: i for i, row in nodes_df.iterrows()}
    
    occ_fort = occ_raw[occ_raw['bairro'].apply(clean_name).isin(node_map)].copy()
    occ_fort['n_idx'] = occ_fort['bairro'].apply(clean_name).map(node_map)
    occ_fort['t_idx'] = occ_fort['data'].map(date_map)
    occ_fort = occ_fort.dropna(subset=['t_idx', 'n_idx'])
    
    # Preenchimento Agrupado
    logging.info("🔄 Processando Canais de Dados...")
    # Canal 0: CVP
    cvp_group = occ_fort[occ_fort['tipo'] == 'cvp'].groupby(['n_idx', 't_idx']).size()
    for (n, t), val in cvp_group.items(): features[int(n), int(t), 0] = val
    
    # Canal 1: CVLI (Indicador de Choque/Conflito)
    cvli_group = occ_fort[occ_fort['tipo'] == 'cvli'].groupby(['n_idx', 't_idx']).size()
    for (n, t), val in cvli_group.items(): features[int(n), int(t), 1] = val
        
    # Canal 27: Mobilidade/Veículos
    veic_group = occ_fort[occ_fort['is_veiculo']].groupby(['n_idx', 't_idx']).size()
    for (n, t), val in veic_group.items(): features[int(n), int(t), 27] = val

    # Sazonalidade e Canais Derivados
    for t_idx, date in enumerate(date_range):
        # DOW (3-9)
        features[:, t_idx, 3 + date.weekday()] = 1.0
        # Mês (10-21)
        features[:, t_idx, 10 + date.month - 1] = 1.0
        
        # CANAL 22: WEEKEND FLAG (Sábado = 5, Domingo = 6)
        if date.weekday() >= 5:
            features[:, t_idx, 22] = 1.0

    for n in range(N):
        # Canal 2: Supressão por Facção (0 a 1)
        features[n, :, 2] = nodes_df.iloc[n]['grau']
        
        # Canal 24: Momentum (Média Móvel 7 dias do CVP)
        # Informa se o bairro está em uma "onda" ascendente de roubos
        features[n, :, 24] = pd.Series(features[n, :, 0]).rolling(window=7, min_periods=1).mean().values
        
        # Canal 28: Pulso Global (Volume total da cidade no dia)
        features[n, :, 28] = features[:, :, 0].sum(axis=0)

    # 5. Salvar
    output_path = os.path.join(OUTPUT_DIR, 'processed_fortaleza_CVP.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({'node_features': features, 'adj_geo': adj_geo, 'bairros': nodes_df['name'].tolist(), 'dates': date_range}, f)
    
    logging.info(f"✅ Dataset Rico Gerado! Canais ativados: 0(CVP), 1(CVLI), 2(Tensão), 3-21(Sazonal), 22(Weekend), 24(Momentum), 27(Veículos), 28(Global).")

if __name__ == "__main__":
    prepare_cvp_paradigm_v5()
