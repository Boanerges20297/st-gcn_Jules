import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import re
import unicodedata
import random
import logging
import sys
import geopandas as gpd
from scipy.spatial.distance import cdist

# --- CONFIGURAÇÕES ---
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
from src.core.architectures import DeepSTGAT_64

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_jules_final.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 120
LR = 0.05
DROPOUT = 0.5
RANKING_WEIGHT = 20.0
EPOCHS = 60
GRAD_ACCUM = 32 

RMF_OFFICIAL = [
    'AQUIRAZ', 'BEBERIBE', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 
    'GUAIUBA', 'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 
    'PACATUBA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE', 
    'SAO LUIS DO CURU', 'TRAIRI'
]

SUBDIVISION_TO_CITY = {
    'GUADALAJARA': 'CAUCAIA', 'GUAJERU': 'CAUCAIA', 'INDUSTRIAL': 'CAUCAIA',
    'IPARANA': 'CAUCAIA', 'MARECHAL RONDON': 'CAUCAIA', 'PARQUE ALBANO': 'CAUCAIA',
    'PARQUE SOLEDADE': 'CAUCAIA', 'ALTO ALEGRE': 'MARACANAU', 'CIDADE NOVA': 'MARACANAU',
    'URUCUTUBA': 'CAUCAIA'
}

def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def clean_name(n):
    n = normalize_text(n)
    merges = ['CONJUNTO CEARA', 'PRAIA DO FUTURO', 'VILA MANOEL SATIRO', 'EDSON QUEIROZ', 'JOSE WALTER']
    for m in merges:
        if m in n: return m
    n = re.sub(r'\s+[IVXLCDM]+$', '', n)
    n = re.sub(r'\s+\d+$', '', n)
    n = n.strip()
    return SUBDIVISION_TO_CITY.get(n, n)

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

def rebuild_datasets():
    logging.info("🛠️ Reconstruindo Datasets (Filtro Jules + 29 CANAIS ORIGINAIS)...")
    
    # 1. Carregar Ocorrências Master
    with open('data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.json', 'r', encoding='utf-8') as f:
        occ_raw = json.load(f)
    
    clean_records = []
    for item in occ_raw:
        if not isinstance(item, dict): continue
        if 'data' in item and isinstance(item['data'], dict): item = item['data']
        clean_item = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in item.items()}
        clean_records.append(clean_item)
    occ_df = pd.DataFrame(clean_records)
    occ_df['data'] = pd.to_datetime(occ_df['data'].astype(str), errors='coerce')
    occ_df = occ_df.dropna(subset=['data']).sort_values('data')
    occ_df['tipo_upper'] = occ_df.get('tipo_evento', pd.Series()).fillna('').astype(str).str.upper()
    
    # 2. Carregar e Filtrar Nós
    with open('data/raw/bairros_centros_latlong.json', 'r', encoding='utf-8') as f:
        nodes_raw = json.load(f)
    
    months = 1000 / 30.0
    cvli_counts = occ_df[occ_df['tipo'].str.lower() == 'cvli'].copy()
    cvli_counts['loc_clean'] = cvli_counts.apply(lambda r: clean_name(r.get('bairro_geo') or r.get('bairro') or r.get('municipio') or r.get('cidade')), axis=1)
    counts_map = cvli_counts.groupby('loc_clean').size()

    records = []
    for name, info in nodes_raw.items():
        c_name = clean_name(name)
        if c_name == 'DIF': continue
        reg = info.get('regiao', 'interior').lower()
        if c_name in RMF_OFFICIAL: reg = 'rmf'
        elif reg == 'rmf': continue 
        
        c_per_m = counts_map.get(c_name, 0) / months
        has_f = info.get('faction', 'NEUTRO').upper() != 'NEUTRO'
        
        # Filtro Jules
        keep = False
        if reg == 'rmf' and c_name in RMF_OFFICIAL: keep = True
        elif has_f: keep = True
        elif reg == 'fortaleza' and c_per_m >= 1.0: keep = True
        elif reg == 'interior' and c_per_m >= 1.0: keep = True
        
        if keep:
            records.append({
                'name': c_name, 'lat': info['lat'], 'long': info['long'],
                'regiao': reg, 'faction': info.get('faction', 'NEUTRO').upper(),
                'tension_index': 1.0 if ('CAUCAIA' in c_name or 'MARACANAU' in c_name) else 0.0
            })
    
    nodes_df = pd.DataFrame(records).drop_duplicates(subset=['name']).reset_index(drop=True)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat), crs="EPSG:4326")

    # 3. Construir Tensores de 29 Canais
    start_date, end_date = occ_df['data'].min(), occ_df['data'].max()
    date_range = pd.date_range(start_date, end_date)
    date_map = {d: i for i, d in enumerate(date_range)}
    
    is_cvli = occ_df['tipo'].fillna('').astype(str).str.lower() == 'cvli'
    is_veiculo = occ_df['tipo_upper'].str.contains('ROUBO.*VEICULO|FURTO.*VEICULO|CARRO|MOTO', regex=True)
    is_intel = occ_df['tipo_upper'].str.contains('LESAO.*BALA|ARMA.*FOGO|DISPARO|TIRO|INVASAO', regex=True)

    for reg in ['fortaleza', 'rmf', 'interior']:
        reg_nodes = nodes_gdf[nodes_gdf['regiao'] == reg].copy().reset_index(drop=True)
        N = len(reg_nodes)
        if N == 0: continue
        
        features = np.zeros((N, len(date_range), 29))
        node_map = {row['name']: i for i, row in reg_nodes.iterrows()}
        
        for _, row in occ_df.iterrows():
            loc_clean = clean_name(row.get('bairro_geo') or row.get('bairro') or row.get('municipio') or row.get('cidade'))
            if loc_clean in node_map:
                n_idx = node_map[loc_clean]
                t_idx = date_map[row['data']]
                if is_cvli.loc[row.name]: features[n_idx, t_idx, 0] += 1
                if is_veiculo.loc[row.name]: features[n_idx, t_idx, 1] += 1
                if is_intel.loc[row.name]: features[n_idx, t_idx, 27] += 1
        
        # Sazonalidades e Inércia
        for d_idx, date in enumerate(date_range):
            features[:, d_idx, 28] = features[:, d_idx, 0].sum() # City Pulse
            features[:, d_idx, 3 + date.weekday()] = 1.0 # DOW
            features[:, d_idx, 10 + date.month - 1] = 1.0 # Month
            if date.weekday() >= 5: features[:, d_idx, 22] = 1.0 # Weekend
            
        for n in range(N):
            features[n, :, 24] = pd.Series(features[n, :, 0]).rolling(window=7, min_periods=1).mean().values # Normal Shock (Trend)
            features[n, :, 2] = reg_nodes.iloc[n]['tension_index'] # Tension Index fixed

        dist_mat = cdist(reg_nodes[['lat', 'long']].values, reg_nodes[['lat', 'long']].values, 'euclidean')
        adj_geo = (dist_mat < 0.05).astype(float)
        
        dataset = {'node_features': features, 'adj_geo': adj_geo, 'adj_conflict': np.eye(N), 'nodes_gdf': reg_nodes, 'dates': date_range}
        with open(f'data/processed/processed_{reg}.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        logging.info(f"✅ {reg.upper()} Dataset Pronto (29 Canais, {N} nós).")

def train_specialist(region_key):
    logging.info(f"\n🚀 TREINO JULES (ESTABILIZADO 29 CANAIS): {region_key.upper()}")
    path = f'data/processed/processed_{region_key}.pkl'
    with open(path, 'rb') as f: data = pickle.load(f)
    
    nf = data['node_features']
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(data['adj_conflict'], dtype=torch.float32).to(DEVICE)
    N, T, C = nf.shape
    
    f_norm = nf.copy()
    for c in range(C):
        m, s = nf[:,:,c].mean(), nf[:,:,c].std() + 1e-5
        f_norm[:,:,c] = (nf[:,:,c] - m) / s

    X_list, y_list = [], []
    for t in range(WINDOW, T - 7):
        X_list.append(torch.tensor(f_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2,0,1).unsqueeze(0))
        y_list.append(torch.tensor(nf[:, t:t+7, 0].sum(axis=1), dtype=torch.float32).unsqueeze(0))
    
    val_size = 60
    train_X, train_y = X_list[:-val_size], y_list[:-val_size]
    val_X, val_y = X_list[-val_size:], y_list[-val_size:]
    
    model = DeepSTGAT_64(num_nodes=N, in_channels=29, time_steps=WINDOW, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_X) // GRAD_ACCUM
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=total_steps, epochs=EPOCHS)

    best_val = 0
    for epoch in range(EPOCHS):
        model.train()
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        optimizer.zero_grad()
        step_count = 0
        for i, idx in enumerate(indices):
            bx, by = train_X[idx].to(DEVICE), train_y[idx].to(DEVICE)
            pred = model(bx, [adj_geo, adj_conf]).squeeze()
            target = by.squeeze()
            
            mse = F.smooth_l1_loss(pred, target / (target.max() + 1e-5))
            k_rank = 15 if region_key == 'fortaleza' else 10
            _, top_idx = torch.topk(target, min(k_rank, N))
            num_neg = min(30, N)
            neg_idx = torch.randint(0, N, (num_neg,), device=DEVICE)
            p_h, p_l = pred[top_idx].unsqueeze(1), pred[neg_idx].unsqueeze(0)
            rank_loss = F.relu(0.4 - (p_h - p_l)).mean()
            
            loss = (mse + RANKING_WEIGHT * rank_loss) / GRAD_ACCUM
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                
                if step_count % 5 == 0 or step_count == 1:
                    with torch.no_grad():
                        y_t_np = target.cpu().numpy()
                        y_p_np = pred.cpu().numpy()
                        p10_s, p20_b = 0.0, 0.0
                        if np.sum(y_t_np) > 0:
                            idx_t, idx_p = np.argsort(y_t_np)[::-1], np.argsort(y_p_np)[::-1]
                            p10_s = len(set(idx_t[:10]) & set(idx_p[:10])) / 10.0
                            p20_b = len(set(idx_t[:20]) & set(idx_p[:20])) / 20.0
                    logging.info(f"   [{region_key.upper()}] E{epoch+1:02d} | Step [{step_count:02d}/{total_steps:02d}] | Loss: {(mse+RANKING_WEIGHT*rank_loss).item():.4f} | P@10: {p10_s*100:.1f}% | P@20: {p20_b*100:.1f}%")
        
        model.eval()
        p10_acc, p20_acc = [], []
        with torch.no_grad():
            for i in range(len(val_X)):
                vx, vy = val_X[i].to(DEVICE), val_y[i].to(DEVICE)
                vpred = model(vx, [adj_geo, adj_conf]).squeeze().cpu().numpy()
                vtrue = vy.squeeze().cpu().numpy()
                if vtrue.sum() > 0:
                    # P@10
                    p_idx10 = np.argsort(vpred)[::-1][:10]
                    t_idx10 = np.argsort(vtrue)[::-1][:10]
                    p10_acc.append(len(set(p_idx10) & set(t_idx10)) / 10.0)
                    
                    # P@20
                    p_idx20 = np.argsort(vpred)[::-1][:20]
                    t_idx20 = np.argsort(vtrue)[::-1][:20]
                    p20_acc.append(len(set(p_idx20) & set(t_idx20)) / 20.0)
        
        mp10 = np.mean(p10_acc or [0])
        mp20 = np.mean(p20_acc or [0])
        logging.info(f"[{region_key.upper()}] Epoch {epoch+1:02d} Final | Val P@10: {mp10*100:.1f}% | P@20: {mp20*100:.1f}%")
        
        # Lógica de Recorde por Região
        is_record = False
        if region_key == 'fortaleza' and mp20 > best_val:
            best_val = mp20
            is_record = True
        elif region_key == 'rmf' and mp10 > best_val:
            best_val = mp10
            is_record = True
        elif region_key == 'interior' and mp20 > best_val:
            best_val = mp20
            is_record = True
            
        if is_record:
            target_metric = "P@20" if region_key != 'rmf' else "P@10"
            torch.save({'model_state_dict': model.state_dict(), 'p10': mp10, 'p20': mp20}, f'models/active/{region_key}_model.pth')
            logging.info(f"🏆 NOVO RECORDE {region_key.upper()} ({target_metric}): {best_val*100:.1f}%")

if __name__ == "__main__":
    # rebuild_datasets() # Já foram construídos e validados
    for reg in ['rmf', 'interior']:
        train_specialist(reg)
