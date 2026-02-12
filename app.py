from flask import Flask, jsonify, render_template, request
import pandas as pd
import geopandas as gpd
import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import copy
import re
import json
import hashlib
import warnings
import logging
import asyncio
import threading
import time
import unicodedata
from datetime import datetime, timezone
from shapely.geometry import Point

# --- Imports Internos ---
from src.model import STGCN
# Seus módulos internos
from src.metrics import MetricReporter
from src.event_manager import EventManager
from src.anomaly_monitor import start_anomaly_monitoring
from src.explanation_generator import ExplanationGenerator
from src.model_update_monitor import start_monitor, get_state as get_monitor_state
from src.predict_logger import PredictLogger

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
warnings.filterwarnings('ignore')
logging.getLogger('werkzeug').setLevel(logging.INFO)

SEED_VALUE = 42
def set_deterministic_mode():
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
    import random
    random.seed(SEED_VALUE)

set_deterministic_mode()

# ============================================================================
# CLASSE DE RANKING LOCAL
# ============================================================================
class RankingModelV3(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128):
        super(RankingModelV3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# VARIÁVEIS GLOBAIS
# ============================================================================
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_CVLI_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_model_v2.pth')
EXOGENOUS_FILE = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
RANKING_BY_DAY_DIR = os.path.join(BASE_DIR, 'models', 'ranking_by_day')
SCALER_PATH = os.path.join(RANKING_BY_DAY_DIR, 'ranking_scaler.pkl')

# Estado
nodes_gdf, nodes_gdf_proj, nodes_centroids_proj = None, None, None
adj_matrix, original_adj_matrix, node_features, dates = None, None, None, None
model_cvli, model_ranking_simple, scaler, device = None, None, None, None
norm_adj_list, exogenous_events = None, []
exogenous_affected_nodes, exogenous_critical_nodes = set(), set()
exogenous_weights_lock = threading.Lock()
exogenous_weights_initialized = False
prediction_lock = threading.Lock()

# Auxiliares
predict_logger, metric_reporter, event_manager, anomaly_monitor = None, None, None, None

try:
    predict_logger = PredictLogger(BASE_DIR, nodes_gdf=None)
    print("✅ PredictLogger inicializado")
except: pass

# ============================================================================
# FUNÇÕES DE SUPORTE
# ============================================================================
def extract_features_clean(X):
    """Extrai 15 features para o Ranking V3."""
    num_nodes = X.shape[0]
    features = np.zeros((num_nodes, 15))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(num_nodes):
            ts = X[i, :]
            features[i, 0] = np.mean(ts)
            features[i, 1] = np.std(ts)
            features[i, 2] = np.max(ts)
            features[i, 3] = np.min(ts)
            if len(ts) > 0:
                features[i, 4] = np.sum(ts > 0) / len(ts)
                features[i, 5] = np.sum(ts) / len(ts)
            if len(ts) > 5:
                features[i, 6] = np.mean(ts[-5:]) - np.mean(ts[:5])
            if len(ts) > 1:
                features[i, 7] = np.mean(np.abs(np.diff(ts)))
            features[i, 8] = np.mean(ts[-3:]) if len(ts) >= 3 else 0
            features[i, 9] = np.mean(ts[-7:]) if len(ts) >= 7 else 0
            features[i, 10] = np.mean(ts[-14:]) if len(ts) >= 14 else 0
            if len(ts) > 1:
                mean_val = np.mean(ts)
                if mean_val > 1e-6: features[i, 11] = np.std(ts) / mean_val
            features[i, 12] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0: features[i, 13] = (max_val - np.min(ts)) / max_val
    return np.nan_to_num(features)

def get_ranking_model_path():
    day_of_week = datetime.now().weekday()
    path = os.path.join(RANKING_BY_DAY_DIR, f'ranking_model_day{day_of_week}_selected.pth')
    if os.path.exists(path): return path
    return None

def compute_norm_adj(adj):
    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv = torch.pow(rowsum, -0.5)
    d_inv[torch.isinf(d_inv)] = 0.
    d_mat = torch.diag(d_inv)
    return torch.mm(torch.mm(d_mat, adj_t), d_mat).to(device)

def load_data_and_models():
    global nodes_gdf, node_features, model_cvli, model_ranking_simple, scaler, device
    global norm_adj_list, adj_matrix, dates, nodes_gdf_proj, nodes_centroids_proj
    
    # 1. Carregar Scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        print("✅ Scaler carregado")
    else: print("⚠️ Scaler não encontrado")

    # 2. Carregar Dados
    if not os.path.exists(DATA_FILE): return
    with open(DATA_FILE, 'rb') as f: data_pack = pickle.load(f)
    
    nodes_gdf = data_pack.get('nodes_gdf')
    adj_geo = data_pack.get('adj_geo')
    adj_faction = data_pack.get('adj_faction')
    node_features = data_pack.get('node_features')
    dates = data_pack.get('dates')
    
    if predict_logger and nodes_gdf is not None: predict_logger.nodes_gdf = nodes_gdf

    # Projeção Geo
    if nodes_gdf is not None:
        try:
            nodes_gdf_proj = nodes_gdf.to_crs(epsg=3857)
            nodes_centroids_proj = nodes_gdf_proj.geometry.centroid
            build_node_search_index()
        except: pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Grafos
    norm_adj_list = [compute_norm_adj(adj_geo), compute_norm_adj(adj_faction)]

    # 3. Carregar ST-GCN
    if os.path.exists(MODEL_CVLI_PATH):
        try:
            raw_state = torch.load(MODEL_CVLI_PATH, map_location=device)
            state_dict = {}
            for k,v in raw_state.items():
                nk = k.replace('module.', '')
                if nk.endswith('.gcn.weight'):
                    base = nk[:-len('.gcn.weight')]
                    state_dict[f"{base}.gcn.weights.0"] = v
                    state_dict[f"{base}.gcn.weights.1"] = v
                else: state_dict[nk] = v
            
            model_cvli = STGCN(num_nodes=node_features.shape[0], in_channels=26, time_steps=30, num_graphs=2)
            model_cvli.load_state_dict(state_dict, strict=False)
            model_cvli.to(device).eval()
            print("✅ ST-GCN Carregado")
        except Exception as e: print(f"❌ Erro ST-GCN: {e}")

    # 4. Carregar Ranking V3 (Local)
    rank_path = get_ranking_model_path()
    if rank_path:
        try:
            m_rank = RankingModelV3(input_dim=15).to(device)
            m_rank.load_state_dict(torch.load(rank_path, map_location=device))
            m_rank.eval()
            model_ranking_simple = m_rank
            print(f"✅ Ranking V3 Carregado: {os.path.basename(rank_path)}")
        except Exception as e: 
            print(f"❌ Erro Ranking: {e}")
            model_ranking_simple = None

    # 5. Módulos
    try:
        global metric_reporter, event_manager, anomaly_monitor
        metric_reporter = MetricReporter()
        if os.path.exists(os.path.join(BASE_DIR, 'data', 'exogenous_events_geocoded.json')):
            event_manager = EventManager(os.path.join(BASE_DIR, 'data', 'exogenous_events_geocoded.json'))
            anomaly_monitor = start_anomaly_monitoring(event_manager, interval_minutes=15)
            print("✅ Monitoramento Ativo")
    except: pass

# ============================================================================
# ROTAS
# ============================================================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/risk')
def get_risk():
    if node_features is None: return jsonify({'error': 'Loading...'}), 503
    
    try:
        # 1. Inferência ST-GCN
        model_ts = 30
        input_slice = node_features[:, -model_ts:, :]
        input_tensor = torch.FloatTensor(input_slice).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model_cvli(input_tensor, norm_adj_list)
            stgcn_score = pred.squeeze(0).cpu().numpy()[:, 0]
            stgcn_score = np.maximum(stgcn_score, 0)

        hist_sum = np.sum(input_slice[:, :, 0], axis=1)

        # 2. Ranking V3
        rank_score = np.zeros_like(stgcn_score)
        if model_ranking_simple and scaler:
            try:
                feats = extract_features_clean(input_slice[:, :, 0])
                feats_scaled = scaler.transform(feats)
                feats_t = torch.FloatTensor(feats_scaled).to(device)
                with torch.no_grad():
                    rank_score = model_ranking_simple(feats_t).cpu().numpy()[:, 0]
            except Exception as e: print(f"Ranking Fail: {e}")

        # 3. Fusão 90/10 com SIGMOID (Evita inflação de risco)
        def sigmoid_norm(v):
            # Transforma score bruto em probabilidade 0-1 de forma suave
            return 1 / (1 + np.exp(-v))
        
        s_norm = sigmoid_norm(stgcn_score)
        r_norm = sigmoid_norm(rank_score)
        
        # Combinação ponderada
        combined = (s_norm * 0.90) + (r_norm * 0.10)
        
        # Converte para escala 0-100 para o dashboard
        final_risk = combined * 100

        # Montar JSON
        results = []
        factions = nodes_gdf['faction'].tolist() if 'faction' in nodes_gdf.columns else [None]*len(nodes_gdf)
        
        for i in range(len(final_risk)):
            score = float(final_risk[i])
            raw_pred = float(stgcn_score[i])
            
            # Anti-Alucinação (Trava de Segurança)
            if raw_pred < 0.5: # Se a previsão absoluta é muito baixa
                score = min(score, 45.0) # Força risco Baixo/Moderado
            elif raw_pred < 1.0:
                score = min(score, 75.0) # Força risco Alto (mas não Crítico)
            
            if hist_sum[i] == 0 and score > 60:
                score = 60.0 # Sem histórico recente não pode ser crítico

            status = 'Crítico' if score >= 85 else 'Alto' if score >= 70 else 'Moderado' if score >= 40 else 'Baixo'
            
            reasons = []
            if raw_pred > 0.5: reasons.append(f"Previsão: {raw_pred:.1f}")
            if hist_sum[i] > 0: reasons.append(f"{int(hist_sum[i])} crimes recentes")
            if not reasons: reasons.append("Estável")

            results.append({
                'node_id': i,
                'risk_score': score,
                'cvli_pred': raw_pred,
                'faction': factions[i],
                'reasons': reasons,
                'status_label': status,
                'risk_text': f"{int(score)}% — {status}"
            })

        # --- ESTATÍSTICAS REAIS (Sem traços) ---
        meta = {'counts': {'crítico':0, 'alto':0, 'moderado':0, 'baixo':0}}
        scores = [r['risk_score'] for r in results]
        
        # Limpeza de NaNs
        scores = np.nan_to_num(scores)
        
        for s in scores:
            if s >= 85: meta['counts']['crítico'] += 1
            elif s >= 70: meta['counts']['alto'] += 1
            elif s >= 40: meta['counts']['moderado'] += 1
            else: meta['counts']['baixo'] += 1
            
        s_sort = sorted(scores, reverse=True)
        
        # Preenchimento garantido
        meta['stats_top5_mean'] = float(np.mean(s_sort[:5])) if len(s_sort) >= 5 else 0.0
        meta['stats_top10_mean'] = float(np.mean(s_sort[:10])) if len(s_sort) >= 10 else 0.0
        meta['stats_top5_min'] = float(s_sort[4]) if len(s_sort) >= 5 else 0.0
        meta['stats_overall_mean'] = float(np.mean(scores)) if len(scores) > 0 else 0.0
        meta['stats_overall_std'] = float(np.std(scores)) if len(scores) > 0 else 0.0
        
        meta['ranking_info'] = {'status': 'Híbrido 90/10 (Sigmoid)'}

        if predict_logger: predict_logger.log_prediction(meta, results)
        return jsonify({'meta': meta, 'data': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top20_micro_nodes')
def get_top20():
    # Fallback dinâmico
    if nodes_gdf is not None:
        try:
            resp = calculate_risk()
            data = resp.get_json()['data']
            top = sorted(data, key=lambda x: x['risk_score'], reverse=True)[:20]
            feats = []
            for item in top:
                idx = item['node_id']
                if idx < len(nodes_gdf):
                    geom = nodes_gdf.iloc[idx].geometry.centroid
                    feats.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
                        "properties": {"name": f"Risco {int(item['risk_score'])}%", "risk": item['risk_score']}
                    })
            return jsonify({"type": "FeatureCollection", "features": feats})
        except: pass
    return jsonify({'features': []})

# --- ROTA DE POLÍGONOS CORRIGIDA (CARREGA TUDO) ---
@app.route('/api/polygons')
def get_polygons():
    features = []
    # Lista de arquivos + Tag de região
    ais_files = [
        ('capital', 'AIS - CAPITAL.geojson'),
        ('rmf', 'AIS - METROPOLITANA.geojson'),
        ('interior', 'AIS - INTERIOR.geojson')
    ]
    
    for region_type, fname in ais_files:
        path = os.path.join(BASE_DIR, 'data', 'static', fname)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for feat in data.get('features', []):
                        if 'properties' in feat:
                            # ETIQUETA OBRIGATÓRIA PARA O FILTRO FUNCIONAR
                            feat['properties']['region_type'] = region_type
                            
                            # Normaliza nome para exibição
                            if 'name' not in feat['properties']:
                                for key in ['Name', 'NOME', 'bairro', 'municipio', 'NM_MUN']:
                                    if key in feat['properties']:
                                        feat['properties']['name'] = feat['properties'][key]
                                        break
                        features.append(feat)
            except Exception as e:
                print(f"Erro lendo {fname}: {e}")
            
    return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/anomaly_status')
def anomaly_status():
    if anomaly_monitor: 
        # Força atualização para não ficar "congelado"
        return jsonify(anomaly_monitor.get_anomaly_summary())
    return jsonify({'monitoring_active': False, 'status': 'Monitor Offline'})

@app.route('/api/model-update-status')
def model_status(): return jsonify(get_monitor_state())

if __name__ == "__main__":
    load_data_and_models()
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)