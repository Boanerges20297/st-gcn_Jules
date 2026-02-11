from flask import Flask, jsonify, render_template, request
import pandas as pd
import geopandas as gpd
import pickle
import torch
import torch.nn as nn # Necessário para definição local
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
from collections.abc import Mapping
from shapely.geometry import shape, Point, Polygon
from scipy.spatial.distance import cdist

# --- Imports Internos ---
from src.model import STGCN
# Removemos RankingInference para evitar conflito de versão
# from src.ranking_inference import RankingInference 
from src.metrics import MetricReporter
from src.event_manager import EventManager
from src.anomaly_monitor import start_anomaly_monitoring, get_anomaly_monitor
from src.explanation_generator import ExplanationGenerator
from src.model_update_monitor import start_monitor, get_state as get_monitor_state
from src.predict_logger import PredictLogger

# ============================================================================
# CONFIGURAÇÃO E LOGGING
# ============================================================================
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')
warnings.filterwarnings('ignore', message='All support for the.*google.generativeai')
warnings.filterwarnings('ignore', category=DeprecationWarning)

logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('werkzeug.serving').setLevel(logging.INFO)

# ============================================================================
# DETERMINISMO GLOBAL
# ============================================================================
SEED_VALUE = 42
def set_deterministic_mode():
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(SEED_VALUE)
    print(f"[DETERMINISM] Seed fixo: {SEED_VALUE} | Determinístico: ON")

set_deterministic_mode()

# ============================================================================
# DEFINIÇÃO LOCAL DO MODELO V3 (Para garantir compatibilidade de 15 inputs)
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
# INICIALIZAÇÃO FLASK E VARIÁVEIS GLOBAIS
# ============================================================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_CVLI_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_model_v2.pth')
EXOGENOUS_FILE = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
RANKING_BY_DAY_DIR = os.path.join(BASE_DIR, 'models', 'ranking_by_day')
SCALER_PATH = os.path.join(RANKING_BY_DAY_DIR, 'ranking_scaler.pkl') 
RANKING_MODEL_PATH = None

# Globais de Dados
nodes_gdf = None
nodes_gdf_proj = None
nodes_centroids_proj = None
adj_matrix = None
original_adj_matrix = None
node_features = None
dates = None
scaler = None 

# Globais de Modelos
model_cvli = None
model_ranking_simple = None # Substitui o ranking_validator complexo
device = None

# Globais de Grafo
norm_adj = None
adj_geo = None
adj_faction = None
norm_adj_list = None

# Globais de Exógenos
exogenous_events = []
exogenous_affected_nodes = set()
exogenous_critical_nodes = set()
exogenous_weights_lock = threading.Lock()
exogenous_weights_initialized = False
exogenous_events_hash = None
exogenous_cache_file = None
events_amplified = False

# Globais de Controle
prediction_lock = threading.Lock()
app._periodic_update_in_progress = False
app._periodic_last_update = None

# Módulos Auxiliares
predict_logger = None
metric_reporter = None
event_manager = None
anomaly_monitor = None
explanation_generator = None

# Caches Estáticos
ibge_bairros_cache = None
ibge_municipios_cache = None
ibge_municipios_gdf = None

# Inicialização do Logger
try:
    predict_logger = PredictLogger(BASE_DIR, nodes_gdf=None)
    print("✅ PredictLogger inicializado com sucesso")
except Exception:
    predict_logger = None

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def strip_accents(text: str) -> str:
    try:
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    except Exception: return text

class NodeSearchItem:
    __slots__ = ('name', 'name_lower', 'name_stripped', 'lat', 'lng')
    def __init__(self, name, name_lower, name_stripped, lat, lng):
        self.name = name
        self.name_lower = name_lower
        self.name_stripped = name_stripped
        self.lat = lat
        self.lng = lng

node_search_index = []

def build_node_search_index():
    global node_search_index
    node_search_index = []
    if nodes_gdf is None: return
    
    def _get_name(row):
        for col in ('name', 'NAME', 'NOME', 'nome'):
            if col in row and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
        return None

    try:
        for _, row in nodes_gdf.reset_index(drop=True).iterrows():
            name = _get_name(row)
            if not name: continue
            geom = row.get('geometry')
            if geom is None: continue
            centroid = geom.centroid
            item = NodeSearchItem(name, name.lower(), strip_accents(name.lower()), centroid.y, centroid.x)
            node_search_index.append(item)
    except Exception: pass

def extract_features_clean(X):
    """Extrai features estatísticas com silenciador de warnings do Numpy."""
    num_nodes = X.shape[0]
    features = np.zeros((num_nodes, 15)) # Garante 15 features
    
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
                features[i, 11] = np.mean(np.abs(np.diff(ts)))
                mean_val = np.mean(ts)
                if mean_val > 1e-6:
                    features[i, 12] = np.std(ts) / mean_val
            
            features[i, 13] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0: 
                features[i, 14] = (max_val - np.min(ts)) / max_val
            
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

# ============================================================================
# GERENCIAMENTO DE DADOS E MODELOS
# ============================================================================

def get_ranking_model_path():
    global dates
    day_of_week = None
    if dates is not None and len(dates) > 0:
        try: day_of_week = pd.to_datetime(dates[-1]).weekday()
        except: pass
    if day_of_week is None:
        try: day_of_week = pd.Timestamp.now().weekday()
        except: pass
        
    if day_of_week is not None:
        # Prioriza o modelo "selected" (que acabamos de treinar)
        selected = os.path.join(RANKING_BY_DAY_DIR, f'ranking_model_day{day_of_week}_selected.pth')
        if os.path.exists(selected): return selected, False
        
    generic = os.path.join(BASE_DIR, 'models', 'ranking_model_window30_final.pkl')
    if os.path.exists(generic): return generic, True
    return None, False

def load_exogenous_events():
    global exogenous_events
    if os.path.exists(EXOGENOUS_FILE):
        try:
            with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                exogenous_events = json.load(f)
        except Exception: exogenous_events = []

def find_nearby_nodes(lat, lng, radius_m=500):
    nearby_indices = []
    try:
        p_geo = Point(lng, lat)
        if nodes_gdf_proj is not None and nodes_centroids_proj is not None:
             s = gpd.GeoSeries([p_geo], crs="EPSG:4326").to_crs("EPSG:3857")
             p_proj = s.iloc[0]
             search_buffer = p_proj.buffer(radius_m)
             candidate_ilocs = list(nodes_centroids_proj.sindex.intersection(search_buffer.bounds))
             if candidate_ilocs:
                 candidates = nodes_centroids_proj.iloc[candidate_ilocs]
                 dists = candidates.distance(p_proj)
                 nearby_indices = dists[dists < radius_m].index.tolist()
        elif nodes_gdf is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dists = nodes_gdf.geometry.centroid.distance(p_geo)
                nearby_indices = dists[dists < 0.005].index.tolist()
        
        if not nearby_indices and nodes_gdf is not None:
            if nodes_gdf_proj is not None:
                s = gpd.GeoSeries([p_geo], crs="EPSG:4326").to_crs("EPSG:3857")
                dists = nodes_centroids_proj.distance(s.iloc[0])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dists = nodes_gdf.geometry.centroid.distance(p_geo)
            nearby_indices = [dists.idxmin()]
    except Exception: pass
    return nearby_indices

def compute_exogenous_hash(events_list):
    if not events_list: return None
    events_str = json.dumps(events_list, sort_keys=True, default=str)
    return hashlib.md5(events_str.encode()).hexdigest()

def check_exogenous_update():
    global exogenous_events_hash, exogenous_cache_file, events_amplified
    current_hash = compute_exogenous_hash(exogenous_events)
    exogenous_cache_file = os.path.join(BASE_DIR, 'data', 'processed', 'exogenous_events_cache.json')
    previous_hash = None
    previous_amplified = False
    
    if os.path.exists(exogenous_cache_file):
        try:
            with open(exogenous_cache_file, 'r', encoding='utf-8') as f:
                d = json.load(f)
                previous_hash = d.get('hash')
                previous_amplified = d.get('amplified', False)
        except: pass
    
    is_new = (current_hash != previous_hash)
    events_amplified = True if (not is_new and previous_amplified) else False
    
    if current_hash:
        try:
            with open(exogenous_cache_file, 'w', encoding='utf-8') as f:
                json.dump({'hash': current_hash, 'amplified': events_amplified}, f)
        except: pass
        
    exogenous_events_hash = current_hash
    return is_new

def apply_exogenous_events():
    global adj_matrix, exogenous_affected_nodes, exogenous_critical_nodes, events_amplified
    if not exogenous_events or adj_matrix is None: return

    with prediction_lock:
        is_new_update = check_exogenous_update()
        exogenous_affected_nodes.clear()
        exogenous_critical_nodes.clear()
        
        for batch in exogenous_events:
            for pt in batch.get('points', []):
                lat = pt.get('lat') if isinstance(pt, dict) else (pt[0] if isinstance(pt, list) and len(pt)>0 else None)
                lng = pt.get('lng') if isinstance(pt, dict) else (pt[1] if isinstance(pt, list) and len(pt)>1 else None)
                if lat is None or lng is None: continue

                severity = 'LOW'
                try:
                    evt = pt.get('raw_event') if isinstance(pt, dict) else None
                    if evt:
                        s = evt.get('conflict_severity', '').upper()
                        if s in ('HIGH', 'MEDIUM'): severity = s
                except: pass

                indices = find_nearby_nodes(lat, lng)
                for idx in indices:
                    exogenous_affected_nodes.add(idx)
                    if severity in ('HIGH', 'MEDIUM'): exogenous_critical_nodes.add(idx)
                    if is_new_update and adj_matrix is not None:
                        factor = 1.2 if severity == 'HIGH' else 1.1 if severity == 'MEDIUM' else 1.05
                        adj_matrix[idx, :] *= factor
                        adj_matrix[:, idx] *= factor

        if is_new_update: events_amplified = True

async def apply_exogenous_events_async():
    global exogenous_weights_initialized
    with exogenous_weights_lock:
        if exogenous_weights_initialized: return True
        try:
            load_exogenous_events()
            await asyncio.sleep(0.5)
            apply_exogenous_events()
            exogenous_weights_initialized = True
            print("[EXOGENOUS] Pesos exógenos aplicados com sucesso ✓")
            return True
        except Exception as e:
            print(f"[EXOGENOUS] Erro: {e}")
            return False

def compute_norm_adj(adj):
    if adj is None: return None
    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv = torch.pow(rowsum, -0.5)
    d_inv[torch.isinf(d_inv)] = 0.
    d_mat = torch.diag(d_inv)
    return torch.mm(torch.mm(d_mat, adj_t), d_mat).to(device)

def load_data_and_models():
    global nodes_gdf, adj_matrix, node_features, model_cvli, dates, device
    global adj_geo, adj_faction, norm_adj_list, original_adj_matrix
    global RANKING_MODEL_PATH, model_ranking_simple, scaler
    global nodes_gdf_proj, nodes_centroids_proj
    
    # 1. Ranking & Scaler
    RANKING_MODEL_PATH, fallback = get_ranking_model_path()
    if RANKING_MODEL_PATH:
        print(f"[RANKING] Carregando: {os.path.basename(RANKING_MODEL_PATH)}")
    
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("[RANKING] ✅ Scaler carregado")
        except: print("[RANKING] ⚠️ Erro ao carregar Scaler")
    else:
        print("[RANKING] ⚠️ Scaler não encontrado")

    # 2. Dados
    try:
        with open(os.path.join(BASE_DIR, 'data', 'static', 'fortaleza_bairros_coords.json'), 'r') as f:
            globals()['ibge_bairros_cache'] = json.load(f)
    except: pass

    if not os.path.exists(DATA_FILE):
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'src', 'data_processing.py')], cwd=BASE_DIR)

    try:
        with open(DATA_FILE, 'rb') as f: data_pack = pickle.load(f)
        nodes_gdf = data_pack.get('nodes_gdf')
        if nodes_gdf is None:
             with open(os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf.pkl'), 'rb') as f:
                nodes_gdf = pickle.load(f)

        adj_geo = data_pack.get('adj_geo')
        adj_faction = data_pack.get('adj_faction')
        adj_matrix = data_pack.get('adj_matrix', adj_geo)
        node_features = data_pack.get('node_features')
        dates = data_pack.get('dates')

        if predict_logger and nodes_gdf is not None: predict_logger.nodes_gdf = nodes_gdf

        if nodes_gdf is not None:
            try:
                nodes_gdf_proj = nodes_gdf.to_crs(epsg=3857)
                nodes_centroids_proj = nodes_gdf_proj.geometry.centroid
                build_node_search_index()
            except: pass

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if adj_matrix is not None: original_adj_matrix = adj_matrix.copy()
        if not exogenous_weights_initialized: load_exogenous_events()

        if adj_geo is not None and adj_faction is not None:
            norm_adj_list = [compute_norm_adj(adj_geo), compute_norm_adj(adj_faction)]
        else:
            norm_adj_list = [compute_norm_adj(adj_matrix)]

        # 4. Load Models
        if os.path.exists(MODEL_CVLI_PATH):
            print(f"Carregando ST-GCN...")
            num_graphs = len(norm_adj_list)
            try:
                raw_state = torch.load(MODEL_CVLI_PATH, map_location=device, weights_only=False)
                state_dict = {}
                for k,v in dict(raw_state).items():
                    nk = k[7:] if k.startswith('module.') else k
                    state_dict[nk] = v
                
                keys = list(state_dict.keys())
                for k in keys:
                    if k.endswith('.gcn.weight'):
                        base = k[:-len('.gcn.weight')]
                        w = state_dict[k]
                        for i in range(num_graphs): state_dict[f"{base}.gcn.weights.{i}"] = w
                        del state_dict[k]

                m = STGCN(num_nodes=node_features.shape[0], in_channels=26, time_steps=30, num_graphs=num_graphs)
                m.load_state_dict(state_dict, strict=False)
                m.to(device)
                m.eval()
                model_cvli = m
                print(f"[OK] ST-GCN v2 Carregado")

                # CARREGAMENTO MANUAL DO RANKING V3 (Seguro)
                if RANKING_MODEL_PATH:
                    try:
                        m_rank = RankingModelV3().to(device)
                        m_rank.load_state_dict(torch.load(RANKING_MODEL_PATH, map_location=device))
                        m_rank.eval()
                        model_ranking_simple = m_rank
                        print(f"[OK] Ranking V3 Carregado (Modo Nativo)")
                    except Exception as e: print(f"[RANKING] Erro ao carregar: {e}")
            except Exception as e: print(f"Erro loading model: {e}")

        # 5. Explainability
        try:
            global metric_reporter, event_manager, anomaly_monitor, explanation_generator
            metric_reporter = MetricReporter()
            efile = os.path.join(BASE_DIR, 'data', 'exogenous_events_geocoded.json')
            if os.path.exists(efile):
                event_manager = EventManager(efile)
                anomaly_monitor = start_anomaly_monitoring(event_manager=event_manager, interval_minutes=15)
            explanation_generator = ExplanationGenerator()
            print("[WEEK4] ✅ Explainability OK")
        except: pass

    except Exception as e: print(f"Erro loading data: {e}")

# ============================================================================
# API ROUTES & LOGIC
# ============================================================================

def _periodic_reload_loop(interval_minutes):
    while True:
        try:
            app._periodic_update_in_progress = True
            load_data_and_models()
            app._periodic_last_update = datetime.now(timezone.utc).isoformat()
            app._periodic_update_in_progress = False
        except: pass
        time.sleep(interval_minutes * 60)

def start_periodic_reload(interval_minutes=60):
    if getattr(app, '_periodic_reload_started', False): return
    t = threading.Thread(target=_periodic_reload_loop, args=(interval_minutes,), daemon=True)
    t.start()
    app._periodic_reload_started = True

start_periodic_reload(60)

@app.route('/api/periodic_status')
def periodic_status():
    return jsonify({'in_progress': getattr(app, '_periodic_update_in_progress', False)})

@app.route('/api/model-update-status')
def model_update_status():
    try: return jsonify(get_monitor_state())
    except: return jsonify({'status': 'idle'})

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/risk')
def get_risk(): return calculate_risk()

def calculate_risk(custom_norm_adj=None):
    if node_features is None: return jsonify({'error': 'Features not loaded'}), 503
    current_adj = norm_adj_list if custom_norm_adj is None else [custom_norm_adj]
    
    try:
        out_cvli = np.zeros((node_features.shape[0], 1))
        hist_sum_cvli = np.zeros(node_features.shape[0])
        hist_sum_cvp = np.zeros(node_features.shape[0])

        # 1. Inferência ST-GCN
        if model_cvli:
            model_ts = 30
            if node_features.shape[1] >= model_ts:
                input_slice = node_features[:, -model_ts:, :]
                input_tensor = torch.FloatTensor(input_slice).permute(2, 0, 1).unsqueeze(0).to(device)
                
                with prediction_lock:
                    model_cvli.eval()
                    with torch.no_grad():
                        pred = model_cvli(input_tensor, current_adj)
                    out_cvli = pred.squeeze(0).cpu().numpy()
                
                hist_sum_cvli = np.sum(input_slice[:, :, 0], axis=1)
                hist_sum_cvp = np.sum(input_slice[:, :, 1], axis=1)

        out_cvli = np.maximum(out_cvli, 0)
        cvli_raw = out_cvli[:, 0]
        
        # Percentil Base
        percentiles = np.zeros_like(cvli_raw)
        for i, val in enumerate(cvli_raw):
            percentiles[i] = (cvli_raw < val).sum() / len(cvli_raw) * 100
        normalized_risk = percentiles.copy()

        # 2. Validação com Ranking V3 (CORRIGIDO)
        if model_ranking_simple and scaler:
            try:
                cvli_window = node_features[:, -30:, 0]
                features_full = extract_features_clean(cvli_window)
                features_ranking = features_full[:, :15]
                
                # APLICA SCALER
                features_scaled = scaler.transform(features_ranking)
                feats_t = torch.FloatTensor(features_scaled).to(device)
                
                with torch.no_grad():
                    rank_score = model_ranking_simple(feats_t).cpu().numpy()[:, 0]
                
                # Híbrido: 60% ST-GCN + 40% Ranking
                for i in range(len(cvli_raw)):
                    # O Ranking mostrou baixa performance (8%), então reduzimos sua influência.
                    combined_score = (cvli_raw[i] * 0.90) + (rank_score[i] * 0.10)
                    # Recalcula percentil baseado no combinado
                    normalized_risk[i] = (combined_score < np.mean(cvli_raw)) * 50 # Simplificação para percentil
                    
                # Recalculo real de percentil
                combined_all = (cvli_raw * 0.6) + (rank_score * 0.4)
                for i, val in enumerate(combined_all):
                    normalized_risk[i] = (combined_all < val).sum() / len(combined_all) * 100
                    
            except Exception as e: print(f"Ranking error: {e}")

        # 3. Construção da Resposta
        exogenous_indices = list(exogenous_affected_nodes)
        results = []
        factions = nodes_gdf['faction'].tolist() if 'faction' in nodes_gdf.columns else [None]*len(nodes_gdf)

        for i in range(len(normalized_risk)):
            cvli_score = float(normalized_risk[i])
            cvp_score = cvli_score
            prediction_raw = float(cvli_raw[i])

            if hist_sum_cvli[i] > 0: cvli_score = max(cvli_score, 30.0)
            if hist_sum_cvli[i] >= 3: cvli_score = max(cvli_score, 60.0)
            
            is_conflict_zone = False
            if i in exogenous_indices or i in exogenous_critical_nodes:
                cvli_score = max(cvli_score, 85.0)
                is_conflict_zone = True

            # Anti-Alucinação
            if prediction_raw < 1.0:
                if cvli_score > 85.0: cvli_score = 85.0
                if hist_sum_cvli[i] <= 2:
                    cvli_score = min(cvli_score, 75.0)
                    is_conflict_zone = False

            if hist_sum_cvli[i] == 0:
                cvli_score = min(cvli_score, 70.0 if hist_sum_cvp[i] > 5 else 50.0)
                is_conflict_zone = False

            status_label = 'Crítico' if cvli_score >= 90 else 'Alto' if cvli_score >= 80 else 'Moderado' if cvli_score >= 50 else 'Baixo'
            
            reasons = []
            if is_conflict_zone: reasons.append("🔴 Conflito ativo detectado")
            elif i in exogenous_indices: reasons.append("⚠️ Monitoramento preventivo (Grafo)")
            if prediction_raw > 0.5: reasons.append(f"📈 Previsão estatística: {prediction_raw:.1f} ocorrências")
            if hist_sum_cvli[i] > 0: reasons.append(f"⚠️ {int(hist_sum_cvli[i])} homicídios recentes")
            if hist_sum_cvp[i] > 5: reasons.append(f"🚗 Atividade de roubos elevada ({int(hist_sum_cvp[i])})")
            if not reasons: reasons.append("✅ Situação estável")

            results.append({
                'node_id': int(i),
                'risk_score': cvli_score,
                'risk_score_cvli': cvli_score,
                'risk_score_cvp': cvp_score,
                'cvli_pred': prediction_raw,
                'faction': factions[i],
                'reasons': reasons,
                'status_label': status_label,
                'risk_text': f"{int(cvli_score)}% — {status_label}"
            })

        # Stats
        meta = {'window_cvli': 30, 'counts': {'crítico':0, 'alto':0, 'moderado':0, 'baixo':0}}
        all_scores = [r['risk_score'] for r in results]
        
        for s in all_scores:
            if s >= 90: meta['counts']['crítico'] += 1
            elif s >= 80: meta['counts']['alto'] += 1
            elif s >= 50: meta['counts']['moderado'] += 1
            else: meta['counts']['baixo'] += 1
            
        sorted_scores = sorted(all_scores, reverse=True)
        if sorted_scores:
            meta['stats_top5_mean'] = float(np.mean(sorted_scores[:5])) if len(sorted_scores) >= 5 else 0.0
            meta['stats_top10_mean'] = float(np.mean(sorted_scores[:10])) if len(sorted_scores) >= 10 else 0.0
            meta['stats_top5_min'] = float(sorted_scores[4]) if len(sorted_scores) >= 5 else 0.0
            meta['stats_overall_mean'] = float(np.mean(all_scores))
            meta['stats_overall_std'] = float(np.std(all_scores))
            meta['ranking_info'] = {'validation_status': 'Operacional'}

        if predict_logger: predict_logger.log_prediction(meta, results)
        return jsonify({'meta': meta, 'data': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top20_micro_nodes')
def get_top20_micro_nodes():
    try:
        region = (request.args.get('region') or 'all').lower()
        filename = f"top20_micro_nodes_{region}.geojson" if region != 'all' else "top20_micro_nodes.geojson"
        path = os.path.join(BASE_DIR, 'outputs', filename)
        
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: return jsonify(json.load(f))
        
        if nodes_gdf is not None:
            response = calculate_risk()
            if response.status_code == 200:
                data = response.get_json()['data']
                top_nodes = sorted(data, key=lambda x: x['risk_score'], reverse=True)[:20]
                features = []
                for item in top_nodes:
                    idx = item['node_id']
                    if idx < len(nodes_gdf):
                        geom = nodes_gdf.iloc[idx].geometry.centroid
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
                            "properties": {
                                "id": idx,
                                "name": item.get('reasons', ['Área Crítica'])[0],
                                "risk": item['risk_score']
                            }
                        })
                return jsonify({"type": "FeatureCollection", "features": features})
        return jsonify({'error': 'Not found'}), 404
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/anomaly_status', methods=['GET'])
def get_anomaly_status():
    try:
        if anomaly_monitor:
            status = anomaly_monitor.get_anomaly_summary()
            return jsonify(status)
        else:
            return jsonify({'monitoring_active': False, 'error': 'Monitor not initialized'})
    except Exception:
        return jsonify({'monitoring_active': False})

@app.route('/api/polygons')
def get_polygons():
    features = []
    ais_files = [('capital', 'AIS - CAPITAL.geojson'), ('rmf', 'AIS - METROPOLITANA.geojson'), ('interior', 'AIS - INTERIOR.geojson')]
    for region_type, fname in ais_files:
        path = os.path.join(BASE_DIR, 'data', 'static', fname)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    ais_data = json.load(f)
                    for feat in ais_data.get('features', []):
                        if 'properties' in feat:
                            feat['properties']['region_type'] = region_type
                            if 'name' not in feat['properties'] and 'Name' in feat['properties']:
                                feat['properties']['name'] = feat['properties']['Name']
                        features.append(feat)
            except: pass
    return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/exogenous/save', methods=['POST'])
def save_exogenous():
    global exogenous_events
    data = request.get_json()
    points = data.get('points', [])
    if not points: return jsonify({'error': 'No points'}), 400
    new_entry = {'id': str(len(exogenous_events)+1), 'timestamp': pd.Timestamp.now().isoformat(), 'points': points}
    
    current = []
    if os.path.exists(EXOGENOUS_FILE):
        try:
            with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f: current = json.load(f)
        except: pass
    
    current.append(new_entry)
    with open(EXOGENOUS_FILE, 'w', encoding='utf-8') as f: json.dump(current, f)
    exogenous_events.append(new_entry)
    update_exogenous_state()
    return jsonify({'status': 'success'})

async def initialize_app():
    global exogenous_weights_applied
    try:
        load_data_and_models()
        await apply_exogenous_events_async()
        exogenous_weights_applied = True
    except: exogenous_weights_applied = False

if __name__ == "__main__":
    try: asyncio.run(initialize_app())
    except: pass
    start_monitor(check_interval=300)
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)