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

# Janela padrão para considerar eventos exógenos (em dias).
# Pode ser sobrescrita via variável de ambiente EXOGENOUS_DAYS_BACK.
EXOGENOUS_DAYS_BACK = int(os.getenv('EXOGENOUS_DAYS_BACK', '14'))

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


def compute_predictions():
    """Compute model predictions and return (meta, results, raw arrays).
    Returns: (meta_dict, results_list, final_risk_array, stgcn_score, hist_sum)
    """
    # 1. Inferência ST-GCN
    model_ts = 30
    input_slice = node_features[:, -model_ts:, :]
    input_tensor = torch.FloatTensor(input_slice).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model_cvli(input_tensor, norm_adj_list)
        stgcn_score = pred.squeeze(0).cpu().numpy()[:, 0]
        stgcn_score = np.maximum(stgcn_score, 0)

    hist_sum = np.sum(input_slice[:, :, 0], axis=1)

    # 2. Ranking V3 (opcional)
    rank_score = np.zeros_like(stgcn_score)
    ranking_used = False
    if model_ranking_simple and scaler:
        try:
            feats = extract_features_clean(input_slice[:, :, 0])
            feats_scaled = scaler.transform(feats)
            feats_t = torch.FloatTensor(feats_scaled).to(device)
            with torch.no_grad():
                rank_score = model_ranking_simple(feats_t).cpu().numpy()[:, 0]
            ranking_used = not np.allclose(rank_score, 0.0)
        except Exception as e:
            print(f"Ranking Fail: {e}")
            rank_score = np.zeros_like(stgcn_score)
            ranking_used = False

    # 3. Normalização por percentil + blend 70/30 (ST-GCN / Ranking)
    def to_percentile(arr: np.ndarray) -> np.ndarray:
        """Converte vetor em percentis [0, 100], estável mesmo com empates."""
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return arr
        # ranking baseado na ordenação (0 .. N-1)
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr), dtype=float)
        # evitar divisão por zero quando N=1
        denom = max(len(arr) - 1, 1)
        return (ranks / denom) * 100.0

    stgcn_pct = to_percentile(stgcn_score)

    # Incorporar informação de histórico recente (soma de CVLI na janela)
    if np.any(hist_sum > 0):
        hist_pct = to_percentile(hist_sum)
    else:
        hist_pct = np.zeros_like(hist_sum, dtype=float)

    if ranking_used:
        rank_pct = to_percentile(rank_score)
        # Blend mais rico: ST-GCN (60%) + Ranking (25%) + Histórico (15%)
        final_risk = (0.60 * stgcn_pct) + (0.25 * rank_pct) + (0.15 * hist_pct)
        ranking_source = "stgcn_percentile+ranking_v3+history"
        ranking_status = "Híbrido 60/25/15 (ST-GCN/Ranking/Histórico)"
    else:
        # Sem ranking: ST-GCN (75%) + Histórico (25%)
        final_risk = (0.75 * stgcn_pct) + (0.25 * hist_pct)
        ranking_source = "stgcn_percentile+history"
        ranking_status = "ST-GCN+Histórico (Percentil, sem ranking)"

    # Apply local exogenous adjustments: map recent exogenous points to nearest nodes
    try:
        def _compute_local_severity(days_back=1):
            """Retorna severidade exógena por nó em [0,1].
            Usa, por padrão, `exogenous_events.json` (CIOPS estruturado).
            """
            path = EXOGENOUS_FILE
            if not os.path.exists(path):
                # fallback para arquivo geocodificado antigo, se existir
                alt = os.path.join(BASE_DIR, 'data', 'exogenous_events_geocoded.json')
                if os.path.exists(alt):
                    path = alt
                else:
                    return np.zeros(len(nodes_gdf))

            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
            except Exception:
                return np.zeros(len(nodes_gdf))

            # Build KDTree of node coords (lat, lon)
            try:
                from scipy.spatial import KDTree
            except Exception:
                return np.zeros(len(nodes_gdf))

            node_coords = [(g.y, g.x) for g in nodes_gdf.geometry.centroid]
            tree = KDTree(node_coords)

            severities = np.zeros(len(nodes_gdf), dtype=float)
            cutoff_date = None
            # accept both list and dict-wrapped files
            events_list = data if isinstance(data, list) else data.get('events', []) if isinstance(data, dict) else []

            for ev in events_list:
                # parse timestamp if present
                tstr = ev.get('timestamp') or ev.get('date') or ev.get('event_date')
                if tstr and days_back is not None:
                    try:
                        from datetime import datetime, timedelta
                        # try ISO then DD/MM/YYYY
                        try:
                            ed = datetime.fromisoformat(tstr)
                        except Exception:
                            try:
                                ed = datetime.strptime(tstr, '%d/%m/%Y')
                            except Exception:
                                ed = None
                        if ed is not None:
                            if cutoff_date is None:
                                cutoff_date = datetime.now() - timedelta(days=days_back)
                            if ed < cutoff_date:
                                continue
                    except Exception:
                        pass

                for pt in ev.get('points', []) if isinstance(ev.get('points', []), list) else []:
                    lat = pt.get('lat')
                    lng = pt.get('lng')
                    if lat is None or lng is None:
                        continue
                    try:
                        dist, idx = tree.query((float(lat), float(lng)))
                        # determine severity from raw_event if available
                        sev = 0.5
                        raw = pt.get('raw_event') or ev.get('raw_event') or ev
                        if isinstance(raw, dict):
                            sev = float(raw.get('conflict_severity') or raw.get('severity') or raw.get('conflict') or 0.5)
                            # normalize string labels if present
                            if isinstance(sev, str):
                                mapping = {'LOW': 0.2, 'MEDIUM': 0.5, 'HIGH': 0.8, 'CRITICAL': 1.0}
                                sev = mapping.get(sev.upper(), 0.5)
                        severities[int(idx)] = max(severities[int(idx)], min(max(sev, 0.0), 1.0))
                    except Exception:
                        continue

            return severities

        # Usar janela configurável (por padrão 14 dias) para exógenos
        local_sev = _compute_local_severity(days_back=EXOGENOUS_DAYS_BACK)
        if local_sev is None:
            local_sev = np.zeros(len(nodes_gdf))
        # Multiplicador local mais forte: até +120% quando severity==1.0
        local_factor = 1.0 + (local_sev * 1.2)
        final_risk = np.clip(final_risk * local_factor, 0.0, 100.0)

        # Boost mínimo para nós com exógenos muito fortes:
        # - severidade >= 0.9  -> pelo menos 90% de risco
        # - severidade >= 0.7  -> pelo menos 80% de risco
        high_crit_mask = local_sev >= 0.9
        high_mask = (local_sev >= 0.7) & ~high_crit_mask

        if np.any(high_crit_mask):
            final_risk = np.where(high_crit_mask & (final_risk < 90.0), 90.0, final_risk)
        if np.any(high_mask):
            final_risk = np.where(high_mask & (final_risk < 80.0), 80.0, final_risk)
    except Exception as e:
        print(f"[WARN] local exogenous adjustment failed: {e}")

    # Montar JSON básico (sem ajustes de anomalia)
    results = []
    factions = nodes_gdf['faction'].tolist() if 'faction' in nodes_gdf.columns else [None]*len(nodes_gdf)

    for i in range(len(final_risk)):
        score = float(final_risk[i])
        raw_pred = float(stgcn_score[i])

        if raw_pred < 0.5:
            score = min(score, 45.0)
        elif raw_pred < 1.0:
            score = min(score, 75.0)

        if hist_sum[i] == 0 and score > 60:
            score = 60.0

        status = 'Crítico' if score >= 90 else 'Alto' if score >= 80 else 'Moderado' if score >= 50 else 'Baixo'

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

    meta = {
        'counts': {
            'crítico': 0,
            'alto': 0,
            'moderado': 0,
            'baixo': 0,
            'sem risco': 0,
        }
    }
    scores = [r['risk_score'] for r in results]
    scores = np.nan_to_num(scores)
    for s in scores:
        if s >= 90:
            meta['counts']['crítico'] += 1
        elif s >= 80:
            meta['counts']['alto'] += 1
        elif s >= 50:
            meta['counts']['moderado'] += 1
        elif s >= 20:
            meta['counts']['baixo'] += 1
        else:
            meta['counts']['sem risco'] += 1

    s_sort = sorted(scores, reverse=True)
    meta['stats_top5_mean'] = float(np.mean(s_sort[:5])) if len(s_sort) >= 5 else 0.0
    meta['stats_top10_mean'] = float(np.mean(s_sort[:10])) if len(s_sort) >= 10 else 0.0
    meta['stats_top5_min'] = float(s_sort[4]) if len(s_sort) >= 5 else 0.0
    meta['stats_overall_mean'] = float(np.mean(scores)) if len(scores) > 0 else 0.0
    meta['stats_overall_std'] = float(np.std(scores)) if len(scores) > 0 else 0.0
    meta['ranking_info'] = {'status': ranking_status}
    meta['ranking_source'] = ranking_source
    meta['window_cvli'] = 30

    return meta, results, final_risk, stgcn_score, hist_sum

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
    import traceback
    # Log do estado das variáveis principais
    print("[DEBUG] node_features:", type(node_features), "shape:", getattr(node_features, 'shape', None))
    print("[DEBUG] model_cvli:", type(model_cvli))
    print("[DEBUG] norm_adj_list:", type(norm_adj_list))
    print("[DEBUG] model_ranking_simple:", type(model_ranking_simple))
    print("[DEBUG] scaler:", type(scaler))
    print("[DEBUG] nodes_gdf:", type(nodes_gdf))
    if node_features is None:
        print("[ERROR] node_features is None")
        return jsonify({'error': 'Loading...'}), 503

    try:
        # Use helper to compute baseline predictions
        meta, results, final_risk, stgcn_score, hist_sum = compute_predictions()

        # Apply anomaly-based real-time adjustment (option B)
        try:
            adj_info = {}
            if anomaly_monitor is not None:
                ctx = anomaly_monitor.get_anomaly_context_for_retraining()
                # Prefer today's severity when available
                s = ctx.get('today', {}).get('severity', None) or ctx.get('statistics', {}).get('average_severity', 0.0)
                try:
                    s = float(s)
                except Exception:
                    s = 0.0

                # Small conservative multiplier: up to +25% global risk when severity=1
                adj_factor = 1.0 + (s * 0.25)
                final_risk = np.clip(final_risk * adj_factor, 0.0, 100.0)

                adj_info = {'severity': s, 'adj_factor': adj_factor, 'recommendation': ctx.get('recommendation', {})}

                # Update results' risk_score with adjusted values
                for i in range(len(results)):
                    # Keep per-node safety caps similar to baseline logic
                    score = float(final_risk[i])
                    raw_pred = float(stgcn_score[i])
                    if raw_pred < 0.5:
                        score = min(score, 45.0)
                    elif raw_pred < 1.0:
                        score = min(score, 75.0)
                    if hist_sum[i] == 0 and score > 60:
                        score = 60.0
                    results[i]['risk_score'] = score
                    results[i]['risk_text'] = f"{int(score)}% — {'Crítico' if score >= 90 else 'Alto' if score >= 80 else 'Moderado' if score >= 50 else 'Baixo'}"

            else:
                adj_info = {'severity': 0.0, 'adj_factor': 1.0, 'recommendation': {}}
        except Exception as e:
            print(f"[WARN] anomaly adjustment failed: {e}")
            adj_info = {'severity': 0.0, 'adj_factor': 1.0, 'recommendation': {}}

        # Recompute meta scores after adjustment
        scores = [r['risk_score'] for r in results]
        scores = np.nan_to_num(scores)
        s_sort = sorted(scores, reverse=True)
        meta['stats_top5_mean'] = float(np.mean(s_sort[:5])) if len(s_sort) >= 5 else 0.0
        meta['stats_top10_mean'] = float(np.mean(s_sort[:10])) if len(s_sort) >= 10 else 0.0
        meta['stats_top5_min'] = float(s_sort[4]) if len(s_sort) >= 5 else 0.0
        meta['stats_overall_mean'] = float(np.mean(scores)) if len(scores) > 0 else 0.0
        meta['stats_overall_std'] = float(np.std(scores)) if len(scores) > 0 else 0.0
        meta['anomaly_adjustment'] = adj_info

        if predict_logger: predict_logger.log_prediction(meta, results)
        return jsonify({'meta': meta, 'data': results})

    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] Exception in /api/risk:", e)
        print(tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500


# -----------------------------
# Endpoints para eventos exógenos
# -----------------------------
@app.route('/api/exogenous', methods=['GET', 'POST'])
def handle_exogenous():
    """Lista ou registra um evento exógeno simples (persistido em JSON).
    GET: retorna lista de eventos salvos.
    POST: recebe JSON com evento e anexa em `EXOGENOUS_FILE`.
    """
    try:
        if request.method == 'POST':
            data = request.get_json()
            if data is None:
                return jsonify({'error': 'JSON inválido ou cabecalho Content-Type ausente.'}), 400

            os.makedirs(os.path.dirname(EXOGENOUS_FILE), exist_ok=True)

            existing = []
            if os.path.exists(EXOGENOUS_FILE):
                try:
                    with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                except Exception:
                    existing = []

            existing.append(data)
            with open(EXOGENOUS_FILE, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            return jsonify({"status": "success", "message": "Evento registrado."})

        # GET
        if os.path.exists(EXOGENOUS_FILE):
            try:
                with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                    return jsonify(json.load(f))
            except Exception:
                return jsonify([])
        return jsonify([])
    except Exception as e:
        print(f"[ERROR] /api/exogenous: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exogenous/parse', methods=['POST'])
def parse_exogenous():
    """Parseia texto de ocorrência (CIOPS) e retorna pontos geocodificados.
    Retorna 400 se JSON inválido ou se cidade estiver faltando em ocorrências detectadas.
    """
    try:
        payload = request.get_json()
        if not payload or 'text' not in payload:
            return jsonify({'error': 'JSON inválido ou campo "text" ausente.'}), 400

        text = payload.get('text') or ''
        block_type = payload.get('block_type')

        from src.llm_service import process_exogenous_text

        events = process_exogenous_text(text, block_type=block_type)
        if events is None:
            return jsonify({'error': 'Falha ao parsear o texto.'}), 500

        points = []
        missing_city = []

        # Helper: normalize
        def _norm(s):
            return (s or '').strip().upper()

        for ev in events:
            municipio = (ev.get('municipio') or '').strip()
            bairro = (ev.get('bairro') or '').strip()
            if municipio == '':
                missing_city.append(ev)

            lat = None
            lng = None
            # Try to match against loaded nodes_gdf first
            try:
                if nodes_gdf is not None and bairro:
                    # match by normalized name contains
                    m = nodes_gdf[nodes_gdf['name'].str.upper().str.contains(_norm(bairro))]
                    if len(m) > 0:
                        geom = m.iloc[0].geometry
                        # geometry may be polygon or point
                        try:
                            c = geom.centroid
                            lat = float(c.y)
                            lng = float(c.x)
                        except Exception:
                            lat = float(getattr(geom, 'y', None) or 0)
                            lng = float(getattr(geom, 'x', None) or 0)

            except Exception:
                pass

            # Fallback: try loading bairro centers from data/raw (if exists)
            if (lat is None or lng is None) and os.path.exists(os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')):
                try:
                    from src.data_processing import load_nodes_from_json
                    gdf_nodes, _ = load_nodes_from_json(os.path.join('data', 'raw', 'bairros_centros_latlong.json'))
                    if bairro:
                        mm = gdf_nodes[gdf_nodes['name'].str.upper() == _norm(bairro)]
                        if len(mm) > 0:
                            geom = mm.iloc[0].geometry
                            lat = float(geom.y)
                            lng = float(geom.x)
                except Exception:
                    pass

            # If still not found, but municipio is present, try geocoding by municipio
            if (lat is None or lng is None) and municipio:
                try:
                    if nodes_gdf is not None:
                        # Try to match a city node (node_type == 'cidade') or name equal
                        mm = nodes_gdf[nodes_gdf['name'].str.upper() == _norm(municipio)]
                        if len(mm) == 0 and 'node_type' in nodes_gdf.columns:
                            mm = nodes_gdf[(nodes_gdf['node_type'] == 'cidade') & (nodes_gdf['name'].str.upper().str.contains(_norm(municipio)))]
                        if len(mm) > 0:
                            geom = mm.iloc[0].geometry
                            try:
                                c = geom.centroid
                                lat = float(c.y)
                                lng = float(c.x)
                            except Exception:
                                lat = float(getattr(geom, 'y', None) or 0)
                                lng = float(getattr(geom, 'x', None) or 0)

                except Exception:
                    pass

            if (lat is None or lng is None) and municipio:
                # Try municipality coords file
                mun_path = os.path.join(BASE_DIR, 'data', 'static', 'ceara_municipios_coords.json')
                try:
                    if os.path.exists(mun_path):
                        with open(mun_path, 'r', encoding='utf-8') as fh:
                            jm = json.load(fh)
                            # jm expected structure: { 'MUNICIPIO': {'lat':..., 'long':...}, ... }
                            for k, v in jm.items():
                                if _norm(k) == _norm(municipio) or _norm(municipio) in _norm(k):
                                    lat = float(v.get('lat') or v.get('latitude') or 0)
                                    lng = float(v.get('long') or v.get('longitude') or 0)
                                    break
                except Exception:
                    pass

            if lat is not None and lng is not None:
                points.append({'lat': lat, 'lng': lng, 'bairro': bairro, 'municipio': municipio, 'natureza': ev.get('natureza', '')})

        # If any event lacked municipio, return 400 with details so frontend can show missing cities
        if missing_city:
            return jsonify({'error': 'Falta a cidade na sua ocorrência!', 'missing_city': missing_city}), 400

        return jsonify({'points': points, 'events': events})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] /api/exogenous/parse: {e}\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/api/exogenous/save', methods=['POST'])
def save_exogenous():
    """Persiste eventos exógenos geocodificados (recebe points + original_text)."""
    try:
        payload = request.get_json()
        if not payload or 'points' not in payload:
            return jsonify({'error': 'JSON inválido; espere {points, original_text}'}), 400

        points = payload.get('points', [])
        original = payload.get('original_text', '')

        # Build entry following existing pattern in file
        os.makedirs(os.path.dirname(EXOGENOUS_FILE), exist_ok=True)
        existing = []
        if os.path.exists(EXOGENOUS_FILE):
            try:
                with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        # Determine next id
        next_id = 1
        try:
            ids = [int(item.get('id')) for item in existing if isinstance(item.get('id'), (int, str))]
            if ids:
                next_id = max(ids) + 1
        except Exception:
            next_id = len(existing) + 1

        # If parser provided event details, use them to populate raw_event
        events = payload.get('events', []) or []

        built_points = []
        for idx, p in enumerate(points):
            lat = p.get('lat')
            lng = p.get('lng')
            bairro = p.get('bairro') or ''
            municipio = p.get('municipio') or ''
            natureza = p.get('natureza') or p.get('event_type') or ''

            # Try to find corresponding event by matching fields
            raw_event = None
            for ev in events:
                # match by resumo/natureza/municipio/bairro
                if (ev.get('natureza') and ev.get('natureza') == natureza) or (ev.get('resumo') and ev.get('resumo') == p.get('description')):
                    raw_event = ev
                    break

            if raw_event is None:
                raw_event = {
                    'bairro': bairro,
                    'municipio': municipio,
                    'natureza': natureza,
                    'localizacao_completa': p.get('localizacao_completa', ''),
                    'raw_text': p.get('raw_text', original),
                    'resumo': p.get('resumo', '')
                }

            description = raw_event.get('resumo') or f"{natureza} - {municipio or bairro}"

            built_points.append({
                'description': description,
                'lat': lat,
                'lng': lng,
                'raw_event': raw_event,
                'type': 'exogenous'
            })

        entry = {
            'id': str(next_id),
            'timestamp': datetime.now().isoformat(),
            'original_text': original,
            'points': built_points
        }

        existing.append(entry)
        with open(EXOGENOUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        return jsonify({'status': 'saved', 'count': len(existing), 'id': entry['id']})

    except Exception as e:
        print(f"[ERROR] /api/exogenous/save: {e}")
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


@app.route('/api/explain/<int:node_id>')
def explain_node(node_id: int):
    """Generate explanation for a specific node using current predictions."""
    try:
        if node_features is None:
            return jsonify({'error': 'Loading...'}), 503

        meta, results, final_risk, stgcn_score, hist_sum = compute_predictions()

        if node_id < 0 or node_id >= len(results):
            return jsonify({'error': 'Node not found'}), 404

        # Determine rank
        sorted_nodes = sorted(results, key=lambda x: x['risk_score'], reverse=True)
        ranks = {r['node_id']: idx+1 for idx, r in enumerate(sorted_nodes)}
        rank = ranks.get(node_id, None)

        # Nearby: take top-5 highest risk nodes excluding self
        nearby = [n['node_id'] for n in sorted_nodes if n['node_id'] != node_id][:5]

        # Temporal pattern: simple heuristic from recent mean trend
        recent = node_features[node_id, -14:, 0]
        if np.mean(recent[-7:]) > np.mean(recent[:7]):
            temporal = 'Aumento nas últimas semanas'
        else:
            temporal = 'Estável'

        # Events: include recent exogenous events if available
        evs = []
        if event_manager:
            evs = event_manager.get_recent_events(days_back=7)

        context = {
            'score': results[node_id]['risk_score'],
            'temporal_pattern': temporal,
            'nearby_nodes': nearby,
            'events': evs,
            'confidence': 0.85,
            'tier': 'top_5' if rank and rank <= 5 else 'long_tail_20' if rank and rank <= 20 else 'tail'
        }

        gen = ExplanationGenerator()
        explanation = gen.explain_node_ranking(node_id, rank or 0, context)
        return jsonify({'explanation': explanation})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] /api/explain: {e}\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Simulate applying an exogenous shock to a node and return adjusted top list.
    Payload: { node_id: int, severity: float(0-1) }
    """
    try:
        payload = request.get_json() or {}
        node_id = payload.get('node_id')
        severity = float(payload.get('severity', 0.0))

        if node_id is None:
            return jsonify({'error': 'node_id required'}), 400

        meta, results, final_risk, stgcn_score, hist_sum = compute_predictions()

        if node_id < 0 or node_id >= len(results):
            return jsonify({'error': 'Node not found'}), 404

        # Apply local bump
        sim_scores = final_risk.copy()
        bump = 1.0 + min(max(severity, 0.0), 1.0) * 0.6
        sim_scores[node_id] = float(np.clip(sim_scores[node_id] * bump, 0.0, 100.0))

        # Build top summary before and after
        before = sorted([(r['node_id'], r['risk_score']) for r in results], key=lambda x: x[1], reverse=True)[:20]
        after = sorted([(i, float(sim_scores[i])) for i in range(len(sim_scores))], key=lambda x: x[1], reverse=True)[:20]

        return jsonify({'before_top20': before, 'after_top20': after, 'node_id': node_id, 'severity': severity})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] /api/simulate: {e}\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500

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