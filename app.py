from flask import Flask, jsonify, render_template, request
import pandas as pd
import geopandas as gpd
import pickle
import torch
import numpy as np
import os
import copy
import re
import json
from shapely.geometry import shape, Point, Polygon
from scipy.spatial.distance import cdist
from src.model import STGCN
from src.llm_service import process_exogenous_text
import threading
import time
import unicodedata
from datetime import datetime, timezone

# Desscale mapping (loaded from diagnostics report if present)
_DESSCALE_A = None
_DESSCALE_B = None

def load_desscale_mapping():
    global _DESSCALE_A, _DESSCALE_B
    path = os.path.join(BASE_DIR, 'reports', 'desscale_mapping.txt')
    if os.path.exists(path):
        try:
            a = None
            b = None
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('a='):
                        a = float(line.split('=',1)[1])
                    elif line.startswith('b='):
                        b = float(line.split('=',1)[1])
            if a is not None and b is not None:
                _DESSCALE_A = a
                _DESSCALE_B = b
                print(f'Loaded desscale mapping: a={a}, b={b}')
        except Exception:
            pass

# attempt to load mapping at startup (silently)
try:
    load_desscale_mapping()
except Exception:
    pass

app = Flask(__name__)

# Configuração e Carregamento de Dados (usar caminhos absolutos relativos a este arquivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_CVLI_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_model.pth') # Unified Model Path
# MODEL_CVP_PATH is legacy if we have a unified model, but if user kept 'models/stgcn_model.pth' as the main one:
MODEL_CVP_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvp.pth') # Keep for legacy check or fallback
EXOGENOUS_FILE = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')

# Valores padrão caso arquivos não estejam presentes
nodes_gdf = None
polygons_json_cache = None
nodes_gdf_proj = None
nodes_centroids_proj = None
adj_matrix = None
original_adj_matrix = None
node_features = None
model_cvli = None # This will now hold the MAIN model (which might predict everything or just CVLI)
model_cvp = None
device = None
norm_adj = None
adj_geo = None
adj_faction = None
norm_adj_list = None
dates = None

# Static Data Cache
ibge_bairros_cache = None
ibge_municipios_cache = None
ibge_municipios_gdf = None
GEOCODING_ENABLED = True
exogenous_events = []
exogenous_affected_nodes = set()
exogenous_critical_nodes = set()
# Periodic update status flags (used by frontend polling)
app._periodic_update_in_progress = False
app._periodic_last_update = None

class NodeSearchItem:
    __slots__ = ('name', 'name_lower', 'name_stripped', 'lat', 'lng')
    def __init__(self, name, name_lower, name_stripped, lat, lng):
        self.name = name
        self.name_lower = name_lower
        self.name_stripped = name_stripped
        self.lat = lat
        self.lng = lng

node_search_index = []

MANUAL_LOCATIONS = {
    "TAIBA": (-3.535, -38.892),
    "TAÍBA": (-3.535, -38.892),
}


def strip_accents(text: str) -> str:
    try:
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    except Exception:
        return text


def normalize_city_label(name: str) -> str:
    if not name or not isinstance(name, str):
        return name
    try:
        s = strip_accents(name).strip()
        return s.title()
    except Exception:
        return name


def build_node_search_index():
    global node_search_index
    node_search_index = []
    if nodes_gdf is None:
        return

    def _get_name_from_row(row):
        for col in ('name', 'NAME', 'NOME', 'nome'):
            if col in row and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
        return None

    try:
        for _, row in nodes_gdf.reset_index(drop=True).iterrows():
            name = _get_name_from_row(row)
            if not name:
                continue
            geom = row.get('geometry') if 'geometry' in row else None
            if geom is None:
                continue
            centroid = geom.centroid
            lat = centroid.y
            lng = centroid.x

            name_lower = name.lower()
            name_stripped = strip_accents(name_lower)

            item = NodeSearchItem(name=name, name_lower=name_lower, name_stripped=name_stripped, lat=lat, lng=lng)
            node_search_index.append(item)
    except Exception as e:
        print(f"Erro ao construir node_search_index: {e}")

# Parâmetros de janela
WINDOW_CVLI = 7
WINDOW_CVP = 7

def load_exogenous_events():
    global exogenous_events
    if os.path.exists(EXOGENOUS_FILE):
        try:
            with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                exogenous_events = json.load(f)
            print(f"Carregados {len(exogenous_events)} lotes de eventos exógenos.")
        except Exception as e:
            print(f"Erro ao carregar eventos exógenos: {e}")
            exogenous_events = []
    else:
        print(f"AVISO: Arquivo de eventos exógenos não encontrado: {EXOGENOUS_FILE}")

def find_nearby_nodes(lat, lng, radius_m=500):
    nearby_indices = []
    try:
        p_geo = Point(lng, lat)

        if nodes_gdf_proj is not None:
             s = gpd.GeoSeries([p_geo], crs="EPSG:4326").to_crs("EPSG:3857")
             p_proj = s.iloc[0]

             if nodes_centroids_proj is not None:
                 search_buffer = p_proj.buffer(radius_m)
                 candidate_ilocs = list(nodes_centroids_proj.sindex.intersection(search_buffer.bounds))
                 if candidate_ilocs:
                     candidates = nodes_centroids_proj.iloc[candidate_ilocs]
                     dists = candidates.distance(p_proj)
                     nearby_indices = dists[dists < radius_m].index.tolist()
             else:
                 centroids = nodes_gdf_proj.geometry.centroid
                 dists = centroids.distance(p_proj)
                 nearby_indices = dists[dists < radius_m].index.tolist()
        else:
             if nodes_gdf is not None:
                 centroids = nodes_gdf.geometry.centroid
                 dists = centroids.distance(p_geo)
                 nearby_indices = dists[dists < 0.005].index.tolist()

        if not nearby_indices:
             if nodes_gdf is not None:
                 if nodes_gdf_proj is not None:
                     if nodes_centroids_proj is not None:
                         dists = nodes_centroids_proj.distance(p_proj)
                         nearby_indices = [dists.idxmin()]
                     else:
                         dists = nodes_gdf_proj.geometry.centroid.distance(p_proj)
                         nearby_indices = [dists.idxmin()]
                 else:
                     dists = nodes_gdf.geometry.centroid.distance(p_geo)
                     nearby_indices = [dists.idxmin()]

    except Exception as e:
        print(f"Erro ao buscar nodes próximos: {e}")

    return nearby_indices

def apply_exogenous_events():
    global adj_matrix, exogenous_affected_nodes, adj_geo, adj_faction, exogenous_critical_nodes
    if not exogenous_events or adj_matrix is None:
        return

    print("Aplicando eventos exógenos na malha...")

    exogenous_affected_nodes.clear()
    exogenous_critical_nodes.clear()

    count_affected = 0
    for batch in exogenous_events:
        points = batch.get('points', [])

        for pt in points:
            lat = pt.get('lat') if isinstance(pt, dict) else (pt[0] if isinstance(pt, list) and len(pt)>0 else None)
            lng = pt.get('lng') if isinstance(pt, dict) else (pt[1] if isinstance(pt, list) and len(pt)>1 else None)

            if lat is None or lng is None: continue

            def _is_critical_event(pt_item):
                try:
                    evt = None
                    if isinstance(pt_item, dict):
                        evt = pt_item.get('raw_event') or pt_item
                    if not evt:
                        return False
                    text_fields = []
                    for k in ('natureza','nature','resumo','description'):
                        v = evt.get(k) if isinstance(evt, dict) else None
                        if v:
                            text_fields.append(str(v).lower())
                    txt = ' '.join(text_fields)
                    keywords = ['homic', 'homicídio', 'morte', 'morto', 'tiro', 'lesão a bala', 'lesao a bala', 'lesão', 'lesao', 'ferido', 'assassin']
                    return any(k in txt for k in keywords)
                except Exception:
                    return False

            indices = find_nearby_nodes(lat, lng)

            for idx in indices:
                exogenous_affected_nodes.add(idx)
                critical_flag = _is_critical_event(pt)
                amplification_factor = 10.0 if critical_flag else 5.0
                adj_matrix[idx, :] *= amplification_factor
                adj_matrix[:, idx] *= amplification_factor
                if critical_flag:
                    exogenous_critical_nodes.add(idx)

                try:
                    if adj_geo is not None:
                        adj_geo[idx, :] *= amplification_factor
                        adj_geo[:, idx] *= amplification_factor
                    if adj_faction is not None:
                        adj_faction[idx, :] *= amplification_factor
                        adj_faction[:, idx] *= amplification_factor
                except Exception:
                    pass
                count_affected += 1

    print(f"Malha adaptada: {count_affected} modificações aplicadas.")

def compute_norm_adj(adj_matrix_input):
    if adj_matrix_input is None: return None
    adj_tensor = torch.FloatTensor(adj_matrix_input)
    rowsum = adj_tensor.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)

def compute_norm_adj_list(adj_list):
    if adj_list is None:
        return None
    res = []
    for a in adj_list:
        res.append(compute_norm_adj(a))
    return res

def update_exogenous_state():
    global adj_matrix, norm_adj, original_adj_matrix
    global adj_geo, adj_faction, norm_adj_list

    if original_adj_matrix is None:
        print("AVISO: Matriz original não disponível para atualização incremental. Recarregando tudo.")
        load_data_and_models()
        return

    print("Atualizando estado exógeno incrementalmente...")

    adj_matrix = original_adj_matrix.copy()
    if adj_geo is not None:
        try:
            adj_geo = original_adj_matrix.copy()
        except Exception:
            pass
    if adj_faction is not None:
        try:
            adj_faction = original_adj_matrix.copy()
        except Exception:
            pass

    apply_exogenous_events()

    if adj_geo is not None and adj_faction is not None:
        norm_adj_list = compute_norm_adj_list([adj_geo, adj_faction])
        norm_adj = norm_adj_list[0] if norm_adj_list else None
    else:
        norm_adj = compute_norm_adj(adj_matrix)

def load_data_and_models():
    global nodes_gdf, polygons_json_cache, nodes_gdf_proj, nodes_centroids_proj, adj_matrix, node_features, model_cvli, model_cvp, device, norm_adj, dates, original_adj_matrix
    global adj_geo, adj_faction, norm_adj_list
    global ibge_bairros_cache, ibge_municipios_cache, ibge_municipios_gdf

    # Load Static Data
    try:
        static_file = os.path.join(BASE_DIR, 'data', 'static', 'fortaleza_bairros_coords.json')
        if os.path.exists(static_file):
            with open(static_file, 'r', encoding='utf-8') as f:
                ibge_bairros_cache = json.load(f)

        static_file_mun = os.path.join(BASE_DIR, 'data', 'static', 'ceara_municipios_coords.json')
        if os.path.exists(static_file_mun):
            with open(static_file_mun, 'r', encoding='utf-8') as f:
                ibge_municipios_cache = json.load(f)
        try:
            possible_paths = [
                os.path.join(BASE_DIR, 'data', 'static', 'ceara_municipios.geojson'),
                os.path.join(BASE_DIR, 'data', 'static', 'ceara_municipios.json'),
                os.path.join(BASE_DIR, 'data', 'static', 'municipios_ceara.geojson'),
                os.path.join(BASE_DIR, 'data', 'static', 'municipios_ceara.json')
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    try:
                        ibge_municipios_gdf = gpd.read_file(p)
                        if ibge_municipios_gdf.crs is None:
                            ibge_municipios_gdf.set_crs(epsg=4326, inplace=True)
                        print(f"Loaded municipality polygons from {p}")
                        break
                    except Exception as e:
                        print(f"Failed to read municipalities polygons {p}: {e}")
        except Exception:
            pass
    except Exception as e:
        print(f"Erro ao carregar dados estáticos: {e}")

    data_pack = None
    if not os.path.exists(DATA_FILE):
        print("AVISO: processed_graph_data.pkl não encontrado!")
        return

    print("Carregando dados para API...")
    try:
        if data_pack is None:
            with open(DATA_FILE, 'rb') as f:
                data_pack = pickle.load(f)

        # Use temporary local variables to allow validation before assignment
        _nodes_gdf = data_pack.get('nodes_gdf')
        polygons_json_cache = None
        _adj_geo = data_pack.get('adj_geo')
        _adj_faction = data_pack.get('adj_faction')
        _adj_matrix = data_pack.get('adj_matrix')
        if _adj_matrix is None:
            _adj_matrix = _adj_geo
        _node_features = data_pack.get('node_features')
        _dates = data_pack.get('dates')

        # --- VALIDATION: Check Data Integrity (Paradigm Shift) ---
        if _node_features is not None:
            # Check Node Count (Expected ~319 for Admin boundaries, NOT ~2378 for Grid)
            if _node_features.shape[0] > 1000:
                err_msg = (
                    f"CRITICAL: Dados obsoletos detectados! (Nós: {_node_features.shape[0]}). "
                    "O sistema agora usa limites administrativos (~319 nós). "
                    "Por favor, execute 'python src/data_processing.py' para regenerar os dados."
                )
                print(err_msg)
                raise RuntimeError(err_msg)

            # Check Channels (Expected 3: CVLI, CVP, Tension)
            if len(_node_features.shape) < 3 or _node_features.shape[2] != 3:
                channels = _node_features.shape[2] if len(_node_features.shape) > 2 else 1
                err_msg = (
                    f"CRITICAL: Dados obsoletos detectados! (Canais: {channels}). "
                    "O sistema requer 3 canais (CVLI, CVP, Tensão). "
                    "Por favor, execute 'python src/data_processing.py' para regenerar os dados."
                )
                print(err_msg)
                raise RuntimeError(err_msg)

        # Validation Passed - Assign to Globals
        nodes_gdf = _nodes_gdf
        adj_geo = _adj_geo
        adj_faction = _adj_faction
        adj_matrix = _adj_matrix
        node_features = _node_features
        dates = _dates

        # --- FIXED: Node Paradigm Loading ---
        if nodes_gdf is not None:
             # Ensure CIDADE is populated correctly based on node_type
             if 'node_type' in nodes_gdf.columns:
                 # Logic: If node_type='cidade', CIDADE = name. If 'bairro', CIDADE = Fortaleza (or default logic)
                 # Use vectorized numpy where for speed
                 nodes_gdf['CIDADE'] = np.where(
                     nodes_gdf['node_type'] == 'cidade',
                     nodes_gdf['name'],
                     'Fortaleza' # Default for Capital/RMF bairros for now, can be refined
                 )
                 print("Applied Node Paradigm logic: Cities assigned to CIDADE column.")

        if nodes_gdf is not None:
            try:
                nodes_gdf_proj = nodes_gdf.to_crs(epsg=3857)
                nodes_centroids_proj = nodes_gdf_proj.geometry.centroid
                _ = nodes_centroids_proj.sindex
            except Exception as e:
                print(f"Erro ao projetar nodes_gdf: {e}")

            build_node_search_index()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = node_features.shape[0]

        if adj_matrix is not None:
            original_adj_matrix = adj_matrix.copy()

        load_exogenous_events()
        apply_exogenous_events()

        if adj_geo is not None and adj_faction is not None:
            try:
                norm_adj_list = compute_norm_adj_list([adj_geo, adj_faction])
                norm_adj = norm_adj_list[0] if norm_adj_list else None
            except Exception as e:
                print(f"Erro ao normalizar adjacências: {e}")
                norm_adj = compute_norm_adj(adj_matrix)
        else:
            norm_adj = compute_norm_adj(adj_matrix)

        # --- FIXED: 3-Channel Model Loading ---
        def _adapt_state_dict_for_multigraph(state_dict, num_graphs):
            try:
                sd = dict(state_dict)
            except Exception:
                sd = state_dict
            keys = list(sd.keys())
            for k in keys:
                if k.endswith('.gcn.weight'):
                    base = k[:-len('.gcn.weight')]
                    w = sd[k]
                    for i in range(num_graphs):
                        newk = f"{base}.gcn.weights.{i}"
                        sd[newk] = w
                    del sd[k]
            return sd

        if os.path.exists(MODEL_CVLI_PATH):
            print(f"Carregando modelo UNIFICADO de {MODEL_CVLI_PATH}...")
            num_graphs = len(norm_adj_list) if norm_adj_list is not None else 1
            try:
                raw_state = torch.load(MODEL_CVLI_PATH, map_location=device)
                state_dict = {}
                for k,v in dict(raw_state).items():
                    nk = k[7:] if k.startswith('module.') else k
                    state_dict[nk] = v

                state_dict = _adapt_state_dict_for_multigraph(state_dict, num_graphs)

                ck_time_steps = None
                if 'conv_final.weight' in state_dict:
                    try:
                        ck_time_steps = state_dict['conv_final.weight'].shape[-1]
                    except Exception:
                        ck_time_steps = None

                instantiate_ts = WINDOW_CVLI if ck_time_steps is None else ck_time_steps

                # FIXED: in_channels=3
                m_cvli = STGCN(num_nodes=num_nodes, in_channels=3, time_steps=instantiate_ts, num_classes=1, num_graphs=num_graphs)
                m_cvli.load_state_dict(state_dict, strict=False)
                m_cvli.to(device)
                m_cvli.eval()
                model_cvli = m_cvli
            except Exception as e:
                print(f"Erro ao carregar state_dict CVLI: {e}")
        else:
            print(f"AVISO: Modelo CVLI não encontrado em {MODEL_CVLI_PATH}")

    except Exception as e:
        print(f"Erro ao carregar dados/modelos: {e}")
        # Only print stack trace if it's NOT the expected validation error
        if "CRITICAL" not in str(e):
            import traceback
            traceback.print_exc()


def _periodic_reload_loop(interval_minutes: int):
    interval = max(1, int(interval_minutes)) * 60
    while True:
        try:
            print("[PeriodicReload] Scheduled reload starting...")
            try:
                app._periodic_update_in_progress = True
            except Exception:
                pass

            load_data_and_models()

            try:
                app._periodic_last_update = datetime.now(timezone.utc).isoformat()
            except Exception:
                pass

            try:
                app._periodic_update_in_progress = False
            except Exception:
                pass

            print("[PeriodicReload] Scheduled reload finished.")
        except Exception as e:
            print(f"[PeriodicReload] Error during scheduled reload: {e}")
        time.sleep(interval)


def start_periodic_reload(interval_minutes: int = 30):
    if getattr(app, '_periodic_reload_started', False):
        return
    t = threading.Thread(target=_periodic_reload_loop, args=(interval_minutes,), daemon=True)
    t.start()
    app._periodic_reload_started = True


@app.route('/api/periodic_status')
def periodic_status():
    try:
        return jsonify({
            'in_progress': bool(getattr(app, '_periodic_update_in_progress', False)),
            'last_update': getattr(app, '_periodic_last_update', None)
        })
    except Exception:
        return jsonify({'in_progress': False, 'last_update': None})

start_periodic_reload(30)

def format_trend(prediction, history_avg, risk_score=None):
    if history_avg == 0:
        if prediction > 0.001:
            return "Nova atividade criminal detectada"
        else:
            return "Situação estável (Baixo Risco)"

    change_pct = ((prediction - history_avg) / history_avg) * 100

    if change_pct > 15:
        return "Aumento recente da criminalidade"
    elif change_pct < -15:
        return "Redução da atividade criminal"
    else:
        if risk_score is not None:
            if risk_score > 60:
                return "Valor histórico alto para o período"
            elif risk_score > 20:
                return "Atividade criminal moderada"

        return "Situação estável (Baixo Risco)"

def get_region_name(feature_props, geometry):
    city = feature_props.get('CIDADE', '')
    if city and isinstance(city, str) and len(city.strip()) > 1:
        return city
    try:
        from shapely.geometry import Point
    except Exception:
        Point = None

    if geometry is not None:
        try:
            centroid = shape(geometry).centroid
            lon, lat = centroid.x, centroid.y
        except Exception:
            lon = lat = None

        if lon is not None and lat is not None and Point is not None:
            pt = Point(lon, lat)

            if ibge_municipios_gdf is not None and getattr(ibge_municipios_gdf, 'geometry', None) is not None:
                possible_idx = []
                try:
                    possible_idx = list(ibge_municipios_gdf.sindex.intersection(pt.bounds))
                except Exception:
                    possible_idx = []

                if possible_idx:
                    candidates = ibge_municipios_gdf.iloc[possible_idx]
                    for _, mun_row in candidates.iterrows():
                        poly = mun_row.geometry if 'geometry' in mun_row else None
                        if poly is not None and poly.contains(pt):
                            for col in ('name', 'NAME', 'NOME', 'nome', 'municipio', 'NM_MUNICIP'):
                                if col in mun_row and isinstance(mun_row[col], str) and mun_row[col].strip():
                                    return mun_row[col].strip()
                            return str(mun_row.name)

            if lon is not None and lat is not None:
                if -38.66 <= lon <= -38.40 and -3.90 <= lat <= -3.65:
                    return "Fortaleza"

    return "RMF/Interior"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/polygons')
def get_polygons():
    global polygons_json_cache
    if nodes_gdf is None:
        return jsonify({'error': 'Dados de polígonos não carregados.'}), 503
    try:
        if polygons_json_cache is not None:
            return polygons_json_cache

        polygons_json_cache = nodes_gdf.to_json()
        return polygons_json_cache
    except Exception as e:
        return jsonify({'error': f'Erro ao serializar polígonos: {e}'}), 500

@app.route('/api/risk')
def get_risk():
    return calculate_risk()

@app.route('/connections')
def connections_view():
    return render_template('connections.html')

@app.route('/api/network-graph')
def get_network_graph():
    if adj_matrix is None:
        return jsonify({'error': 'Matriz de adjacência não carregada.'}), 503

    response = calculate_risk()
    if response.status_code != 200:
        return response

    risk_data = response.get_json()
    all_nodes = risk_data.get('data', [])

    sorted_nodes = sorted(all_nodes, key=lambda x: x['risk_score'], reverse=True)
    top_30 = sorted_nodes[:30]

    graph_nodes = []
    top_indices = set()

    for item in top_30:
        nid = item['node_id']
        top_indices.add(nid)

        name = f"Area {nid}"
        if nodes_gdf is not None and nid < len(nodes_gdf):
            name = nodes_gdf.iloc[nid].get('name') or nodes_gdf.iloc[nid].get('nome') or name

        graph_nodes.append({
            'id': nid,
            'name': name,
            'risk': item['risk_score'],
            'faction': item.get('faction'),
            'reasons': item.get('reasons', [])
        })

    links = []

    for source in top_30:
        u = source['node_id']
        for target in top_30:
            v = target['node_id']
            if u == v: continue

            try:
                weight = float(adj_matrix[u, v])
                if weight > 0:
                    links.append({
                        'source': u,
                        'target': v,
                        'weight': weight,
                        'type': 'influence'
                    })
            except Exception:
                pass

    return jsonify({
        'nodes': graph_nodes,
        'links': links
    })

@app.route('/api/simulate', methods=['POST'])
def simulate_risk():
    if node_features is None or adj_matrix is None or nodes_gdf is None:
        return jsonify({'error': 'Dados não carregados.'}), 503

    try:
        data = request.get_json()
        points = data.get('points', [])
        sim_type = data.get('type', 'suppression')

        adj_copy = adj_matrix.copy()
        affected_nodes = set()
        centroids = None
        if nodes_gdf_proj is not None:
             if nodes_centroids_proj is None:
                 centroids = nodes_gdf_proj.geometry.centroid
        else:
             centroids = nodes_gdf.geometry.centroid

        for pt in points:
            if len(pt) == 2:
                p_geo = Point(pt[1], pt[0])
                nearby_indices = []

                if nodes_gdf_proj is not None:
                    try:
                        s = gpd.GeoSeries([p_geo], crs="EPSG:4326").to_crs("EPSG:3857")
                        p_proj = s.iloc[0]

                        if nodes_centroids_proj is not None:
                             search_buffer = p_proj.buffer(500)
                             candidate_ilocs = list(nodes_centroids_proj.sindex.intersection(search_buffer.bounds))
                             if candidate_ilocs:
                                 candidates = nodes_centroids_proj.iloc[candidate_ilocs]
                                 dists = candidates.distance(p_proj)
                                 nearby_indices = dists[dists < 500].index.tolist()
                        else:
                             dists = centroids.distance(p_proj)
                             nearby_indices = dists[dists < 500].index.tolist()
                    except Exception as e:
                        print(f"Erro ao projetar ponto na simulação: {e}")
                        dists = nodes_gdf.geometry.centroid.distance(p_geo)
                        nearby_indices = [dists.idxmin()]
                else:
                    dists = centroids.distance(p_geo)
                    nearby_indices = dists[dists < 0.005].index.tolist()
                    if not nearby_indices:
                        nearby_indices = [dists.idxmin()]

                for idx in nearby_indices:
                    affected_nodes.add(idx)
                    if sim_type == 'suppression':
                        suppression_factor = 0.05
                        adj_copy[idx, :] *= suppression_factor
                        adj_copy[:, idx] *= suppression_factor
                    elif sim_type == 'exogenous':
                        amplification_factor = 5.0
                        adj_copy[idx, :] *= amplification_factor
                        adj_copy[:, idx] *= amplification_factor

        if adj_geo is not None and adj_faction is not None:
            adj_geo_copy = adj_geo.copy()
            adj_faction_copy = adj_faction.copy()
            sim_norm_list = compute_norm_adj_list([adj_geo_copy, adj_faction_copy])
            response = calculate_risk(custom_norm_adj=sim_norm_list)
        else:
            adj_tensor = torch.FloatTensor(adj_copy)
            rowsum = adj_tensor.sum(1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            sim_norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)
            response = calculate_risk(custom_norm_adj=sim_norm_adj)

        if response.status_code == 200:
            result_json = response.get_json()
            data_list = result_json.get('data', [])

            for item in data_list:
                nid = item['node_id']
                if nid in affected_nodes:
                    if sim_type == 'suppression':
                        factor = 0.15
                        item['risk_score'] = item['risk_score'] * factor
                        item['risk_score_cvli'] = item['risk_score_cvli'] * factor
                        item['risk_score_cvp'] = item['risk_score_cvp'] * factor
                        item['reasons'].insert(0, "Área sob controle (Equipe Tática)")
                        item['cvli_pred'] *= factor
                        item['cvp_pred'] *= factor

                    elif sim_type == 'exogenous':
                        boost = 1.5
                        new_score = max(item['risk_score'] * boost, 80.0)
                        item['risk_score'] = min(new_score, 100.0)
                        item['risk_score_cvli'] = min(max(item['risk_score_cvli'] * boost, 80.0), 100.0)
                        item['risk_score_cvp'] = min(max(item['risk_score_cvp'] * boost, 80.0), 100.0)
                        item['reasons'].insert(0, "Conflito Ativo (Simulação)")

            return jsonify(result_json)
        else:
            return response

    except Exception as e:
        print(f"Erro na simulação: {e}")
        return jsonify({'error': f'Erro na simulação: {e}'}), 500

def calculate_risk(custom_norm_adj=None):
    global dates
    if node_features is None or nodes_gdf is None:
        return jsonify({'error': 'Dados não carregados.'}), 503

    current_norm_adj = custom_norm_adj if custom_norm_adj is not None else norm_adj
    def _ensure_adj_list(adj):
        if adj is None:
            return None
        if isinstance(adj, (list, tuple)):
            return list(adj)
        if norm_adj_list is not None:
            return norm_adj_list
        return [adj]

    adj_for_model = _ensure_adj_list(current_norm_adj)
    if adj_for_model is None:
         return jsonify({'error': 'Matriz de adjacência não disponível.'}), 503

    try:
        # ---------------------------
        # Previsão UNIFICADA (Multimodal)
        # ---------------------------
        out_cvli = np.zeros((node_features.shape[0], 1))
        hist_avg_cvli = np.zeros(node_features.shape[0])
        hist_sum_cvli = np.zeros(node_features.shape[0])

        # We also need CVP history/output to show in UI
        hist_avg_cvp = np.zeros(node_features.shape[0])
        hist_sum_cvp = np.zeros(node_features.shape[0])

        if model_cvli:
            try:
                model_ts = model_cvli.conv_final.kernel_size[-1]
            except Exception:
                model_ts = WINDOW_CVLI

            if node_features.shape[1] >= model_ts:
                # --- FIXED: Slice all 3 channels ---
                input_slice = node_features[:, -model_ts:, :] # (N, T, 3)

                # Check for channel mismatch (Runtime Safety)
                if input_slice.shape[2] != 3:
                    # Pad or trim if desperate (should be caught by startup check, but for robustness)
                    if input_slice.shape[2] == 2:
                        # Append zero tension channel
                        zeros = np.zeros((input_slice.shape[0], input_slice.shape[1], 1), dtype=input_slice.dtype)
                        input_slice = np.concatenate([input_slice, zeros], axis=2)
                        print("AVISO: Adicionado canal de Tensão (zeros) dinamicamente para prevenir crash.")

                input_tensor = torch.FloatTensor(input_slice).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, N, T)

                with torch.no_grad():
                    pred = model_cvli(input_tensor, adj_for_model)
                out_cvli = pred.squeeze(0).cpu().numpy() # (N, 1)

                try:
                    if _DESSCALE_A is not None:
                        out_cvli = (_DESSCALE_A * out_cvli) + _DESSCALE_B
                except Exception:
                    pass

                # Channel 0 is CVLI
                input_cvli = input_slice[:, :, 0]
                daily_avg = np.mean(input_cvli, axis=1)
                hist_avg_cvli = daily_avg * 3
                hist_sum_cvli = np.sum(input_cvli, axis=1)

                # Channel 1 is CVP
                input_cvp = input_slice[:, :, 1]
                daily_avg_cvp = np.mean(input_cvp, axis=1)
                hist_avg_cvp = daily_avg_cvp
                hist_sum_cvp = np.sum(input_cvp, axis=1)

            else:
                print("AVISO: Dados insuficientes para janela temporal")

        # ---------------------------
        # Processamento e Normalização (Híbrida)
        # ---------------------------

        # CVLI
        out_cvli = np.maximum(out_cvli, 0)
        cvli_raw = out_cvli[:, 0]
        cvli_adj = cvli_raw * 1.5

        min_cvli = np.min(cvli_adj)
        shifted_cvli = cvli_adj - min_cvli
        max_shift_cvli = np.max(shifted_cvli) if np.max(shifted_cvli) > 0 else 1
        normalized_risk_cvli = (shifted_cvli / max_shift_cvli) * 100

        active_indices = hist_sum_cvli > 0
        normalized_risk_cvli[active_indices] = np.maximum(normalized_risk_cvli[active_indices], 25.0)

        very_active = hist_sum_cvli >= 3
        normalized_risk_cvli[very_active] = np.maximum(normalized_risk_cvli[very_active], 50.0)

        if exogenous_affected_nodes:
            exo_indices = list(exogenous_affected_nodes)
            exo_indices = [i for i in exo_indices if i < len(normalized_risk_cvli)]
            if exo_indices:
                normalized_risk_cvli[exo_indices] = np.maximum(normalized_risk_cvli[exo_indices], 80.0)

            try:
                if exogenous_critical_nodes:
                    crit_idxs = [i for i in exogenous_critical_nodes if i < len(normalized_risk_cvli)]
                    if crit_idxs:
                        normalized_risk_cvli[crit_idxs] = np.maximum(normalized_risk_cvli[crit_idxs], 95.0)
                        top_k = 10
                        try:
                            if adj_faction is not None:
                                for ci in crit_idxs:
                                    neigh = np.where(np.array(adj_faction[ci]) > 0)[0]
                                    neigh = [n for n in neigh if n != ci]
                                    if not neigh: continue
                                    selected = neigh[:top_k]
                                    if selected:
                                        normalized_risk_cvli[selected] = np.maximum(normalized_risk_cvli[selected], 90.0)
                        except Exception:
                            pass
            except Exception:
                pass

        # Use same risk for CVP UI indication for now, or just normalized history?
        # Since we only predict CVLI now, we don't have a separate CVP prediction model in this loop unless we kept it?
        # The user said "unified". So out_cvli IS the risk.
        # But for UI 'risk_score_cvp', we can use the history as proxy or just copy main risk.
        # Let's derive CVP Risk from CVP history + Context for now to populate the UI field.
        # Or even better, assume CVP Risk correlates with Main Risk (since it's a precursor).
        normalized_risk_cvp = normalized_risk_cvli.copy()
        # But CVP is raw count, not risk.
        # Let's just normalize CVP History for visualization?
        # Actually, let's leave it as copy of main risk, but potentially modified by local CVP intensity.

        # ---------------------------
        # Construção da Resposta
        # ---------------------------
        results = []

        try:
            conn = np.array(adj_matrix).sum(axis=1)
            conn_mean = float(np.mean(conn))
        except Exception:
            conn = None
            conn_mean = 0

        try:
            cutoff_cvli = float(np.percentile(normalized_risk_cvli, 90))
        except Exception:
            cutoff_cvli = 80.0

        factions = nodes_gdf['faction'].tolist() if 'faction' in nodes_gdf.columns else [None] * len(nodes_gdf)

        for i in range(len(normalized_risk_cvli)):
            cvli_score = float(normalized_risk_cvli[i])
            cvp_score = float(normalized_risk_cvp[i])

            cvli_val = float(cvli_adj[i])
            # For CVP 'prediction' in UI, use history avg if we don't have explicit pred
            cvp_val = float(hist_avg_cvp[i])

            trend_cvli = format_trend(cvli_raw[i], hist_avg_cvli[i], risk_score=cvli_score)
            trend_cvp = format_trend(cvp_val, hist_avg_cvp[i], risk_score=cvp_score)

            reasons = []

            if cvli_val > 0.01 or cvli_score > 20:
                reasons.append(f'CVLI: {trend_cvli}')

            if hist_sum_cvli[i] > 0 and cvli_val < 0.01:
                reasons.append('Histórico recente de violência')

            if i in exogenous_affected_nodes:
                reasons.insert(0, "Conflito Ativo")

            if len(reasons) == 0:
                reasons.append('Situação estável (Baixo Risco)')

            if conn is not None and conn_mean > 0 and conn[i] > conn_mean * 1.5:
                reasons.append('Alta conectividade (Rota de fuga/acesso)')

            def _status_label(score: float) -> str:
                if score >= 90:
                    return 'Crítico'
                if score >= 70:
                    return 'Alto'
                if score >= 40:
                    return 'Médio'
                return 'Baixo'

            def _prediction_text(val: float, kind: str='CVLI') -> str:
                horizon_days = 7
                try:
                    if val <= 0.01:
                        return f'Sem novas ocorrências previstas (próx. {horizon_days} dias)'
                    if val < 1.0:
                        return f'Menos de 1 ocorrência prevista (próx. {horizon_days} dias)'
                    rounded = round(val, 1)
                    unit = 'ocorrência' if rounded == 1 else 'ocorrências'
                    return f'Estimativa: ~{rounded} {unit} previstas (próx. {horizon_days} dias)'
                except Exception:
                    return 'Estimativa indisponível'

            status_label = _status_label(cvli_score)
            risk_text = f"{int(round(cvli_score))}% — {status_label}"
            cvli_pred_text = _prediction_text(cvli_val, 'CVLI')
            cvp_pred_text = _prediction_text(cvp_val, 'CVP')

            results.append({
                'node_id': int(i),
                'risk_score': cvli_score,
                'risk_score_cvli': cvli_score,
                'risk_score_cvp': cvp_score,
                'cvli_pred': cvli_val,
                'cvp_pred': cvp_val,
                'faction': factions[i],
                'reasons': reasons,
                'priority_cvli': bool(cvli_score >= cutoff_cvli),
                'status_label': status_label,
                'risk_text': risk_text,
                'cvli_prediction_text': cvli_pred_text,
                'cvp_prediction_text': cvp_pred_text
            })

        try:
            if (dates is None) or (hasattr(dates, '__len__') and len(dates) == 0):
                graph_dir = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
                dpkl = os.path.join(graph_dir, 'dates.pkl')
                if os.path.exists(dpkl):
                    with open(dpkl, 'rb') as fh:
                        dates = pickle.load(fh)
        except Exception:
            pass

        meta = {
            'window_cvli': WINDOW_CVLI,
            'window_cvp': WINDOW_CVP,
            'start_cvli': '—',
            'start_cvp': '—',
            'last_date': '—',
            'window_start': '—',
            'window_end': '—'
        }

        if dates is not None and hasattr(dates, '__len__') and len(dates) > 0:
            try:
                last_date_obj = pd.to_datetime(dates[-1])
                meta['last_date'] = str(last_date_obj.date())
                meta['start_cvli'] = str(pd.to_datetime(dates[-min(len(dates), WINDOW_CVLI)]).date())
                meta['start_cvp'] = meta['start_cvli']
                meta['window_start'] = meta.get('start_cvli', '')
                meta['window_end'] = meta.get('last_date', '')
            except Exception:
                pass

        meta['model_window_cvli'] = WINDOW_CVLI
        meta['model_window_cvp'] = WINDOW_CVP

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        print(f"ERROR details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro ao calcular risco: {e}'}), 500

def enrich_regions():
    global nodes_gdf, polygons_json_cache
    if nodes_gdf is None: return

    if 'CIDADE' not in nodes_gdf.columns:
        nodes_gdf['CIDADE'] = ''

    # Ensure region_type is available for frontend filters
    if 'regiao' in nodes_gdf.columns:
        # Normalize 'fortaleza' to 'capital' for frontend consistency
        nodes_gdf['region_type'] = nodes_gdf['regiao'].replace('fortaleza', 'capital')
    else:
        nodes_gdf['region_type'] = 'interior' # Fallback default

    print("Enriching regions...")

    # --- FIXED: Use node_type to enforce City names ---
    if 'node_type' in nodes_gdf.columns:
        # If node is city, its name IS the city.
        nodes_gdf.loc[nodes_gdf['node_type'] == 'cidade', 'CIDADE'] = nodes_gdf.loc[nodes_gdf['node_type'] == 'cidade', 'name']
        # If node is bairro, default to Fortaleza if empty
        nodes_gdf.loc[(nodes_gdf['node_type'] == 'bairro') & (nodes_gdf['CIDADE'] == ''), 'CIDADE'] = 'Fortaleza'

    # Continue with fallback logic for anything else
    mask_empty = (nodes_gdf['CIDADE'].isna()) | (nodes_gdf['CIDADE'].astype(str).str.len() < 2)

    if mask_empty.any():
        indices_to_update = nodes_gdf[mask_empty].index
        count_fortaleza = 0
        for idx in indices_to_update:
            row = nodes_gdf.loc[idx]
            reg = get_region_name(row, row.geometry)
            if reg == "Fortaleza":
                nodes_gdf.at[idx, 'CIDADE'] = reg
                count_fortaleza += 1
        print(f"Assigned Fortaleza to {count_fortaleza} nodes via BBox.")

    mask_still_empty = (nodes_gdf['CIDADE'].isna()) | (nodes_gdf['CIDADE'].astype(str).str.len() < 2) | (nodes_gdf['CIDADE'] == 'RMF/Interior')

    if mask_still_empty.any() and ibge_municipios_cache:
        try:
            nodes_sub = nodes_gdf[mask_still_empty]
            mun_names = list(ibge_municipios_cache.keys())
            mun_coords = list(ibge_municipios_cache.values())
            mun_coords_arr = np.array(mun_coords)

            centroids_proj = nodes_sub.to_crs(epsg=3857).geometry.centroid
            centroids = centroids_proj.to_crs(nodes_sub.crs)
            node_coords = np.column_stack((centroids.y.values, centroids.x.values))

            dists = cdist(node_coords, mun_coords_arr)
            nearest_indices = np.argmin(dists, axis=1)

            assigned_cities = [mun_names[i] for i in nearest_indices]
            nodes_gdf.loc[mask_still_empty, 'CIDADE'] = assigned_cities

            print(f"Inferred cities for {len(assigned_cities)} nodes using proximity.")

        except Exception as e:
            print(f"Erro na inferência por distância: {e}")

    polygons_json_cache = None

def normalize_location(text):
    if not text: return ""
    text = text.upper()
    text = text.replace("PQ.", "PARQUE")
    text = text.replace("AV.", "AVENIDA")
    text = text.replace("S/", "SEM ")
    return text.strip()

def find_node_coordinates(location_str):
    if nodes_gdf is None:
        return None

    loc_norm = normalize_location(location_str)
    loc_lower = loc_norm.lower()
    loc_stripped = strip_accents(loc_lower)

    if loc_norm in MANUAL_LOCATIONS:
         return (*MANUAL_LOCATIONS[loc_norm], 'manual')

    best_match_item = None
    best_match_len = 0

    for item in node_search_index:
        name = item.name_lower
        name_stripped = item.name_stripped

        if name in loc_lower:
            if len(name) > best_match_len:
                best_match_item = item
                best_match_len = len(name)
        elif name_stripped in loc_stripped:
             if len(name) > best_match_len:
                best_match_item = item
                best_match_len = len(name)
        elif loc_lower in name and len(loc_lower) > 4:
             if len(loc_lower) > best_match_len:
                best_match_item = item
                best_match_len = len(loc_lower)
        elif loc_stripped in name_stripped and len(loc_stripped) > 4:
             if len(loc_stripped) > best_match_len:
                best_match_item = item
                best_match_len = len(loc_stripped)

    if best_match_item is not None:
        return (best_match_item.lat, best_match_item.lng, 'specific')

    if ibge_bairros_cache:
        try:
            for bairro, coords in ibge_bairros_cache.items():
                b_norm = normalize_location(bairro).lower()
                b_stripped = strip_accents(b_norm)
                if b_norm in loc_lower or b_stripped in loc_stripped:
                    return (coords[0], coords[1], 'specific')
        except Exception as e:
             pass

    if ibge_municipios_cache:
        try:
            for municipio, coords in ibge_municipios_cache.items():
                m_norm = normalize_location(municipio).lower()
                if m_norm in loc_lower:
                    return (coords[0], coords[1], 'city')
        except Exception as e:
             pass

    if ibge_municipios_cache:
        try:
             import re
             words = re.findall(r'\b\w+\b', loc_lower)
             for word in words:
                 if len(word) < 4: continue
                 for mun, coords in ibge_municipios_cache.items():
                     if mun.lower() == word:
                         return (coords[0], coords[1], 'city')
        except Exception as e:
            pass

    if GEOCODING_ENABLED:
        try:
            geo_res = geocode_address(loc_norm)
            if geo_res:
                return (*geo_res, 'geocode')
        except Exception:
            pass

    return None


def geocode_address(location_str, timeout=10):
    try:
        from geopy.geocoders import Nominatim
    except Exception:
        return None

    try:
        geolocator = Nominatim(user_agent="st-gcn-geocoder", timeout=timeout)
        loc = geolocator.geocode(location_str)
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        return None
    return None

@app.route('/api/exogenous/parse', methods=['POST'])
def parse_exogenous():
    data = request.get_json()
    text = data.get('text', '')

    events = process_exogenous_text(text)
    missing_city = []
    for idx, evt in enumerate(events):
        muni = evt.get('municipio') if isinstance(evt, dict) else None
        if not muni or (isinstance(muni, str) and muni.strip() == ''):
            missing_city.append({'index': idx, 'natureza': evt.get('natureza'), 'resumo': evt.get('resumo'), 'localizacao_completa': evt.get('localizacao_completa')})

    if missing_city:
        return jsonify({
            'error': 'Falta a cidade na sua ocorrência!',
            'message': 'Por favor preencha o município nas ocorrências indicadas antes de prosseguir.',
            'missing_city': missing_city
        }), 400
    points = []

    for evt in events:
        found_lat = None
        found_lng = None
        match_quality = 0

        if evt.get('localizacao_completa'):
            res = find_node_coordinates(evt['localizacao_completa'])
            if res:
                found_lat, found_lng, mtype = res
                match_quality = 3 if mtype == 'specific' else 2

        if (not found_lat or match_quality < 2) and evt.get('bairro'):
            res = find_node_coordinates(evt['bairro'])
            if res:
                lat_b, lng_b, mtype_b = res
                if mtype_b == 'specific':
                     found_lat, found_lng = lat_b, lng_b
                     match_quality = 3

        if not found_lat and evt.get('municipio'):
            res = find_node_coordinates(evt['municipio'])
            if res:
                found_lat, found_lng, _ = res
                match_quality = 1

        if found_lat is not None and found_lng is not None:
            import html
            desc = f"{evt.get('natureza', 'EVENTO')} - {evt.get('resumo', '')}"
            if not evt.get('resumo'):
                desc = f"{evt.get('natureza', 'EVENTO')} - {evt.get('localizacao_completa', '')}"

            safe_desc = html.escape(desc)

            points.append({
                'lat': found_lat,
                'lng': found_lng,
                'description': safe_desc,
                'type': 'exogenous',
                'raw_event': evt
            })

    return jsonify({
        'events_processed': len(events),
        'points_found': len(points),
        'points': points
    })

@app.route('/api/exogenous/save', methods=['POST'])
def save_exogenous():
    global exogenous_events
    import json
    data = request.get_json()
    points = data.get('points', [])
    original_text = data.get('original_text', '')

    if not points:
        return jsonify({'error': 'Nenhum ponto para salvar.'}), 400

    new_entry = {
        'id': str(len(exogenous_events) + 1),
        'timestamp': pd.Timestamp.now().isoformat(),
        'original_text': original_text,
        'points': points
    }

    try:
        current_events = []
        if os.path.exists(EXOGENOUS_FILE):
             with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                 try:
                     current_events = json.load(f)
                 except json.JSONDecodeError:
                     current_events = []

        current_events.append(new_entry)

        with open(EXOGENOUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(current_events, f, ensure_ascii=False, indent=2)

        exogenous_events.append(new_entry)

        update_exogenous_state()

        return jsonify({'status': 'success', 'message': 'Eventos salvos e malha atualizada.'})

    except Exception as e:
        print(f"Erro ao salvar eventos: {e}")
        return jsonify({'error': f'Erro ao salvar: {e}'}), 500

try:
    load_data_and_models()
    enrich_regions()
except Exception as e:
    print(f"Startup error: {e}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
