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
MODEL_CVLI_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvli.pth')
MODEL_CVP_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvp.pth')
EXOGENOUS_FILE = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')

# Valores padrão caso arquivos não estejam presentes
nodes_gdf = None
polygons_json_cache = None
nodes_gdf_proj = None
nodes_centroids_proj = None
adj_matrix = None
original_adj_matrix = None
node_features = None
model_cvli = None
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


def build_node_search_index():
    """Populate global `node_search_index` from `nodes_gdf`.

    Each item contains original name, lowercased name, stripped name (no accents),
    and centroid coordinates (lat, lng).
    """
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
WINDOW_CVLI = 180
WINDOW_CVP = 30

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
    # Helper to find nodes within radius
    nearby_indices = []
    try:
        p_geo = Point(lng, lat)

        if nodes_gdf_proj is not None:
             # Project point to 3857 for meters distance
             s = gpd.GeoSeries([p_geo], crs="EPSG:4326").to_crs("EPSG:3857")
             p_proj = s.iloc[0]

             if nodes_centroids_proj is not None:
                 # Spatial Index Search
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
             # Fallback (degrees)
             if nodes_gdf is not None:
                 centroids = nodes_gdf.geometry.centroid
                 dists = centroids.distance(p_geo)
                 nearby_indices = dists[dists < 0.005].index.tolist() # Approx 0.005 deg ~ 500m

        if not nearby_indices:
             # Find at least one closest
             if nodes_gdf is not None:
                 if nodes_gdf_proj is not None:
                     if nodes_centroids_proj is not None:
                         # Fallback to linear search for robustness (sindex.nearest showed inconsistencies in tests)
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

    # We modify adj_matrix IN PLACE.
    # Reloading from pickle resets it, so this is deterministic replay.

    # Reset affected nodes set for this deterministic run
    exogenous_affected_nodes.clear()
    exogenous_critical_nodes.clear()

    count_affected = 0
    for batch in exogenous_events:
        points = batch.get('points', [])

        for pt in points:
            # Support both format {lat, lng} and [lat, lng] just in case
            lat = pt.get('lat') if isinstance(pt, dict) else (pt[0] if isinstance(pt, list) and len(pt)>0 else None)
            lng = pt.get('lng') if isinstance(pt, dict) else (pt[1] if isinstance(pt, list) and len(pt)>1 else None)

            if lat is None or lng is None: continue

            # detect if this point's raw_event indicates a critical (lethal) event
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
                # Apply amplification (Conflict Logic)
                critical_flag = _is_critical_event(pt)
                amplification_factor = 10.0 if critical_flag else 5.0
                adj_matrix[idx, :] *= amplification_factor
                adj_matrix[:, idx] *= amplification_factor
                # Track critical nodes
                if critical_flag:
                    exogenous_critical_nodes.add(idx)

                # Also apply to geo/faction matrices if present
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

    # 1. Restore clean state for all matrices
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

    # 2. Apply all events
    apply_exogenous_events()

    # 3. Recompute normalization
    if adj_geo is not None and adj_faction is not None:
        norm_adj_list = compute_norm_adj_list([adj_geo, adj_faction])
        norm_adj = norm_adj_list[0] if norm_adj_list else None
    else:
        norm_adj = compute_norm_adj(adj_matrix)

def load_data_and_models():
    global nodes_gdf, polygons_json_cache, nodes_gdf_proj, nodes_centroids_proj, adj_matrix, node_features, model_cvli, model_cvp, device, norm_adj, dates, original_adj_matrix
    global adj_geo, adj_faction, norm_adj_list
    global ibge_bairros_cache, ibge_municipios_cache

    # Load Static Data
    try:
        # json is imported globally
        static_file = os.path.join(BASE_DIR, 'data', 'static', 'fortaleza_bairros_coords.json')
        if os.path.exists(static_file):
            with open(static_file, 'r', encoding='utf-8') as f:
                ibge_bairros_cache = json.load(f)

        static_file_mun = os.path.join(BASE_DIR, 'data', 'static', 'ceara_municipios_coords.json')
        if os.path.exists(static_file_mun):
            with open(static_file_mun, 'r', encoding='utf-8') as f:
                ibge_municipios_cache = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar dados estáticos: {e}")

    # Prefer single-file package, but support split 'graph_data' directory when the pkl is absent
    data_pack = None
    if not os.path.exists(DATA_FILE):
        graph_dir = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
        if os.path.exists(graph_dir) and os.path.isdir(graph_dir):
            print("AVISO: processed_graph_data.pkl não encontrado — tentando carregar chunks em data/processed/graph_data/")
            try:
                data_pack = {}
                # nodes_gdf: prefer pickle backup
                ng_path_pkl = os.path.join(graph_dir, 'nodes_gdf.pkl')
                ng_path_json = os.path.join(graph_dir, 'nodes_gdf.json')
                if os.path.exists(ng_path_pkl):
                    with open(ng_path_pkl, 'rb') as f:
                        data_pack['nodes_gdf'] = pickle.load(f)
                elif os.path.exists(ng_path_json):
                    # load geojson as GeoDataFrame
                    try:
                        data_pack['nodes_gdf'] = gpd.read_file(ng_path_json)
                    except Exception:
                        with open(ng_path_json, 'r', encoding='utf8') as f:
                            geojson = json.load(f)
                        data_pack['nodes_gdf'] = gpd.GeoDataFrame.from_features(geojson['features'])

                # adjacencies: .npz or .npy
                for name in ('adj_geo', 'adj_faction', 'adj_matrix'):
                    p_npz = os.path.join(graph_dir, f"{name}.npz")
                    p_npy = os.path.join(graph_dir, f"{name}.npy")
                    if os.path.exists(p_npz):
                        try:
                            import scipy.sparse as sp
                            data_pack[name] = sp.load_npz(p_npz).toarray()
                        except Exception:
                            data_pack[name] = np.load(p_npz, allow_pickle=True)
                    elif os.path.exists(p_npy):
                        data_pack[name] = np.load(p_npy, allow_pickle=True)

                # node_features
                nf = os.path.join(graph_dir, 'node_features.npy')
                if os.path.exists(nf):
                    data_pack['node_features'] = np.load(nf, allow_pickle=True)

                # dates (json or pickle)
                djson = os.path.join(graph_dir, 'dates.json')
                dpkl = os.path.join(graph_dir, 'dates.pkl')
                if os.path.exists(djson):
                    try:
                        with open(djson, 'r', encoding='utf8') as f:
                            data_pack['dates'] = json.load(f)
                    except Exception:
                        pass
                elif os.path.exists(dpkl):
                    try:
                        with open(dpkl, 'rb') as f:
                            data_pack['dates'] = pickle.load(f)
                    except Exception:
                        pass

                # If we found nothing useful, fall back to warning
                if not data_pack:
                    print("AVISO: Nenhum chunk foi encontrado em data/processed/graph_data/")
                    return

                print("Carregado pacote a partir de chunks com chaves:", list(data_pack.keys()))
            except Exception as e:
                print(f"Erro ao reconstruir a partir de chunks: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            print("AVISO: Arquivo de dados não encontrado. Execute o processamento de dados.")
            return

    print("Carregando dados para API...")
    try:
        # If not already reconstructed from chunks, load the single-file package
        if data_pack is None:
            with open(DATA_FILE, 'rb') as f:
                data_pack = pickle.load(f)

        nodes_gdf = data_pack.get('nodes_gdf')
        polygons_json_cache = None
        # Load both adjacencies if available
        adj_geo = data_pack.get('adj_geo')
        adj_faction = data_pack.get('adj_faction')
        # Backward compat: prefer explicit adj_matrix if present, else use adj_geo
        adj_matrix = data_pack.get('adj_matrix')
        if adj_matrix is None:
            adj_matrix = adj_geo
        node_features = data_pack.get('node_features')
        dates = data_pack.get('dates')
        # Fallback: if dates missing from the package, try to load from graph_data/dates.pkl or dates.json
        if not dates:
            try:
                graph_dir = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
                dpkl = os.path.join(graph_dir, 'dates.pkl')
                djson = os.path.join(graph_dir, 'dates.json')
                if os.path.exists(dpkl):
                    try:
                        with open(dpkl, 'rb') as fh:
                            loaded = pickle.load(fh)
                            if (loaded is not None) and (hasattr(loaded, '__len__') and len(loaded) > 0):
                                dates = loaded
                                print(f"[load_data_and_models] Fallback loaded dates from {dpkl}: type={type(dates)}, len={len(dates)}")
                    except Exception as e:
                        print(f"[load_data_and_models] Failed to load dates.pkl fallback: {e}")
                elif os.path.exists(djson):
                    try:
                        with open(djson, 'r', encoding='utf-8') as fh:
                            loaded = json.load(fh)
                            if (loaded is not None) and (hasattr(loaded, '__len__') and len(loaded) > 0):
                                try:
                                    dates = pd.to_datetime(loaded)
                                except Exception:
                                    dates = loaded
                                print(f"[load_data_and_models] Fallback loaded dates from {djson}: type={type(dates)}, len={(len(dates) if hasattr(dates,'__len__') else 'unknown')}")
                    except Exception as e:
                        print(f"[load_data_and_models] Failed to load dates.json fallback: {e}")
            except Exception:
                pass

        # Create projected GeoDataFrame for metric calculations (EPSG:3857 - Web Mercator)
        # This fixes UserWarning about geographic CRS during distance calculation
        if nodes_gdf is not None:
            try:
                nodes_gdf_proj = nodes_gdf.to_crs(epsg=3857)
                nodes_centroids_proj = nodes_gdf_proj.geometry.centroid
                # Force spatial index build
                _ = nodes_centroids_proj.sindex
            except Exception as e:
                print(f"Erro ao projetar nodes_gdf: {e}")

            # Build search index
            build_node_search_index()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = node_features.shape[0]

        # Capture original matrix before modifications
        if adj_matrix is not None:
            original_adj_matrix = adj_matrix.copy()
        # If we have both adj_geo and adj_faction, compute normalized list later

        # Load and Apply Exogenous Events (Permanent Adaptation)
        load_exogenous_events()
        apply_exogenous_events()

        # Pre-computar normalização das adjacências (com modificações)
        if adj_geo is not None and adj_faction is not None:
            try:
                norm_adj_list = compute_norm_adj_list([adj_geo, adj_faction])
                norm_adj = norm_adj_list[0] if norm_adj_list else None
            except Exception as e:
                print(f"Erro ao normalizar adjacências: {e}")
                norm_adj = compute_norm_adj(adj_matrix)
        else:
            norm_adj = compute_norm_adj(adj_matrix)

        # Carregar Modelos
        def _adapt_state_dict_for_multigraph(state_dict, num_graphs):
            # If checkpoint was saved with single-graph GCN (`gcn.weight`), expand into
            # multiple `gcn.weights.{i}` entries expected by current MultiGraphConvolution.
            try:
                sd = dict(state_dict)
            except Exception:
                sd = state_dict
            keys = list(sd.keys())
            for k in keys:
                if k.endswith('.gcn.weight'):
                    base = k[:-len('.gcn.weight')]
                    # replicate the single weight across all graph slots
                    w = sd[k]
                    for i in range(num_graphs):
                        newk = f"{base}.gcn.weights.{i}"
                        sd[newk] = w
                    # remove old key to avoid unexpected-key warnings
                    del sd[k]
            return sd

        if os.path.exists(MODEL_CVLI_PATH):
            print(f"Carregando modelo CVLI de {MODEL_CVLI_PATH}...")
            num_graphs = len(norm_adj_list) if norm_adj_list is not None else 1
            try:
                raw_state = torch.load(MODEL_CVLI_PATH, map_location=device)
                # normalize DataParallel keys (module.) if present
                state_dict = {}
                for k,v in dict(raw_state).items():
                    nk = k[7:] if k.startswith('module.') else k
                    state_dict[nk] = v

                # adapt single-graph weight layout if needed
                state_dict = _adapt_state_dict_for_multigraph(state_dict, num_graphs)

                # If checkpoint conv_final temporal kernel differs from expected WINDOW_CVLI,
                # instantiate model with the checkpoint time_steps so shapes match.
                ck_time_steps = None
                if 'conv_final.weight' in state_dict:
                    try:
                        ck_time_steps = state_dict['conv_final.weight'].shape[-1]
                    except Exception:
                        ck_time_steps = None

                instantiate_ts = WINDOW_CVLI if ck_time_steps is None else ck_time_steps
                if instantiate_ts != WINDOW_CVLI:
                    print(f"Checkpoint time_steps={ck_time_steps} != WINDOW_CVLI={WINDOW_CVLI}, adapting model instantiation.")

                m_cvli = STGCN(num_nodes=num_nodes, in_channels=1, time_steps=instantiate_ts, num_classes=1, num_graphs=num_graphs)
                m_cvli.load_state_dict(state_dict, strict=False)
                m_cvli.to(device)
                m_cvli.eval()
                model_cvli = m_cvli
            except Exception as e:
                print(f"Erro ao carregar state_dict CVLI: {e}")
        else:
            print(f"AVISO: Modelo CVLI não encontrado em {MODEL_CVLI_PATH}")

        if os.path.exists(MODEL_CVP_PATH):
            print(f"Carregando modelo CVP de {MODEL_CVP_PATH}...")
            num_graphs = len(norm_adj_list) if norm_adj_list is not None else 1
            try:
                raw_state = torch.load(MODEL_CVP_PATH, map_location=device)
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

                instantiate_ts = WINDOW_CVP if ck_time_steps is None else ck_time_steps
                if instantiate_ts != WINDOW_CVP:
                    print(f"Checkpoint time_steps={ck_time_steps} != WINDOW_CVP={WINDOW_CVP}, adapting model instantiation.")

                m_cvp = STGCN(num_nodes=num_nodes, in_channels=1, time_steps=instantiate_ts, num_classes=1, num_graphs=num_graphs)
                m_cvp.load_state_dict(state_dict, strict=False)
                m_cvp.to(device)
                m_cvp.eval()
                model_cvp = m_cvp
            except Exception as e:
                print(f"Erro ao carregar state_dict CVP: {e}")
        else:
            print(f"AVISO: Modelo CVP não encontrado em {MODEL_CVP_PATH}")

    except Exception as e:
        print(f"Erro ao carregar dados/modelos: {e}")
        import traceback
        traceback.print_exc()

# Executa carregamento inicial
load_data_and_models()


def _periodic_reload_loop(interval_minutes: int):
    interval = max(1, int(interval_minutes)) * 60
    while True:
        try:
            print("[PeriodicReload] Scheduled reload starting...")
            try:
                app._periodic_update_in_progress = True
            except Exception:
                pass

            # Reuse the same load procedure used at startup
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


# Start periodic reloads (30 minutes)
start_periodic_reload(30)

def format_trend(prediction, history_avg, risk_score=None):
    if history_avg == 0:
        if prediction > 0.001:
            return "Nova atividade criminal detectada"
        else:
            return "Situação estável (Baixo Risco)"

    change_pct = ((prediction - history_avg) / history_avg) * 100

    # Limiares de 15% para variação significativa
    if change_pct > 15:
        return "Aumento recente da criminalidade"
    elif change_pct < -15:
        return "Redução da atividade criminal"
    else:
        # Estabilidade: verificar o nível do risco absoluto
        if risk_score is not None:
            if risk_score > 60:
                return "Valor histórico alto para o período"
            elif risk_score > 20:
                return "Atividade criminal moderada"

        return "Situação estável (Baixo Risco)"

def get_region_name(feature_props, geometry):
    # Tenta usar a coluna CIDADE se existir e não for vazia
    city = feature_props.get('CIDADE', '')
    if city and isinstance(city, str) and len(city.strip()) > 1:
        return city

    # Fallback geográfico simples (BBox Fortaleza)
    # Bounds aprox: -38.66, -3.90 to -38.40, -3.65
    try:
        if geometry:
            centroid = shape(geometry).centroid
            lon, lat = centroid.x, centroid.y
            if -38.66 <= lon <= -38.40 and -3.90 <= lat <= -3.65:
                return "Fortaleza"
    except Exception:
        pass

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

        # Atualizar props com região inferida antes de enviar?
        # GeoJSON serialization is tricky to inject props on the fly efficiently.
        # Mas o frontend espera CIDADE.
        # Vamos deixar o frontend usar CIDADE, e se estiver vazio, usar o bbox logic.
        # Mas o bbox logic é server side aqui.
        # Melhor: injetar 'region_inferred' no geojson

        # Convert to json string first
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

    # Reuse calculate_risk to get fresh data
    # calculate_risk returns a Flask Response object
    response = calculate_risk()
    if response.status_code != 200:
        return response

    risk_data = response.get_json()
    all_nodes = risk_data.get('data', [])

    # Sort by risk score descending
    sorted_nodes = sorted(all_nodes, key=lambda x: x['risk_score'], reverse=True)

    # Take Top 30
    top_30 = sorted_nodes[:30]

    # Build Nodes List
    graph_nodes = []
    top_indices = set()

    for item in top_30:
        nid = item['node_id']
        top_indices.add(nid)

        # Get name from nodes_gdf if possible, or use ID
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

    # Build Links List
    links = []

    # 1. Spatial/Influence Links (from Adjacency Matrix)
    # Iterate only through the top 30 nodes to find connections BETWEEN them
    for source in top_30:
        u = source['node_id']
        for target in top_30:
            v = target['node_id']
            if u == v: continue

            # Check adjacency weight
            # adj_matrix is numpy array
            try:
                weight = float(adj_matrix[u, v])
                # Threshold: usually normalized matrix has small values.
                # If it's the raw matrix, values are 0 or 1 (or distance based).
                # If it's loaded from 'processed_graph_data.pkl', it usually contains 1 for connection + self loops.
                # Let's check if weight > 0.
                if weight > 0:
                    links.append({
                        'source': u,
                        'target': v,
                        'weight': weight,
                        'type': 'influence' # Influence direction u -> v
                    })
            except Exception:
                pass

    # 2. Faction Links (Optional: Connect nodes of same faction if they are not already connected?)
    # Visual clutter risk. Let's rely on node color for faction and links for influence as requested.

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
        sim_type = data.get('type', 'suppression') # 'suppression' or 'exogenous'

        # Copiar matriz original
        adj_copy = adj_matrix.copy()

        # Track affected nodes for post-processing
        affected_nodes = set()

        # Use projected centroids if available for accurate distance (Meters)
        centroids = None
        if nodes_gdf_proj is not None:
             if nodes_centroids_proj is None:
                 centroids = nodes_gdf_proj.geometry.centroid
        else:
             centroids = nodes_gdf.geometry.centroid

        for pt in points:
            if len(pt) == 2:
                # pt is [lat, lon], Create Point(lon, lat)
                p_geo = Point(pt[1], pt[0])

                nearby_indices = []

                if nodes_gdf_proj is not None:
                    # Project point to 3857
                    try:
                        s = gpd.GeoSeries([p_geo], crs="EPSG:4326").to_crs("EPSG:3857")
                        p_proj = s.iloc[0]

                        if nodes_centroids_proj is not None:
                             # Spatial Index Search
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
                        # Fallback
                        dists = nodes_gdf.geometry.centroid.distance(p_geo)
                        nearby_indices = [dists.idxmin()]
                else:
                    # Fallback (degrees)
                    dists = centroids.distance(p_geo)
                    # Approx 0.005 degrees ~ 500m
                    nearby_indices = dists[dists < 0.005].index.tolist()
                    if not nearby_indices:
                        nearby_indices = [dists.idxmin()]

                # Modify Adjacency
                for idx in nearby_indices:
                    affected_nodes.add(idx)
                    if sim_type == 'suppression':
                        # Reduzir pesos drasticamente (isolamento)
                        suppression_factor = 0.05
                        adj_copy[idx, :] *= suppression_factor
                        adj_copy[:, idx] *= suppression_factor
                    elif sim_type == 'exogenous':
                        # Amplificar difusão
                        amplification_factor = 5.0
                        adj_copy[idx, :] *= amplification_factor
                        adj_copy[:, idx] *= amplification_factor

        # Re-normalizar matriz
        if adj_geo is not None and adj_faction is not None:
            # Apply same modifications to copies of geo/faction matrices
            adj_geo_copy = adj_geo.copy()
            adj_faction_copy = adj_faction.copy()
            # If we modified adj_copy earlier, propagate the same changes to both copies
            # (adj_copy was modified in place for affected indices)
            # For simplicity, if adj_copy differs we will recompute geo/faction normalized lists from their copies
            sim_norm_list = compute_norm_adj_list([adj_geo_copy, adj_faction_copy])
            response = calculate_risk(custom_norm_adj=sim_norm_list)
        else:
            adj_tensor = torch.FloatTensor(adj_copy)
            rowsum = adj_tensor.sum(1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            sim_norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)

            # Calculate Risk with modified graph
            response = calculate_risk(custom_norm_adj=sim_norm_adj)

        # Post-Process Result for Immediate Visual Feedback
        if response.status_code == 200:
            result_json = response.get_json()
            data_list = result_json.get('data', [])

            for item in data_list:
                nid = item['node_id']
                if nid in affected_nodes:
                    if sim_type == 'suppression':
                        # Force risk down significantly (Mitigation)
                        factor = 0.15
                        item['risk_score'] = item['risk_score'] * factor
                        item['risk_score_cvli'] = item['risk_score_cvli'] * factor
                        item['risk_score_cvp'] = item['risk_score_cvp'] * factor

                        # Add descriptive reason
                        item['reasons'].insert(0, "Área sob controle (Equipe Tática)")

                        # Adjust visual prediction values
                        item['cvli_pred'] *= factor
                        item['cvp_pred'] *= factor

                    elif sim_type == 'exogenous':
                        # Force risk up (Conflict)
                        boost = 1.5
                        # Ensure at least 80 (Critical) or boost existing
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
    # Helper to ensure we pass a list of normalized adjacencies to the model
    def _ensure_adj_list(adj):
        if adj is None:
            return None
        if isinstance(adj, (list, tuple)):
            return list(adj)
        # If global norm_adj_list exists, prefer it
        if norm_adj_list is not None:
            return norm_adj_list
        return [adj]

    adj_for_model = _ensure_adj_list(current_norm_adj)
    if adj_for_model is None:
         return jsonify({'error': 'Matriz de adjacência não disponível.'}), 503

    try:
        # ---------------------------
        # Previsão CVLI
        # ---------------------------
        out_cvli = np.zeros((node_features.shape[0], 1))
        hist_avg_cvli = np.zeros(node_features.shape[0])
        hist_sum_cvli = np.zeros(node_features.shape[0])

        if model_cvli:
            # Ensure we slice the input to the temporal kernel the model expects so conv_final
            # reduces the temporal dimension to 1. If the checkpoint/model was instantiated
            # with a different time_steps than WINDOW_CVLI, use the model's conv_final kernel.
            try:
                model_ts = model_cvli.conv_final.kernel_size[-1]
            except Exception:
                model_ts = WINDOW_CVLI

            if node_features.shape[1] >= model_ts:
                input_cvli = node_features[:, -model_ts:, 0:1]
                input_tensor = torch.FloatTensor(input_cvli).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_cvli(input_tensor, adj_for_model)
                out_cvli = pred.squeeze(0).cpu().numpy()
                # apply desscale mapping if available
                try:
                    if _DESSCALE_A is not None:
                        out_cvli = (_DESSCALE_A * out_cvli) + _DESSCALE_B
                except Exception:
                    pass

                daily_avg = np.mean(input_cvli, axis=1)
                hist_avg_cvli = daily_avg[:, 0] * 3
                hist_sum_cvli = np.sum(input_cvli, axis=1)[:, 0]
            else:
                print("AVISO: Dados insuficientes para janela CVLI")

        # ---------------------------
        # Previsão CVP
        # ---------------------------
        out_cvp = np.zeros((node_features.shape[0], 1))
        hist_avg_cvp = np.zeros(node_features.shape[0])
        hist_sum_cvp = np.zeros(node_features.shape[0])

        if model_cvp:
            try:
                model_ts_cvp = model_cvp.conv_final.kernel_size[-1]
            except Exception:
                model_ts_cvp = WINDOW_CVP

            if node_features.shape[1] >= model_ts_cvp:
                input_cvp = node_features[:, -model_ts_cvp:, 1:2]
                input_tensor = torch.FloatTensor(input_cvp).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_cvp(input_tensor, adj_for_model)
                out_cvp = pred.squeeze(0).cpu().numpy()

                daily_avg = np.mean(input_cvp, axis=1)
                hist_avg_cvp = daily_avg[:, 0] * 1
                hist_sum_cvp = np.sum(input_cvp, axis=1)[:, 0]
            else:
                print("AVISO: Dados insuficientes para janela CVP")

        # ---------------------------
        # Processamento e Normalização (Híbrida)
        # ---------------------------

        # CVLI
        out_cvli = np.maximum(out_cvli, 0)
        cvli_raw = out_cvli[:, 0]
        cvli_adj = cvli_raw * 1.5

        # Boost baseado em histórico: se teve atividade, risco mínimo de 30%
        # Se prediction é 0 mas history > 0 -> Risk = 30
        # Se prediction > 0 -> Risk = normalized(prediction) + boost

        # Normalize prediction first
        min_cvli = np.min(cvli_adj)
        shifted_cvli = cvli_adj - min_cvli
        max_shift_cvli = np.max(shifted_cvli) if np.max(shifted_cvli) > 0 else 1
        normalized_risk_cvli = (shifted_cvli / max_shift_cvli) * 100

        # Apply History Boost
        # Find nodes with history > 0
        active_indices = hist_sum_cvli > 0
        # Ensure they have at least score 25 (Low-Medium)
        normalized_risk_cvli[active_indices] = np.maximum(normalized_risk_cvli[active_indices], 25.0)

        # If history is significantly high (> 3 events in 180 days?), boost to Medium-High (50)
        # Total events is low (max 3 daily), so sum > 5 is significant?
        very_active = hist_sum_cvli >= 3
        normalized_risk_cvli[very_active] = np.maximum(normalized_risk_cvli[very_active], 50.0)

        # Apply Exogenous Events Boost (Permanent/Persisted)
        if exogenous_affected_nodes:
            # Convert set to list or use numpy indexing
            # exogenous_affected_nodes contains int indices
            exo_indices = list(exogenous_affected_nodes)
            # Filter indices just in case they are out of bounds (unlikely but safe)
            exo_indices = [i for i in exo_indices if i < len(normalized_risk_cvli)]
            if exo_indices:
                # Force Risk >= 80.0 (Critical)
                normalized_risk_cvli[exo_indices] = np.maximum(normalized_risk_cvli[exo_indices], 80.0)

            # Force absolute critical for explicitly flagged lethal exogenous nodes
            try:
                if exogenous_critical_nodes:
                    crit_idxs = [i for i in exogenous_critical_nodes if i < len(normalized_risk_cvli)]
                    if crit_idxs:
                        normalized_risk_cvli[crit_idxs] = np.maximum(normalized_risk_cvli[crit_idxs], 95.0)
                        # Propagate to up to top_k faction neighbors that are geographically close (<=2000m)
                        top_k = 10
                        try:
                            if adj_faction is not None:
                                # prepare centroids in metric CRS if possible
                                centroids_metric = None
                                try:
                                    if nodes_gdf is not None:
                                        centroids_metric = nodes_gdf.to_crs(epsg=3857).geometry.centroid
                                except Exception:
                                    centroids_metric = None

                                for ci in crit_idxs:
                                    neigh = np.where(np.array(adj_faction[ci]) > 0)[0]
                                    neigh = [n for n in neigh if n != ci]
                                    if not neigh:
                                        continue

                                    # If we have metric centroids, compute distances and select nearby top_k
                                    selected = []
                                    if centroids_metric is not None:
                                        try:
                                            ci_pt = centroids_metric.iloc[ci]
                                            neigh_pts = centroids_metric.iloc[neigh]
                                            dists = neigh_pts.distance(ci_pt).values
                                            # keep those within 2000m
                                            close_idx = [neigh[i] for i, d in enumerate(dists) if d <= 2000]
                                            if not close_idx:
                                                # if none within 2000m, pick nearest top_k
                                                order = np.argsort(dists)
                                                selected = [neigh[i] for i in order[:top_k]]
                                            else:
                                                # if more than top_k, pick nearest among them
                                                if len(close_idx) > top_k:
                                                    close_pts = centroids_metric.loc[close_idx]
                                                    d_close = close_pts.distance(ci_pt).values
                                                    order = np.argsort(d_close)
                                                    selected = [close_idx[i] for i in order[:top_k]]
                                                else:
                                                    selected = close_idx
                                        except Exception:
                                            selected = neigh[:top_k]
                                    else:
                                        # Fallback: restrict to those that are also geo-neighbors (if adj_matrix/adj_geo available)
                                        try:
                                            if adj_geo is not None:
                                                geo_neigh = [n for n in neigh if adj_geo[ci, n] > 0]
                                                selected = geo_neigh[:top_k]
                                            else:
                                                selected = neigh[:top_k]
                                        except Exception:
                                            selected = neigh[:top_k]

                                    if selected:
                                        normalized_risk_cvli[selected] = np.maximum(normalized_risk_cvli[selected], 90.0)
                        except Exception:
                            pass
            except Exception:
                pass

        # CVP
        out_cvp = np.maximum(out_cvp, 0)
        cvp_raw = out_cvp[:, 0]
        # Aumentar sensibilidade CVP conforme solicitado (era 1.0)
        cvp_raw = cvp_raw * 1.5

        min_cvp = np.min(cvp_raw)
        shifted_cvp = cvp_raw - min_cvp
        max_shift_cvp = np.max(shifted_cvp) if np.max(shifted_cvp) > 0 else 1
        normalized_risk_cvp = (shifted_cvp / max_shift_cvp) * 100

        # Boost CVP similarly
        active_cvp = hist_sum_cvp > 0
        normalized_risk_cvp[active_cvp] = np.maximum(normalized_risk_cvp[active_cvp], 25.0)

        # Apply Exogenous Boost to CVP too
        if exogenous_affected_nodes:
            exo_indices = list(exogenous_affected_nodes)
            exo_indices = [i for i in exo_indices if i < len(normalized_risk_cvp)]
            if exo_indices:
                normalized_risk_cvp[exo_indices] = np.maximum(normalized_risk_cvp[exo_indices], 80.0)

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

        # Pre-fetch 'faction' column to avoid .iloc access inside the loop (Optimization)
        factions = nodes_gdf['faction'].tolist() if 'faction' in nodes_gdf.columns else [None] * len(nodes_gdf)

        for i in range(len(normalized_risk_cvli)):
            cvli_score = float(normalized_risk_cvli[i])
            cvp_score = float(normalized_risk_cvp[i])

            cvli_val = float(cvli_adj[i])
            cvp_val = float(cvp_raw[i])

            trend_cvli = format_trend(cvli_raw[i], hist_avg_cvli[i], risk_score=cvli_score)
            trend_cvp = format_trend(cvp_raw[i], hist_avg_cvp[i], risk_score=cvp_score)

            reasons = []

            if cvli_val > 0.01 or cvli_score > 20:
                reasons.append(f'CVLI: {trend_cvli}')

            if cvp_val > 0.01 or cvp_score > 20:
                reasons.append(f'CVP: {trend_cvp}')

            if hist_sum_cvli[i] > 0 and cvli_val < 0.01:
                reasons.append('Histórico recente de violência')

            # Check for Exogenous Reason
            if i in exogenous_affected_nodes:
                reasons.insert(0, "Conflito Ativo")

            if len(reasons) == 0:
                reasons.append('Situação estável (Baixo Risco)')

            if conn is not None and conn_mean > 0 and conn[i] > conn_mean * 1.5:
                reasons.append('Alta conectividade (Rota de fuga/acesso)')

            # Inject region logic here if needed, but updating nodes_gdf properties
            # for the separate polygons endpoint is harder.
            # Ideally, polygons endpoint should use the logic.
            # But the tooltip uses feature.properties which comes from polygons endpoint.
            # We can send 'region' in this risk payload and frontend updates it?
            # Yes, frontend updates feature.properties from risk data.

            # Infer Region
            # Access geometry from nodes_gdf? It's slow to do it per request.
            # We should rely on what's in nodes_gdf or frontend logic.
            # I will inject 'inferred_region' here if I can access geometry efficiently.
            # nodes_gdf.geometry.iloc[i]
            # It's fast enough for 2000 points?
            # Let's try to trust the frontend 'CIDADE' or update nodes_gdf in memory?
            # Updating nodes_gdf in memory once is better.

            # Human-friendly labels for non-statistical users
            def _status_label(score: float) -> str:
                if score >= 90:
                    return 'Crítico'
                if score >= 70:
                    return 'Alto'
                if score >= 40:
                    return 'Médio'
                return 'Baixo'

            def _prediction_text(val: float, kind: str='CVLI') -> str:
                # val is model output (aggregated prediction). Produce simple phrases.
                try:
                    if val <= 0.01:
                        return 'Sem novas ocorrências previstas'
                    if val < 1.0:
                        return 'Menos de 1 ocorrência prevista'
                    # round to one decimal for readability
                    rounded = round(val, 1)
                    unit = 'ocorrência' if rounded == 1 else 'ocorrências'
                    # For CVLI we usually predict a short horizon (dias/semana), keep generic
                    return f'Estimativa: ~{rounded} {unit} previstas'
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
                # friendly fields for UI
                'status_label': status_label,
                'risk_text': risk_text,
                'cvli_prediction_text': cvli_pred_text,
                'cvp_prediction_text': cvp_pred_text
            })

        # Update nodes_gdf in memory with inferred regions if missing (Run once?)
        # For simplicity, I will implement a check in get_polygons to update 'CIDADE' if empty.

        # Ensure we have `dates` available: try a best-effort fallback to load from graph_data
        try:
            if (dates is None) or (hasattr(dates, '__len__') and len(dates) == 0):
                graph_dir = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
                dpkl = os.path.join(graph_dir, 'dates.pkl')
                djson = os.path.join(graph_dir, 'dates.json')
                # Prefer pickle (preserves DatetimeIndex)
                if os.path.exists(dpkl):
                    try:
                        with open(dpkl, 'rb') as fh:
                            loaded = pickle.load(fh)
                            if (loaded is not None) and (hasattr(loaded, '__len__') and len(loaded) > 0):
                                dates = loaded
                                print(f"[calculate_risk] Loaded dates from {dpkl}: type={type(dates)}, len={len(dates)}")
                    except Exception as e:
                        print(f"[calculate_risk] Failed loading dates.pkl: {e}")
                elif os.path.exists(djson):
                    try:
                        with open(djson, 'r', encoding='utf-8') as fh:
                            loaded = json.load(fh)
                            if (loaded is not None) and (hasattr(loaded, '__len__') and len(loaded) > 0):
                                # Attempt to parse JSON date list to DatetimeIndex
                                try:
                                    dates = pd.to_datetime(loaded)
                                except Exception:
                                    dates = loaded
                                print(f"[calculate_risk] Loaded dates from {djson}: type={type(dates)}, len={(len(dates) if hasattr(dates,'__len__') else 'unknown')}" )
                    except Exception as e:
                        print(f"[calculate_risk] Failed loading dates.json: {e}")
        except Exception as e:
            print(f"[calculate_risk] Exception while attempting fallback load of dates: {e}")

        # Build meta object with safe defaults to avoid undefined values in the frontend
        # Use a visible fallback (em dash) so frontend doesn't render '?' for missing dates
        meta = {
            'window_cvli': WINDOW_CVLI,
            'window_cvp': WINDOW_CVP,
            'start_cvli': '—',
            'start_cvp': '—',
            'last_date': '—',
            'window_start': '—',
            'window_end': '—'
        }

        # Populate meta from available `dates` if present
        if dates is not None and hasattr(dates, '__len__') and len(dates) > 0:
            try:
                last_date_obj = pd.to_datetime(dates[-1])
                meta['last_date'] = str(last_date_obj.date())

                if len(dates) >= WINDOW_CVLI:
                    start_cvli = pd.to_datetime(dates[-WINDOW_CVLI])
                    meta['start_cvli'] = str(start_cvli.date())
                else:
                    meta['start_cvli'] = str(pd.to_datetime(dates[0]).date())

                if len(dates) >= WINDOW_CVP:
                    start_cvp = pd.to_datetime(dates[-WINDOW_CVP])
                    meta['start_cvp'] = str(start_cvp.date())
                else:
                    meta['start_cvp'] = str(pd.to_datetime(dates[0]).date())

                # Provide window_start/window_end for convenience to frontend
                meta['window_start'] = meta.get('start_cvli', '')
                meta['window_end'] = meta.get('last_date', '')
            except Exception:
                # If parsing fails, keep safe defaults (em dash)
                pass

        # Include which temporal window the loaded models expect (helps UI explain differences)
        try:
            meta['model_window_cvli'] = model_cvli.conv_final.kernel_size[-1] if model_cvli is not None else WINDOW_CVLI
        except Exception:
            meta['model_window_cvli'] = WINDOW_CVLI

        try:
            meta['model_window_cvp'] = model_cvp.conv_final.kernel_size[-1] if model_cvp is not None else WINDOW_CVP
        except Exception:
            meta['model_window_cvp'] = WINDOW_CVP

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        print(f"ERROR details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro ao calcular risco: {e}'}), 500

# Helper to enrich regions (One time run or on request)
def enrich_regions():
    global nodes_gdf, polygons_json_cache
    if nodes_gdf is None: return

    # Check if we need to infer
    # If CIDADE is missing or empty
    if 'CIDADE' not in nodes_gdf.columns:
        nodes_gdf['CIDADE'] = ''

    print("Enriching regions...")

    # 1. First Pass: Apply existing BBox logic (Fortaleza) for empty entries
    # Iterating is acceptable for the first pass or we can vectorize bbox too, but let's keep it simple for now as it handles Geometry logic.
    # Actually, let's optimize the existing loop to only touch empty ones.

    # We can try to do a bulk update for "Fortaleza" based on BBox if possible, but the loop is fine for now.
    # However, to support the "Whole Ceara", we need the distance fallback.

    mask_empty = (nodes_gdf['CIDADE'].isna()) | (nodes_gdf['CIDADE'].astype(str).str.len() < 2)

    if not mask_empty.any():
        return

    # Prepare indices to update
    indices_to_update = nodes_gdf[mask_empty].index

    # Optimization: Use vectorization for BBox check?
    # Bounds aprox: -38.66, -3.90 to -38.40, -3.65
    # Let's just run the loop for BBox check first.

    count_fortaleza = 0
    for idx in indices_to_update:
        # Use existing get_region_name logic which prioritizes BBox for Fortaleza
        # But get_region_name returns "RMF/Interior" if not in BBox.
        # We only want to set "Fortaleza" if it matches BBox.

        # We can reuse get_region_name but check return value.
        row = nodes_gdf.loc[idx]
        reg = get_region_name(row, row.geometry)
        if reg == "Fortaleza":
            nodes_gdf.at[idx, 'CIDADE'] = reg
            count_fortaleza += 1

    print(f"Assigned Fortaleza to {count_fortaleza} nodes via BBox.")

    # 2. Second Pass: Distance-based lookup for remaining empty entries (Whole Ceará)
    mask_still_empty = (nodes_gdf['CIDADE'].isna()) | (nodes_gdf['CIDADE'].astype(str).str.len() < 2) | (nodes_gdf['CIDADE'] == 'RMF/Interior')

    if mask_still_empty.any() and ibge_municipios_cache:
        try:
            nodes_sub = nodes_gdf[mask_still_empty]

            # Prepare Cache Arrays
            mun_names = list(ibge_municipios_cache.keys())
            mun_coords = list(ibge_municipios_cache.values()) # [[lat, lon], ...]
            mun_coords_arr = np.array(mun_coords)

            # Prepare Node Arrays
            # Geometry centroid (lon, lat) -> Need (lat, lon)
            # Fix UserWarning about centroid on geographic CRS by projecting first
            centroids_proj = nodes_sub.to_crs(epsg=3857).geometry.centroid
            centroids = centroids_proj.to_crs(nodes_sub.crs)
            node_coords = np.column_stack((centroids.y.values, centroids.x.values))

            # Calculate Distances
            dists = cdist(node_coords, mun_coords_arr)
            nearest_indices = np.argmin(dists, axis=1)

            # Assign
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

def strip_accents(text):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def find_node_coordinates(location_str):
    if nodes_gdf is None:
        return None

    # Normalize input
    loc_norm = normalize_location(location_str)
    loc_lower = loc_norm.lower()
    loc_stripped = strip_accents(loc_lower)

    # 0. Manual Locations
    if loc_norm in MANUAL_LOCATIONS:
         return (*MANUAL_LOCATIONS[loc_norm], 'manual')

    # Strategy: Check if any node name is contained in the location string
    best_match_item = None
    best_match_len = 0

    # Optimization: Use pre-computed index
    for item in node_search_index:
        name = item.name_lower
        name_stripped = item.name_stripped

        # Check if node name is in input location (e.g. "Timbó" in "Timbó, Maracanaú")
        # OR if input location is in node name (e.g. "Moura Brasil" in "Morro ... Moura Brasil")
        # Try exact match first, then stripped
        if name in loc_lower:
            if len(name) > best_match_len:
                best_match_item = item
                best_match_len = len(name)
        elif name_stripped in loc_stripped:
             # Match without accents
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
        return (best_match_item.lat, best_match_item.lng, 'specific') # lat, lon

    # Fallback 2: IBGE Neighborhoods Static List (Cached)
    if ibge_bairros_cache:
        try:
            for bairro, coords in ibge_bairros_cache.items():
                b_norm = normalize_location(bairro).lower()
                b_stripped = strip_accents(b_norm)
                if b_norm in loc_lower or b_stripped in loc_stripped:
                    return (coords[0], coords[1], 'specific')
        except Exception as e:
             print(f"Erro no fallback IBGE Bairros: {e}")

    # Fallback 3: Ceará Municipalities Static List (Cached)
    if ibge_municipios_cache:
        try:
            for municipio, coords in ibge_municipios_cache.items():
                m_norm = normalize_location(municipio).lower()
                if m_norm in loc_lower:
                    return (coords[0], coords[1], 'city')
        except Exception as e:
             print(f"Erro no fallback IBGE Municípios: {e}")

    # Fallback 4: Loose City Search (Word matching)
    # Checks if any word in the input strictly matches a known city key
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
            print(f"Erro no fallback loose city: {e}")

    # Fallback 5: Geometric Centroid from nodes_gdf (Last Resort)
    # Search for known Cities in GDF
    if 'CIDADE' in nodes_gdf.columns:
        known_cities = nodes_gdf['CIDADE'].unique()
        for city in known_cities:
            if not isinstance(city, str) or len(city) < 3: continue

            if city.lower() in loc_lower:
                city_nodes = nodes_gdf[nodes_gdf['CIDADE'] == city]
                if not city_nodes.empty:
                    # Fix UserWarning by projecting
                    c_proj = city_nodes.to_crs(epsg=3857).geometry.centroid
                    c_geo = c_proj.to_crs(city_nodes.crs)
                    city_centroid_x = c_geo.x.mean()
                    city_centroid_y = c_geo.y.mean()
                    return (city_centroid_y, city_centroid_x, 'fallback')

    # Fallback 6: Strip accents and try again (Recursive-ish but limited)
    # Only try once
    loc_stripped = strip_accents(loc_norm)
    if loc_stripped != loc_norm:
        # Check Manual again
        if loc_stripped in MANUAL_LOCATIONS:
             return (*MANUAL_LOCATIONS[loc_stripped], 'manual')
        # Check cache keys stripped? Too expensive maybe.
        # But we can check normalized cache keys against stripped input?
        # Actually, let's just rely on the fact that if input has no accents but cache does, 'in' might fail.
        # But here we strip input. If cache has accents, 'bairro' (with accents) in 'loc_stripped' (no accents) will fail.
        # We need to compare stripped to stripped.
        # This is getting heavy. Let's trust normalization handles most.
        pass

    return None

@app.route('/api/exogenous/parse', methods=['POST'])
def parse_exogenous():
    data = request.get_json()
    text = data.get('text', '')

    # Use LLM Service (Gemini)
    events = process_exogenous_text(text)
    points = []

    for evt in events:
        found_lat = None
        found_lng = None
        match_quality = 0 # 1=City, 2=Bairro, 3=Specific

        # 1. Try Full Address / Location
        if evt.get('localizacao_completa'):
            res = find_node_coordinates(evt['localizacao_completa'])
            if res:
                found_lat, found_lng, mtype = res
                match_quality = 3 if mtype == 'specific' else 2

        # 2. Try Neighborhood (Bairro) if not found or low quality
        if (not found_lat or match_quality < 2) and evt.get('bairro'):
            res = find_node_coordinates(evt['bairro'])
            if res:
                # If we found nothing before, take this.
                # If we found something before (e.g. city match), take this as it is likely better (neighborhood).
                # find_node_coordinates usually returns 'specific' for neighborhoods.
                lat_b, lng_b, mtype_b = res
                if mtype_b == 'specific':
                     found_lat, found_lng = lat_b, lng_b
                     match_quality = 3

        # 3. Try City (Municipio)
        if not found_lat and evt.get('municipio'):
            res = find_node_coordinates(evt['municipio'])
            if res:
                found_lat, found_lng, _ = res
                match_quality = 1

        if found_lat is not None and found_lng is not None:
            import html
            # Construct description from LLM fields
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
    import json # Safeguard
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
        # Load existing first to append correctly (race condition possible but low risk here)
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

        # Trigger Reload/Re-apply
        # We update the in-memory state efficiently without reloading everything.
        exogenous_events.append(new_entry)

        update_exogenous_state()

        return jsonify({'status': 'success', 'message': 'Eventos salvos e malha atualizada.'})

    except Exception as e:
        print(f"Erro ao salvar eventos: {e}")
        return jsonify({'error': f'Erro ao salvar: {e}'}), 500

# Hook into request or startup
# load_data_and_models calls this?
# load_data_and_models() -> I'll add the call there.

if __name__ == "__main__":
    enrich_regions() # Call here for startup
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # When imported (production), we might want to call it too
    enrich_regions()
