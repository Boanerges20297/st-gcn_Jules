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

app = Flask(__name__)

# Configuração e Carregamento de Dados (usar caminhos absolutos relativos a este arquivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_CVLI_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvli.pth')
MODEL_CVP_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvp.pth')
EXOGENOUS_FILE = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')

# Valores padrão caso arquivos não estejam presentes
nodes_gdf = None
nodes_gdf_proj = None
adj_matrix = None
node_features = None
model_cvli = None
model_cvp = None
device = None
norm_adj = None
dates = None

# Static Data Cache
ibge_bairros_cache = None
ibge_municipios_cache = None
exogenous_events = []

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
             centroids = nodes_gdf_proj.geometry.centroid
             dists = centroids.distance(p_proj)
             nearby_indices = dists[dists < radius_m].index.tolist()
        else:
             # Fallback (degrees)
             centroids = nodes_gdf.geometry.centroid
             dists = centroids.distance(p_geo)
             nearby_indices = dists[dists < 0.005].index.tolist() # Approx 0.005 deg ~ 500m

        if not nearby_indices:
             # Find at least one closest
             if nodes_gdf is not None:
                 if nodes_gdf_proj is not None:
                     dists = nodes_gdf_proj.geometry.centroid.distance(p_proj)
                 else:
                     dists = nodes_gdf.geometry.centroid.distance(p_geo)
                 nearby_indices = [dists.idxmin()]

    except Exception as e:
        print(f"Erro ao buscar nodes próximos: {e}")

    return nearby_indices

def apply_exogenous_events():
    global adj_matrix
    if not exogenous_events or adj_matrix is None:
        return

    print("Aplicando eventos exógenos na malha...")

    # We modify adj_matrix IN PLACE.
    # Reloading from pickle resets it, so this is deterministic replay.

    count_affected = 0
    for batch in exogenous_events:
        points = batch.get('points', [])

        for pt in points:
            # Support both format {lat, lng} and [lat, lng] just in case
            lat = pt.get('lat') if isinstance(pt, dict) else (pt[0] if isinstance(pt, list) and len(pt)>0 else None)
            lng = pt.get('lng') if isinstance(pt, dict) else (pt[1] if isinstance(pt, list) and len(pt)>1 else None)

            if lat is None or lng is None: continue

            indices = find_nearby_nodes(lat, lng)

            for idx in indices:
                # Apply amplification (Conflict Logic)
                amplification_factor = 5.0
                adj_matrix[idx, :] *= amplification_factor
                adj_matrix[:, idx] *= amplification_factor
                count_affected += 1

    print(f"Malha adaptada: {count_affected} modificações aplicadas.")

def load_data_and_models():
    global nodes_gdf, nodes_gdf_proj, adj_matrix, node_features, model_cvli, model_cvp, device, norm_adj, dates
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

    if not os.path.exists(DATA_FILE):
        print("AVISO: Arquivo de dados não encontrado. Execute o processamento de dados.")
        return

    print("Carregando dados para API...")
    try:
        with open(DATA_FILE, 'rb') as f:
            data_pack = pickle.load(f)

        nodes_gdf = data_pack.get('nodes_gdf')
        adj_matrix = data_pack.get('adj_matrix')
        node_features = data_pack.get('node_features')
        dates = data_pack.get('dates')

        # Create projected GeoDataFrame for metric calculations (EPSG:3857 - Web Mercator)
        # This fixes UserWarning about geographic CRS during distance calculation
        if nodes_gdf is not None:
            try:
                nodes_gdf_proj = nodes_gdf.to_crs(epsg=3857)
            except Exception as e:
                print(f"Erro ao projetar nodes_gdf: {e}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = node_features.shape[0]

        # Load and Apply Exogenous Events (Permanent Adaptation)
        load_exogenous_events()
        apply_exogenous_events()

        # Pre-computar normalização da adjacência (com modificações)
        adj_tensor = torch.FloatTensor(adj_matrix)
        rowsum = adj_tensor.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)

        # Carregar Modelos
        if os.path.exists(MODEL_CVLI_PATH):
            print(f"Carregando modelo CVLI de {MODEL_CVLI_PATH}...")
            m_cvli = STGCN(num_nodes=num_nodes, in_channels=1, time_steps=WINDOW_CVLI, num_classes=1)
            try:
                state_dict = torch.load(MODEL_CVLI_PATH, map_location=device)
                m_cvli.load_state_dict(state_dict)
                m_cvli.to(device)
                m_cvli.eval()
                model_cvli = m_cvli
            except Exception as e:
                 print(f"Erro ao carregar state_dict CVLI: {e}")
        else:
            print(f"AVISO: Modelo CVLI não encontrado em {MODEL_CVLI_PATH}")

        if os.path.exists(MODEL_CVP_PATH):
            print(f"Carregando modelo CVP de {MODEL_CVP_PATH}...")
            m_cvp = STGCN(num_nodes=num_nodes, in_channels=1, time_steps=WINDOW_CVP, num_classes=1)
            try:
                state_dict = torch.load(MODEL_CVP_PATH, map_location=device)
                m_cvp.load_state_dict(state_dict)
                m_cvp.to(device)
                m_cvp.eval()
                model_cvp = m_cvp
            except Exception as e:
                print(f"Erro ao carregar state_dict CVP: {e}")
        else:
            print(f"AVISO: Modelo CVP não encontrado em {MODEL_CVP_PATH}")

    except Exception as e:
        print(f"Erro ao carregar dados/modelos: {e}")

# Executa carregamento inicial
load_data_and_models()

def format_trend(prediction, history_avg):
    if history_avg == 0:
        if prediction > 0.001:
            return f"Início de atividade detectada (Surgimento)"
        else:
            return "Estabilidade (Baixa Atividade)"

    change_pct = ((prediction - history_avg) / history_avg) * 100

    if change_pct > 10:
        return f"Tendência de Crescimento (+{change_pct:.1f}%)"
    elif change_pct < -10:
        return f"Tendência de Redução ({change_pct:.1f}%)"
    else:
        return f"Estabilidade ({change_pct:+.1f}%)"

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
    if nodes_gdf is None:
        return jsonify({'error': 'Dados de polígonos não carregados.'}), 503
    try:
        # Atualizar props com região inferida antes de enviar?
        # GeoJSON serialization is tricky to inject props on the fly efficiently.
        # Mas o frontend espera CIDADE.
        # Vamos deixar o frontend usar CIDADE, e se estiver vazio, usar o bbox logic.
        # Mas o bbox logic é server side aqui.
        # Melhor: injetar 'region_inferred' no geojson

        # Convert to json string first
        return nodes_gdf.to_json()
    except Exception as e:
        return jsonify({'error': f'Erro ao serializar polígonos: {e}'}), 500

@app.route('/api/risk')
def get_risk():
    return calculate_risk()

@app.route('/api/simulate', methods=['POST'])
def simulate_risk():
    if node_features is None or adj_matrix is None or nodes_gdf is None:
        return jsonify({'error': 'Dados não carregados.'}), 503

    try:
        data = request.get_json()
        points = data.get('points', [])
        sim_type = data.get('type', 'suppression') # 'suppression' or 'exogenous'

        # Copiar matriz original
        adj_copy = copy.deepcopy(adj_matrix)

        # Track affected nodes for post-processing
        affected_nodes = set()

        # Use projected centroids if available for accurate distance (Meters)
        if nodes_gdf_proj is not None:
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
                        dists = centroids.distance(p_proj)
                        # Radius 500m
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
                        item['reasons'].insert(0, "Presença de Equipe Tática (Risco Mitigado)")

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

                        item['reasons'].insert(0, "Conflito Ativo Detectado (Simulação)")

            return jsonify(result_json)
        else:
            return response

    except Exception as e:
        print(f"Erro na simulação: {e}")
        return jsonify({'error': f'Erro na simulação: {e}'}), 500

def calculate_risk(custom_norm_adj=None):
    if node_features is None or nodes_gdf is None:
        return jsonify({'error': 'Dados não carregados.'}), 503

    current_norm_adj = custom_norm_adj if custom_norm_adj is not None else norm_adj
    if current_norm_adj is None:
         return jsonify({'error': 'Matriz de adjacência não disponível.'}), 503

    try:
        # ---------------------------
        # Previsão CVLI
        # ---------------------------
        out_cvli = np.zeros((node_features.shape[0], 1))
        hist_avg_cvli = np.zeros(node_features.shape[0])
        hist_sum_cvli = np.zeros(node_features.shape[0])

        if model_cvli:
            if node_features.shape[1] >= WINDOW_CVLI:
                input_cvli = node_features[:, -WINDOW_CVLI:, 0:1]
                input_tensor = torch.FloatTensor(input_cvli).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_cvli(input_tensor, current_norm_adj)
                out_cvli = pred.squeeze(0).cpu().numpy()

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
            if node_features.shape[1] >= WINDOW_CVP:
                input_cvp = node_features[:, -WINDOW_CVP:, 1:2]
                input_tensor = torch.FloatTensor(input_cvp).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_cvp(input_tensor, current_norm_adj)
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

        for i in range(len(normalized_risk_cvli)):
            cvli_score = float(normalized_risk_cvli[i])
            cvp_score = float(normalized_risk_cvp[i])

            cvli_val = float(cvli_adj[i])
            cvp_val = float(cvp_raw[i])

            trend_cvli = format_trend(cvli_raw[i], hist_avg_cvli[i])
            trend_cvp = format_trend(cvp_raw[i], hist_avg_cvp[i])

            reasons = []

            if cvli_val > 0.01 or cvli_score > 20:
                reasons.append(f'CVLI: {trend_cvli}')

            if cvp_val > 0.01 or cvp_score > 20:
                reasons.append(f'CVP: {trend_cvp}')

            if hist_sum_cvli[i] > 0 and cvli_val < 0.01:
                reasons.append('Histórico de violência recente (Risco Residual)')

            if len(reasons) == 0:
                reasons.append('Estabilidade (Baixo Risco)')

            if conn is not None and conn_mean > 0 and conn[i] > conn_mean * 1.5:
                reasons.append('Área com alta conectividade')

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

            results.append({
                'node_id': int(i),
                'risk_score': cvli_score,
                'risk_score_cvli': cvli_score,
                'risk_score_cvp': cvp_score,
                'cvli_pred': cvli_val,
                'cvp_pred': cvp_val,
                'faction': nodes_gdf.iloc[i].get('faction') if 'faction' in nodes_gdf.columns else None,
                'reasons': reasons,
                'priority_cvli': bool(cvli_score >= cutoff_cvli)
            })

        # Update nodes_gdf in memory with inferred regions if missing (Run once?)
        # For simplicity, I will implement a check in get_polygons to update 'CIDADE' if empty.

        meta = {
            'window_cvli': WINDOW_CVLI,
            'window_cvp': WINDOW_CVP
        }
        if dates is not None and len(dates) > 0:
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

            if 'start_cvli' in meta:
                meta['window_start'] = meta['start_cvli']
                meta['window_end'] = meta['last_date']

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        print(f"ERROR details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro ao calcular risco: {e}'}), 500

# Helper to enrich regions (One time run or on request)
def enrich_regions():
    global nodes_gdf
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

def parse_ciops_text(text):
    events = []
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Format: 01 - ID - TEAM - OFFICER - NATURE - INVOLVEMENT - LOCATION (AIS) - DP - TIME
        # Split by ' - '
        parts = line.split(' - ')
        if len(parts) >= 7:
            # Heuristic: Index 4 is nature, Index 6 is location
            nature = parts[4].strip()
            location = parts[6].strip()

            # Clean location: remove (AIS...)
            location = re.sub(r'\s*\(AIS\d+\)', '', location)

            events.append({
                'nature': nature,
                'location': location,
                'raw': line
            })

    return events

def find_node_coordinates(location_str):
    if nodes_gdf is None:
        return None

    # Normalize input
    loc_norm = location_str.lower()

    # Strategy: Check if any node name is contained in the location string
    best_match = None
    best_match_len = 0

    for idx, row in nodes_gdf.iterrows():
        name = str(row['name']).lower()
        if not name or len(name) < 3:
            continue

        # Check if node name is in input location (e.g. "Timbó" in "Timbó, Maracanaú")
        # OR if input location is in node name (e.g. "Moura Brasil" in "Morro ... Moura Brasil")
        if name in loc_norm:
            if len(name) > best_match_len:
                best_match = row
                best_match_len = len(name)
        elif loc_norm in name and len(loc_norm) > 4: # Avoid short inputs matching long descriptions incorrectly
             # If input is substring of node name, we consider it a match too.
             # We prioritize the match length of the *matching segment*.
             # Here let's just accept it if we haven't found a better one.
             if len(loc_norm) > best_match_len:
                best_match = row
                best_match_len = len(loc_norm)

    if best_match is not None:
        centroid = best_match.geometry.centroid
        return (centroid.y, centroid.x) # lat, lon

    # Fallback 2: IBGE Neighborhoods Static List (Cached)
    if ibge_bairros_cache:
        try:
            for bairro, coords in ibge_bairros_cache.items():
                if bairro.lower() in loc_norm:
                    return (coords[0], coords[1])
        except Exception as e:
             print(f"Erro no fallback IBGE Bairros: {e}")

    # Fallback 3: Ceará Municipalities Static List (Cached)
    if ibge_municipios_cache:
        try:
            for municipio, coords in ibge_municipios_cache.items():
                if municipio.lower() in loc_norm:
                    return (coords[0], coords[1])
        except Exception as e:
             print(f"Erro no fallback IBGE Municípios: {e}")

    # Fallback 4: Loose City Search (Word matching)
    # Checks if any word in the input strictly matches a known city key
    if ibge_municipios_cache:
        try:
             import re
             words = re.findall(r'\b\w+\b', loc_norm)
             for word in words:
                 if len(word) < 4: continue
                 for mun, coords in ibge_municipios_cache.items():
                     if mun.lower() == word:
                         return (coords[0], coords[1])
        except Exception as e:
            print(f"Erro no fallback loose city: {e}")

    # Fallback 5: Geometric Centroid from nodes_gdf (Last Resort)
    # Search for known Cities in GDF
    if 'CIDADE' in nodes_gdf.columns:
        known_cities = nodes_gdf['CIDADE'].unique()
        for city in known_cities:
            if not isinstance(city, str) or len(city) < 3: continue

            if city.lower() in loc_norm:
                city_nodes = nodes_gdf[nodes_gdf['CIDADE'] == city]
                if not city_nodes.empty:
                    # Fix UserWarning by projecting
                    c_proj = city_nodes.to_crs(epsg=3857).geometry.centroid
                    c_geo = c_proj.to_crs(city_nodes.crs)
                    city_centroid_x = c_geo.x.mean()
                    city_centroid_y = c_geo.y.mean()
                    return (city_centroid_y, city_centroid_x)

    return None

@app.route('/api/exogenous/parse', methods=['POST'])
def parse_exogenous():
    data = request.get_json()
    text = data.get('text', '')

    events = parse_ciops_text(text)
    points = []

    for evt in events:
        coords = find_node_coordinates(evt['location'])
        if coords:
            # Sanitize description
            import html
            safe_desc = html.escape(f"{evt['nature']} - {evt['location']}")
            points.append({
                'lat': coords[0],
                'lng': coords[1],
                'description': safe_desc,
                'type': 'exogenous'
            })

    return jsonify({
        'events_processed': len(events),
        'points_found': len(points),
        'points': points
    })

@app.route('/api/exogenous/save', methods=['POST'])
def save_exogenous():
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
        # We need to reload data to reset adj_matrix and re-apply ALL events including this new one.
        # This is expensive but ensures consistency.
        # load_data_and_models handles full reload.
        load_data_and_models()

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
