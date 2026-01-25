from flask import Flask, jsonify, render_template, request
import pandas as pd
import geopandas as gpd
import pickle
import torch
import numpy as np
import os
import copy
from shapely.geometry import shape, Point, Polygon
from src.model import STGCN

app = Flask(__name__)

# Configuração e Carregamento de Dados (usar caminhos absolutos relativos a este arquivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_CVLI_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvli.pth')
MODEL_CVP_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvp.pth')

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

# Parâmetros de janela
WINDOW_CVLI = 180
WINDOW_CVP = 30

def load_data_and_models():
    global nodes_gdf, nodes_gdf_proj, adj_matrix, node_features, model_cvli, model_cvp, device, norm_adj, dates

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

        # Pre-computar normalização da adjacência
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
    for idx, row in nodes_gdf.iterrows():
        val = row['CIDADE']
        if not val or len(str(val)) < 2:
            reg = get_region_name(row, row.geometry)
            nodes_gdf.at[idx, 'CIDADE'] = reg

# Hook into request or startup
# load_data_and_models calls this?
# load_data_and_models() -> I'll add the call there.

if __name__ == "__main__":
    enrich_regions() # Call here for startup
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # When imported (production), we might want to call it too
    enrich_regions()
