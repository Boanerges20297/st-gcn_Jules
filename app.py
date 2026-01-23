from flask import Flask, jsonify, render_template, request
import pandas as pd
import geopandas as gpd
import pickle
import torch
import numpy as np
import os
import json
from datetime import datetime
from src.model import STGCN
from src.generate_data import generate_dummy_data

app = Flask(__name__)

# Configuração e Carregamento de Dados
DATA_FILE = 'data/processed_graph_data.pkl'
MODEL_CVLI_PATH = 'models/cvli_model.pth'
MODEL_CVP_PATH = 'models/cvp_model.pth'
EXOGENOUS_FILE = 'data/exogenous.json'

# Global vars
model_cvli = None
model_cvp = None
nodes_gdf = None
adj_matrix = None
node_features = None
norm_adj = None
device = None
dates_index = None # Pandas DateTimeIndex

CVLI_WINDOW = 180
CVP_WINDOW = 30

def normalize_adj(adj):
    """Aux function to normalize adjacency matrix."""
    adj_tensor = torch.FloatTensor(adj)
    rowsum = adj_tensor.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)
    return norm_adj

def load_resources():
    global model_cvli, model_cvp, nodes_gdf, adj_matrix, node_features, norm_adj, device, dates_index

    # Check if data exists, if not generate it
    if not os.path.exists(DATA_FILE):
        print("AVISO: Arquivo de dados não encontrado. Gerando dados fictícios...")
        try:
            generate_dummy_data()
        except Exception as e:
            print(f"ERRO CRÍTICO ao gerar dados: {e}")
            return

    print("Carregando dados para API...")
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)

    nodes_gdf = data_pack['nodes_gdf']
    adj_matrix = data_pack['adj_matrix']
    node_features = data_pack['node_features']
    dates_index = data_pack['dates']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]

    # Pre-computar normalização da adjacência original
    norm_adj = normalize_adj(adj_matrix)

    print("Inicializando modelos...")
    try:
        model_cvli = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=CVLI_WINDOW, num_classes=1)
        if os.path.exists(MODEL_CVLI_PATH):
            try:
                model_cvli.load_state_dict(torch.load(MODEL_CVLI_PATH, map_location=device))
                print("Modelo CVLI carregado com sucesso.")
            except:
                print("AVISO: Pesos do modelo CVLI incompatíveis. Reiniciando pesos.")
        else:
             print("AVISO: Modelo CVLI não encontrado. Usando pesos aleatórios.")
        model_cvli.to(device)
        model_cvli.eval()

        model_cvp = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=CVP_WINDOW, num_classes=1)
        if os.path.exists(MODEL_CVP_PATH):
            try:
                model_cvp.load_state_dict(torch.load(MODEL_CVP_PATH, map_location=device))
                print("Modelo CVP carregado com sucesso.")
            except:
                print("AVISO: Pesos do modelo CVP incompatíveis. Reiniciando pesos.")
        else:
             print("AVISO: Modelo CVP não encontrado. Usando pesos aleatórios.")
        model_cvp.to(device)
        model_cvp.eval()
    except Exception as e:
        print(f"Erro ao inicializar modelos: {e}")

load_resources()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/polygons')
def get_polygons():
    """Retorna os polígonos em GeoJSON."""
    if nodes_gdf is not None:
        return nodes_gdf.to_json()
    return jsonify({})

@app.route('/api/risk')
def get_risk():
    """Calcula e retorna o risco previsto."""
    
    simulation_params = request.args.get('simulation') # e.g. "lat,lon"
    selected_date = request.args.get('date') # YYYY-MM-DD
    selected_region = request.args.get('region') # Fortaleza, RMF, Interior
    
    results = []
    
    if model_cvli is None or model_cvp is None:
        # If models are still None for some reason, try reloading or return empty
        # This prevents 500 errors if load_resources failed silently
        if os.path.exists(DATA_FILE):
             load_resources()
        if model_cvli is None:
            return jsonify([])

    # 1. Date Slicing (Time Travel)
    # Find the index of the selected date. If not provided, use the last index.
    time_idx = -1
    if selected_date:
        try:
            target_date = pd.to_datetime(selected_date)
            # Find nearest date or exact match
            # Assumes dates_index is sorted
            locs = dates_index.get_indexer([target_date], method='nearest')
            time_idx = locs[0]
            # Ensure we have enough history from this point BACKWARDS
            # We need at least CVLI_WINDOW (180) days
            if time_idx < CVLI_WINDOW:
                time_idx = CVLI_WINDOW # Clamp to minimum
        except Exception as e:
            print(f"Erro na data: {e}")
            time_idx = len(dates_index) - 1
    else:
         time_idx = len(dates_index) - 1

    # Slice features up to time_idx
    # We need [time_idx - Window : time_idx]
    # Handle boundaries
    end_idx = time_idx + 1 # Slice is exclusive at end
    
    # 2. Determine Adjacency Matrix (Simulation)
    current_norm_adj = norm_adj
    
    if simulation_params:
        try:
            parts = simulation_params.split(',')
            sim_lat = float(parts[0])
            sim_lon = float(parts[1])
            radius = 0.01

            affected_indices = []
            centroids = nodes_gdf.geometry.centroid
            for i, point in enumerate(centroids):
                dist = np.sqrt((point.y - sim_lat)**2 + (point.x - sim_lon)**2)
                if dist < radius:
                    affected_indices.append(i)

            if affected_indices:
                sim_adj_matrix = adj_matrix.copy()
                for idx in affected_indices:
                    sim_adj_matrix[idx, :] *= 0.1
                    sim_adj_matrix[:, idx] *= 0.1
                    sim_adj_matrix[idx, idx] = 1.0
                current_norm_adj = normalize_adj(sim_adj_matrix)

        except Exception as e:
            print(f"Erro na simulação: {e}")

    # 3. CVLI Prediction
    start_idx_cvli = max(0, end_idx - CVLI_WINDOW)
    recent_features_cvli = node_features[:, start_idx_cvli:end_idx, :]
    
    # Check if we have enough data, pad if necessary (though dummy data is long enough)
    if recent_features_cvli.shape[1] < CVLI_WINDOW:
        # Pad with zeros if strictly necessary, or handle gracefully
        pass

    input_cvli = torch.FloatTensor(recent_features_cvli).permute(2, 0, 1).unsqueeze(0).to(device)
    # Ensure temporal dimension matches what model expects?
    # STGCN expects fixed time_steps usually.
    # If input is smaller than window, this might fail if Conv2d kernel is large.
    # Our model is defined with specific time_steps.

    with torch.no_grad():
        try:
             # STGCN fixed size check
            if input_cvli.shape[3] == CVLI_WINDOW:
                pred_cvli = model_cvli(input_cvli, current_norm_adj)
            else:
                # Fallback: Create a zero tensor of correct shape and fill
                padded = torch.zeros(1, input_cvli.shape[1], input_cvli.shape[2], CVLI_WINDOW).to(device)
                padded[:, :, :, -input_cvli.shape[3]:] = input_cvli
                pred_cvli = model_cvli(padded, current_norm_adj)
        except Exception as e:
            print(f"Erro inferencia CVLI: {e}")
            pred_cvli = torch.zeros(1, len(nodes_gdf), 1)

    pred_cvli = pred_cvli.squeeze().cpu().numpy()
    if pred_cvli.ndim == 0: pred_cvli = np.array([pred_cvli]) # Handle single node case?

    # 4. CVP Prediction
    start_idx_cvp = max(0, end_idx - CVP_WINDOW)
    recent_features_cvp = node_features[:, start_idx_cvp:end_idx, :]
    input_cvp = torch.FloatTensor(recent_features_cvp).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        try:
            if input_cvp.shape[3] == CVP_WINDOW:
                pred_cvp = model_cvp(input_cvp, current_norm_adj)
            else:
                padded = torch.zeros(1, input_cvp.shape[1], input_cvp.shape[2], CVP_WINDOW).to(device)
                padded[:, :, :, -input_cvp.shape[3]:] = input_cvp
                pred_cvp = model_cvp(padded, current_norm_adj)
        except Exception as e:
             print(f"Erro inferencia CVP: {e}")
             pred_cvp = torch.zeros(1, len(nodes_gdf), 1)

    pred_cvp = pred_cvp.squeeze().cpu().numpy()
    if pred_cvp.ndim == 0: pred_cvp = np.array([pred_cvp])

    # Percentiles
    cvli_threshold = np.percentile(pred_cvli, 90)
    cvp_threshold = np.percentile(pred_cvp, 90)

    max_cvli = np.max(pred_cvli) if np.max(pred_cvli) > 0 else 1
    max_cvp = np.max(pred_cvp) if np.max(pred_cvp) > 0 else 1

    for i in range(len(nodes_gdf)):
        # Region Filter
        node_region = nodes_gdf.iloc[i]['region']
        if selected_region and selected_region != 'Todos':
            if node_region != selected_region:
                continue

        cvli_val = float(pred_cvli[i])
        cvp_val = float(pred_cvp[i])

        cvli_status = "Crítico" if cvli_val >= cvli_threshold else "Normal"
        cvp_status = "Crítico" if cvp_val >= cvp_threshold else "Normal"

        results.append({
            'node_id': int(nodes_gdf.iloc[i]['node_id']),
            'faction': str(nodes_gdf.iloc[i]['faction']),
            'region': node_region,
            'cvli': {
                'value': float(cvli_val),
                'normalized': float((cvli_val / max_cvli) * 100),
                'status': cvli_status,
                'is_critical': bool(cvli_val >= cvli_threshold)
            },
            'cvp': {
                'value': float(cvp_val),
                'normalized': float((cvp_val / max_cvp) * 100),
                'status': cvp_status,
                'is_critical': bool(cvp_val >= cvp_threshold)
            }
        })
        
    return jsonify(results)

@app.route('/api/exogenous', methods=['GET', 'POST'])
def handle_exogenous():
    """Endpoint para inserir/listar dados exógenos."""
    if request.method == 'POST':
        data = request.json
        if not os.path.exists('data'):
            os.makedirs('data')

        existing_data = []
        if os.path.exists(EXOGENOUS_FILE):
            with open(EXOGENOUS_FILE, 'r') as f:
                try:
                    existing_data = json.load(f)
                except:
                    pass

        existing_data.append(data)

        with open(EXOGENOUS_FILE, 'w') as f:
            json.dump(existing_data, f, indent=4)

        return jsonify({"status": "success", "message": "Evento registrado."})

    else:
        if os.path.exists(EXOGENOUS_FILE):
            with open(EXOGENOUS_FILE, 'r') as f:
                try:
                    return jsonify(json.load(f))
                except:
                    return jsonify([])
        return jsonify([])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
