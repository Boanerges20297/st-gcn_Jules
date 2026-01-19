from flask import Flask, jsonify, render_template, request
import pandas as pd
import geopandas as gpd
import pickle
import torch
import numpy as np
import os
from src.model import STGCN

app = Flask(__name__)

# Configuração e Carregamento de Dados
DATA_FILE = 'data/processed_graph_data.pkl'
MODEL_PATH = 'models/stgcn_model.pth'

# Verificar se os dados existem antes de carregar
if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_PATH):
    print("AVISO: Arquivos de dados ou modelo não encontrados. Execute o treinamento primeiro.")
else:
    print("Carregando dados para API...")
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)

    nodes_gdf = data_pack['nodes_gdf']
    adj_matrix = data_pack['adj_matrix']
    node_features = data_pack['node_features']

    # Carregar Modelo
    print("Carregando modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]
    history_window = 7

    model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=history_window)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Pre-computar normalização da adjacência
    adj_tensor = torch.FloatTensor(adj_matrix)
    rowsum = adj_tensor.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/polygons')
def get_polygons():
    """Retorna os polígonos em GeoJSON."""
    return nodes_gdf.to_json()

@app.route('/api/risk')
def get_risk():
    """Calcula e retorna o risco previsto para o próximo dia."""
    
    # Pegar os últimos 'history_window' dias de dados
    recent_features = node_features[:, -history_window:, :] # (N, T, F)
    
    # Preparar input para o modelo
    input_tensor = torch.FloatTensor(recent_features).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor, norm_adj) # (1, N, 2)
        
    prediction = prediction.squeeze(0).cpu().numpy() # (N, 2) -> [CVLI_pred, CVP_pred]
    
    # Calcular Índice de Risco Ponderado
    # Risco = (CVLI * 5) + (CVP * 1)
    prediction = np.maximum(prediction, 0)
    risk_score = (prediction[:, 0] * 5.0) + (prediction[:, 1] * 1.0)
    
    # Normalizar Risco
    max_risk = np.max(risk_score) if np.max(risk_score) > 0 else 1
    normalized_risk = (risk_score / max_risk) * 100
    
    # Retornar lista
    results = []
    for i, score in enumerate(normalized_risk):
        results.append({
            'node_id': i,
            'risk_score': float(score),
            'cvli_pred': float(prediction[i, 0]),
            'cvp_pred': float(prediction[i, 1]),
            'faction': nodes_gdf.iloc[i]['faction']
        })
        
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
