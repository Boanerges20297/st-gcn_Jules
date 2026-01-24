from flask import Flask, jsonify, render_template, request
import pandas as pd
import geopandas as gpd
import pickle
import torch
import numpy as np
import os
from src.model import STGCN

app = Flask(__name__)

# Configuração e Carregamento de Dados (usar caminhos absolutos relativos a este arquivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_model.pth')

# Valores padrão caso arquivos não estejam presentes — evita NameError nas rotas
nodes_gdf = None
adj_matrix = None
node_features = None
model = None
device = None
norm_adj = None
history_window = 7
dates = None

try:
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_PATH):
        print("AVISO: Arquivos de dados ou modelo não encontrados. Execute o treinamento primeiro.")
    else:
        print("Carregando dados para API...")
        with open(DATA_FILE, 'rb') as f:
            data_pack = pickle.load(f)

        nodes_gdf = data_pack.get('nodes_gdf')
        adj_matrix = data_pack.get('adj_matrix')
        node_features = data_pack.get('node_features')
        dates = data_pack.get('dates')

        # Carregar Modelo
        print("Carregando modelo...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = node_features.shape[0]
        num_features = node_features.shape[2]

        # Carregar state_dict primeiro para inspecionar a arquitetura treinada
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Tentar inferir history_window do tamanho do kernel da camada conv_final
        # conv_final.weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        # Onde kernel_w é o time_steps (history_window)
        if 'conv_final.weight' in state_dict:
            history_window = state_dict['conv_final.weight'].shape[3]
            print(f"Detectado history_window={history_window} do arquivo de modelo.")

        model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=history_window)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Pre-computar normalização da adjacência
        adj_tensor = torch.FloatTensor(adj_matrix)
        rowsum = adj_tensor.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)
except Exception as e:
    print(f"Erro ao carregar dados/modelo: {e}")
    nodes_gdf = None
    adj_matrix = None
    node_features = None
    model = None
    device = None
    norm_adj = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/polygons')
def get_polygons():
    """Retorna os polígonos em GeoJSON."""
    if nodes_gdf is None:
        return jsonify({'error': 'Dados de polígonos não carregados. Execute o treinamento ou verifique DATA_FILE.'}), 503
    try:
        return nodes_gdf.to_json()
    except Exception as e:
        return jsonify({'error': f'Erro ao serializar polígonos: {e}'}), 500

@app.route('/api/risk')
def get_risk():
    """Calcula e retorna o risco previsto para o próximo dia."""
    if model is None or node_features is None or norm_adj is None or nodes_gdf is None:
        return jsonify({'error': 'Modelo ou dados não carregados. Verifique DATA_FILE e MODEL_PATH.'}), 503

    try:
        # aceitar query param 'window' para sobrescrever a janela de histórico usada
        req_window = request.args.get('window', default=history_window, type=int)
        window = max(1, int(req_window))
        total_timesteps = node_features.shape[1]
        if window > total_timesteps:
            window = total_timesteps

        # Pegar os últimos 'window' dias de dados
        recent_features = node_features[:, -window:, :]  # (N, T, F)

        # Preparar input para o modelo
        input_tensor = torch.FloatTensor(recent_features).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(input_tensor, norm_adj)  # (1, N, 2)

        prediction = prediction.squeeze(0).cpu().numpy()  # (N, 2) -> [CVLI_pred, CVP_pred]

        # Calcular Índice de Risco baseado APENAS em CVLI
        prediction = np.maximum(prediction, 0)
        cvli_raw = prediction[:, 0]
        # Aumentar sensibilidade de CVLI em 20%
        cvli_adj = cvli_raw * 1.2
        # Usar apenas CVLI ajustado como risco bruto
        risk_score = cvli_adj

        # Normalizar Risco: shift para positivo antes de normalizar (lida com previsões negativas)
        min_risk = np.min(risk_score)
        shifted = risk_score - min_risk
        max_shift = np.max(shifted) if np.max(shifted) > 0 else 1
        normalized_risk = (shifted / max_shift) * 100

        # Construir razões de risco e sinalizadores de prioridade
        results = []
        # conectividade média para identificar áreas muito conectadas
        try:
            conn = np.array(adj_matrix).sum(axis=1)
            conn_mean = float(np.mean(conn))
        except Exception:
            conn = None
            conn_mean = 0

        # prioridade baseada em percentil (top 5%) para destacar territórios críticos
        try:
            cutoff = float(np.percentile(normalized_risk, 95))
        except Exception:
            cutoff = 90.0

        for i, score in enumerate(normalized_risk):
            cvli = float(cvli_adj[i])
            reasons = []
            if cvli > 0 or score > 0:
                reasons.append(f'Previsão CVLI (ajustada +20%): {cvli:.3f}')
            else:
                reasons.append('CVLI ausente ou insignificante')
            if conn is not None and conn_mean > 0 and conn[i] > conn_mean * 1.5:
                reasons.append('Área com alta conectividade (vizinhança densa)')

            results.append({
                'node_id': int(i),
                'risk_score': float(score),
                'cvli_pred': cvli,
                'faction': nodes_gdf.iloc[i].get('faction') if 'faction' in nodes_gdf.columns else None,
                'reasons': reasons,
                'priority_cvli': bool(score >= cutoff)
            })

        # meta info: janela usada
        meta = {}
        try:
            if dates is not None and len(dates) >= window:
                window_end = pd.to_datetime(dates[-1])
                window_start = pd.to_datetime(dates[-window])
                meta = {
                    'window_size': window,
                    'window_start': str(window_start.date()),
                    'window_end': str(window_end.date())
                }
        except Exception:
            meta = {'window_size': window}

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        return jsonify({'error': f'Erro ao calcular risco: {e}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
