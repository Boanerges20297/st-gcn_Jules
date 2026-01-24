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
MODEL_CVLI_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvli.pth')
MODEL_CVP_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvp.pth')

# Valores padrão caso arquivos não estejam presentes
nodes_gdf = None
adj_matrix = None
node_features = None
model_cvli = None
model_cvp = None
device = None
norm_adj = None
dates = None

# Parâmetros de janela padrão para cada modelo
WINDOW_CVLI = 30
WINDOW_CVP = 15

def load_data_and_models():
    global nodes_gdf, adj_matrix, node_features, model_cvli, model_cvp, device, norm_adj, dates

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = node_features.shape[0]
        # features originals (2: cvli, cvp)
        
        # Pre-computar normalização da adjacência
        adj_tensor = torch.FloatTensor(adj_matrix)
        rowsum = adj_tensor.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)

        # Carregar Modelo CVLI
        if os.path.exists(MODEL_CVLI_PATH):
            print(f"Carregando modelo CVLI de {MODEL_CVLI_PATH}...")
            # CVLI trained with in_channels=1, num_classes=1, window=30
            m_cvli = STGCN(num_nodes=num_nodes, in_channels=1, time_steps=WINDOW_CVLI, num_classes=1)
            state_dict = torch.load(MODEL_CVLI_PATH, map_location=device)
            m_cvli.load_state_dict(state_dict)
            m_cvli.to(device)
            m_cvli.eval()
            model_cvli = m_cvli
        else:
            print(f"AVISO: Modelo CVLI não encontrado em {MODEL_CVLI_PATH}")

        # Carregar Modelo CVP
        if os.path.exists(MODEL_CVP_PATH):
            print(f"Carregando modelo CVP de {MODEL_CVP_PATH}...")
            # CVP trained with in_channels=1, num_classes=1, window=15
            m_cvp = STGCN(num_nodes=num_nodes, in_channels=1, time_steps=WINDOW_CVP, num_classes=1)
            state_dict = torch.load(MODEL_CVP_PATH, map_location=device)
            m_cvp.load_state_dict(state_dict)
            m_cvp.to(device)
            m_cvp.eval()
            model_cvp = m_cvp
        else:
            print(f"AVISO: Modelo CVP não encontrado em {MODEL_CVP_PATH}")

    except Exception as e:
        print(f"Erro ao carregar dados/modelos: {e}")

# Executa carregamento inicial
load_data_and_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/polygons')
def get_polygons():
    """Retorna os polígonos em GeoJSON."""
    if nodes_gdf is None:
        return jsonify({'error': 'Dados de polígonos não carregados.'}), 503
    try:
        return nodes_gdf.to_json()
    except Exception as e:
        return jsonify({'error': f'Erro ao serializar polígonos: {e}'}), 500

@app.route('/api/risk')
def get_risk():
    """Calcula e retorna o risco previsto para o próximo dia usando modelos separados."""
    if node_features is None or norm_adj is None or nodes_gdf is None:
        return jsonify({'error': 'Dados não carregados.'}), 503

    try:
        # Previsão CVLI
        out_cvli = np.zeros((node_features.shape[0], 1))
        if model_cvli:
            # Pegar últimos 30 dias da feature 0 (CVLI)
            # node_features shape: (N, T, F) -> queremos (N, 30, 1)
            if node_features.shape[1] >= WINDOW_CVLI:
                input_cvli = node_features[:, -WINDOW_CVLI:, 0:1]
                input_tensor = torch.FloatTensor(input_cvli).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_cvli(input_tensor, norm_adj) # (1, N, 1)
                out_cvli = pred.squeeze(0).cpu().numpy()
            else:
                print("AVISO: Dados insuficientes para janela CVLI")

        # Previsão CVP
        out_cvp = np.zeros((node_features.shape[0], 1))
        if model_cvp:
            # Pegar últimos 15 dias da feature 1 (CVP)
            if node_features.shape[1] >= WINDOW_CVP:
                input_cvp = node_features[:, -WINDOW_CVP:, 1:2]
                input_tensor = torch.FloatTensor(input_cvp).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_cvp(input_tensor, norm_adj) # (1, N, 1)
                out_cvp = pred.squeeze(0).cpu().numpy()
            else:
                print("AVISO: Dados insuficientes para janela CVP")

        # Combinar: Col 0 = CVLI, Col 1 = CVP
        prediction = np.concatenate([out_cvli, out_cvp], axis=1) # (N, 2)

        # Calcular Índice de Risco baseado APENAS em CVLI (Lógica anterior mantida/adaptada)
        prediction = np.maximum(prediction, 0)
        cvli_raw = prediction[:, 0]

        # Aumentar sensibilidade de CVLI em 20%
        cvli_adj = cvli_raw * 1.2
        # Usar apenas CVLI ajustado como risco bruto
        risk_score = cvli_adj

        # Normalizar Risco
        min_risk = np.min(risk_score)
        shifted = risk_score - min_risk
        max_shift = np.max(shifted) if np.max(shifted) > 0 else 1
        normalized_risk = (shifted / max_shift) * 100

        # Construir resposta
        results = []
        # Conectividade
        try:
            conn = np.array(adj_matrix).sum(axis=1)
            conn_mean = float(np.mean(conn))
        except Exception:
            conn = None
            conn_mean = 0

        # Percentil
        try:
            cutoff = float(np.percentile(normalized_risk, 95))
        except Exception:
            cutoff = 90.0

        for i, score in enumerate(normalized_risk):
            cvli = float(cvli_adj[i])
            cvp_val = float(prediction[i, 1])
            reasons = []
            if cvli > 0 or score > 0:
                reasons.append(f'Previsão CVLI (ajustada +20%): {cvli:.3f}')
            if cvp_val > 0:
                 pass # Opcional: Adicionar razão sobre CVP se desejar

            if cvli == 0 and score == 0:
                reasons.append('CVLI ausente ou insignificante')

            if conn is not None and conn_mean > 0 and conn[i] > conn_mean * 1.5:
                reasons.append('Área com alta conectividade (vizinhança densa)')

            results.append({
                'node_id': int(i),
                'risk_score': float(score),
                'cvli_pred': cvli,
                # Adicionando CVP pred apenas se útil, mas dashboard usa o que retorna
                'cvp_pred': cvp_val,
                'faction': nodes_gdf.iloc[i].get('faction') if 'faction' in nodes_gdf.columns else None,
                'reasons': reasons,
                'priority_cvli': bool(score >= cutoff)
            })

        # Meta info
        meta = {
            'window_cvli': WINDOW_CVLI,
            'window_cvp': WINDOW_CVP
        }
        if dates is not None and len(dates) > 0:
            last_date_obj = pd.to_datetime(dates[-1])
            meta['last_date'] = str(last_date_obj.date())
            # For backward compatibility with UI that expects a window_start (using CVLI window as reference)
            if len(dates) >= WINDOW_CVLI:
                start_date_obj = pd.to_datetime(dates[-WINDOW_CVLI])
                meta['window_start'] = str(start_date_obj.date())
                meta['window_end'] = str(last_date_obj.date())

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        print(f"ERROR details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro ao calcular risco: {e}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
