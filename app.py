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

# Parâmetros de janela
WINDOW_CVLI = 180  # Atualizado para 180 dias
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
            # CVLI trained with in_channels=1, num_classes=1, window=180
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

def format_trend(prediction, history_avg):
    """
    Calcula a variação percentual e retorna string descritiva.
    """
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
        # ---------------------------
        # Previsão CVLI
        # ---------------------------
        out_cvli = np.zeros((node_features.shape[0], 1))
        hist_avg_cvli = np.zeros(node_features.shape[0])

        if model_cvli:
            if node_features.shape[1] >= WINDOW_CVLI:
                # Input for Model
                input_cvli = node_features[:, -WINDOW_CVLI:, 0:1]
                input_tensor = torch.FloatTensor(input_cvli).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_cvli(input_tensor, norm_adj)
                out_cvli = pred.squeeze(0).cpu().numpy()

                # Historical Average for Trend Calc (avg daily * horizon)
                # CVLI Horizon is 3 days (according to new requirement)
                # Input window is 180 days
                # Calculate avg per day in the window
                daily_avg = np.mean(input_cvli, axis=1) # (N, 1)
                hist_avg_cvli = daily_avg[:, 0] * 3 # Scale to 3-day horizon
            else:
                print("AVISO: Dados insuficientes para janela CVLI")

        # ---------------------------
        # Previsão CVP
        # ---------------------------
        out_cvp = np.zeros((node_features.shape[0], 1))
        hist_avg_cvp = np.zeros(node_features.shape[0])

        if model_cvp:
            if node_features.shape[1] >= WINDOW_CVP:
                input_cvp = node_features[:, -WINDOW_CVP:, 1:2]
                input_tensor = torch.FloatTensor(input_cvp).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_cvp(input_tensor, norm_adj)
                out_cvp = pred.squeeze(0).cpu().numpy()

                # Historical Average for Trend Calc
                # CVP Horizon is 1 day
                daily_avg = np.mean(input_cvp, axis=1) # (N, 1)
                hist_avg_cvp = daily_avg[:, 0] * 1
            else:
                print("AVISO: Dados insuficientes para janela CVP")

        # ---------------------------
        # Processamento e Normalização
        # ---------------------------

        # CVLI
        out_cvli = np.maximum(out_cvli, 0)
        cvli_raw = out_cvli[:, 0]
        cvli_adj = cvli_raw * 1.5 # Sensibilidade aumentada

        min_cvli = np.min(cvli_adj)
        shifted_cvli = cvli_adj - min_cvli
        max_shift_cvli = np.max(shifted_cvli) if np.max(shifted_cvli) > 0 else 1
        normalized_risk_cvli = (shifted_cvli / max_shift_cvli) * 100

        # CVP
        out_cvp = np.maximum(out_cvp, 0)
        cvp_raw = out_cvp[:, 0]

        min_cvp = np.min(cvp_raw)
        shifted_cvp = cvp_raw - min_cvp
        max_shift_cvp = np.max(shifted_cvp) if np.max(shifted_cvp) > 0 else 1
        normalized_risk_cvp = (shifted_cvp / max_shift_cvp) * 100

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

        # Percentil para flag de prioridade
        try:
            cutoff_cvli = float(np.percentile(normalized_risk_cvli, 90))
        except Exception:
            cutoff_cvli = 80.0

        for i in range(len(normalized_risk_cvli)):
            cvli_score = float(normalized_risk_cvli[i])
            cvp_score = float(normalized_risk_cvp[i])

            cvli_val = float(cvli_adj[i])
            cvp_val = float(cvp_raw[i])

            # Trend calculation
            trend_cvli = format_trend(cvli_raw[i], hist_avg_cvli[i])
            trend_cvp = format_trend(cvp_raw[i], hist_avg_cvp[i])

            reasons = []

            # Add descriptive reasons instead of raw numbers where possible
            if cvli_val > 0.01 or cvli_score > 5:
                # Use trend description
                reasons.append(f'CVLI: {trend_cvli}')

            if cvp_val > 0.01 or cvp_score > 5:
                reasons.append(f'CVP: {trend_cvp}')

            if len(reasons) == 0:
                reasons.append('Estabilidade (Baixo Risco)')

            if conn is not None and conn_mean > 0 and conn[i] > conn_mean * 1.5:
                reasons.append('Área com alta conectividade')

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

        # Meta info
        meta = {
            'window_cvli': WINDOW_CVLI,
            'window_cvp': WINDOW_CVP
        }
        if dates is not None and len(dates) > 0:
            last_date_obj = pd.to_datetime(dates[-1])
            meta['last_date'] = str(last_date_obj.date())
            # Calculate start dates based on windows
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

            # Legacy fields
            if 'start_cvli' in meta:
                meta['window_start'] = meta['start_cvli']
                meta['window_end'] = meta['last_date']

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        print(f"ERROR details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro ao calcular risco: {e}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
