def normalize_minmax(v):
    v = v.astype(float)
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-9:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)
def normalize_zscore(v):
    v = v.astype(float)
    mean = np.mean(v)
    std = np.std(v)
    if std < 1e-6:
        return np.zeros_like(v)
    return (v - mean) / std

import os
import pickle
import numpy as np
import torch
from pathlib import Path

# Importar função de extração de features de produção
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.train_ranking_final_production import extract_features_clean

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / 'data' / 'processed' / 'processed_graph_data.pkl'
RANKING_DIR = ROOT / 'models' / 'ranking_by_day'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Melhores hiperparâmetros por dia (extraídos do grid search)
BEST_PARAMS = {
    0: {'pos_weight': 10, 'dropout': 0.3, 'epochs': 200},
    1: {'pos_weight': 2, 'dropout': 0.1, 'epochs': 100},
    2: {'pos_weight': 2, 'dropout': 0.1, 'epochs': 100},
    3: {'pos_weight': 2, 'dropout': 0.1, 'epochs': 100},
    4: {'pos_weight': 2, 'dropout': 0.1, 'epochs': 100},
    5: {'pos_weight': 2, 'dropout': 0.1, 'epochs': 100},
    6: {'pos_weight': 2, 'dropout': 0.1, 'epochs': 100},
}

def extract_features_v3(ts_window):
    num_nodes = ts_window.shape[0]
    features = np.zeros((num_nodes, 20))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(num_nodes):
            ts = ts_window[i, :]
            features[i, 0] = np.mean(ts)
            features[i, 1] = np.std(ts)
            features[i, 2] = np.max(ts)
            features[i, 3] = np.min(ts)
            # Proporção de zeros
            features[i, 4] = np.sum(ts == 0) / len(ts) if len(ts) > 0 else 0
            # Proporção de valores > 0
            features[i, 5] = np.sum(ts > 0) / len(ts) if len(ts) > 0 else 0
            # Média início e fim
            split = len(ts) // 3
            features[i, 6] = np.mean(ts[:split]) if split > 0 else 0
            features[i, 7] = np.mean(ts[-split:]) if split > 0 else 0
            # Tendência (diferença média fim - início)
            features[i, 8] = features[i, 7] - features[i, 6]
            # Desvio absoluto médio
            if len(ts) > 1:
                features[i, 9] = np.mean(np.abs(np.diff(ts)))
            # Média últimos 3, 7, 14 dias
            features[i, 10] = np.mean(ts[-3:]) if len(ts) >= 3 else 0
            features[i, 11] = np.mean(ts[-7:]) if len(ts) >= 7 else 0
            features[i, 12] = np.mean(ts[-14:]) if len(ts) >= 14 else 0
            # Coeficiente de variação
            if len(ts) > 1:
                mean_val = np.mean(ts)
                if mean_val > 1e-6:
                    features[i, 13] = np.std(ts) / mean_val
            # Amplitude interquartil
            features[i, 14] = np.percentile(ts, 75) - np.percentile(ts, 25)
            # Pico (valor máximo - média)
            features[i, 15] = np.max(ts) - np.mean(ts)
            # Número de picos (> média + 1 desvio padrão)
            features[i, 16] = np.sum(ts > (np.mean(ts) + np.std(ts)))
            # Autocorrelação lag 1
            if len(ts) > 1:
                ts_mean = np.mean(ts)
                ts1 = ts[:-1] - ts_mean
                ts2 = ts[1:] - ts_mean
                denom = np.sqrt(np.sum(ts1 ** 2) * np.sum(ts2 ** 2))
                features[i, 17] = np.sum(ts1 * ts2) / denom if denom > 0 else 0
            # Skewness
            if len(ts) > 2:
                m = np.mean(ts)
                s = np.std(ts)
                if s > 1e-6:
                    features[i, 18] = np.mean(((ts - m) / s) ** 3)
            # Kurtosis
            if len(ts) > 3:
                m = np.mean(ts)
                s = np.std(ts)
                if s > 1e-6:
                    features[i, 19] = np.mean(((ts - m) / s) ** 4) - 3
    return np.nan_to_num(features)

def normalize(v):
    def normalize_minmax(v):
        v = v.astype(float)
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-9:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)
    def normalize_percentile(v):
        v = v.astype(float)
        n = len(v)
        percentiles = np.zeros_like(v)
        for i, val in enumerate(v):
            percentiles[i] = (v < val).sum() / n
        return percentiles
    v = v.astype(float)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

def normalize_percentile(v):
    def normalize_zscore(v):
        v = v.astype(float)
        mean = np.mean(v)
        std = np.std(v)
        if std < 1e-6:
            return np.zeros_like(v)
        return (v - mean) / std
    v = v.astype(float)
    n = len(v)
    percentiles = np.zeros_like(v)
    for i, val in enumerate(v):
        percentiles[i] = (v < val).sum() / n
    return percentiles

def p_at_k(pred, true, k=5):
    pred = np.array(pred).ravel()
    true = np.array(true).ravel()
    if true.max() == 0:
        return None
    k_actual = min(k, int((true>0).sum()), len(true))
    if k_actual<=0: return None
    pred_top = np.argsort(-pred)[:k_actual]
    true_top = np.argsort(-true)[:k_actual]
    return len(set(pred_top)&set(true_top))/k_actual

def main(num_windows=30, w_stgcn=0.6, w_rank=0.4):
    with open(DATA_FILE, 'rb') as f: data = pickle.load(f)
    node_features = data['node_features']
    dates = data.get('dates')

    # Carregar scalers por dia (compatível com produção)
    scalers_path = RANKING_DIR / 'scalers.pkl'
    with open(scalers_path, 'rb') as f:
        scalers_by_day = pickle.load(f)

    # Carregar modelos por dia com melhores hiperparâmetros
    results = []
    for d in range(7):
        # --- Avaliação ST-GCN puro ---
        HISTORY_WINDOW=30; HORIZON=7
        total_days = node_features.shape[1]
        valid_range = total_days - HISTORY_WINDOW - HORIZON + 1
        indices = [i for i in range(valid_range-30, valid_range) if dates[i+HISTORY_WINDOW].weekday()==d]
        p5s_stgcn = []
        for s in indices:
            window = node_features[:, s:s+HISTORY_WINDOW, 0]
            stgcn_scores = np.sum(window[:,-7:], axis=1)
            # Usar normalização percentil para manter comparação
            stgcn_norm = normalize_percentile(stgcn_scores)
            future = node_features[:, s+HISTORY_WINDOW:s+HISTORY_WINDOW+HORIZON, 0]
            y_true = np.sum(future, axis=1)
            p5 = p_at_k(stgcn_norm, y_true, k=5)
            if p5 is not None:
                p5s_stgcn.append(p5)
        mean_p5_stgcn = float(np.mean(p5s_stgcn)) if p5s_stgcn else None
        print(f"Dia {d}: P@5 ST-GCN puro={mean_p5_stgcn:.3f} em {len(p5s_stgcn)} janelas")
        # --- Normalização min-max ---
        p5s_minmax = []
        model_path = RANKING_DIR / f"ranking_model_day{d}.pth"
        if not model_path.exists():
            print(f"Dia {d}: modelo não encontrado, pulando...")
            continue
        scaler = scalers_by_day[d]
        class M(torch.nn.Module):
            def __init__(self, input_dim=25):
                super().__init__()
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 64),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(32),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(32, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                )
            def forward(self, x):
                return self.fc(x).squeeze()
        m = M(input_dim=25).to(DEVICE)
        state = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and 'model_state' in state:
            converted_state = {}
            for key, value in state['model_state'].items():
                new_key = key.replace('net.', 'fc.')
                converted_state[new_key] = value
            m.load_state_dict(converted_state)
        else:
            converted_state = {}
            for key, value in state.items():
                new_key = key.replace('net.', 'fc.')
                converted_state[new_key] = value
            m.load_state_dict(converted_state)
        m.eval()
        HISTORY_WINDOW=30; HORIZON=7
        total_days = node_features.shape[1]
        valid_range = total_days - HISTORY_WINDOW - HORIZON + 1
        indices = [i for i in range(valid_range-30, valid_range) if dates[i+HISTORY_WINDOW].weekday()==d]
        for s in indices:
            window = node_features[:, s:s+HISTORY_WINDOW, 0]
            feats = extract_features_clean(window)
            feats_in = scaler.transform(feats)
            with torch.no_grad():
                out = m(torch.FloatTensor(feats_in).to(DEVICE)).cpu().numpy().ravel()
            stgcn_scores = np.sum(window[:,-7:], axis=1)
            stgcn_norm = normalize_minmax(stgcn_scores)
            rank_norm = normalize_minmax(out)
            print("ST-GCN norm (min-max):", stgcn_norm)
            print("Ranking norm (min-max):", rank_norm)
            comb = w_stgcn * stgcn_norm + w_rank * rank_norm
            future = node_features[:, s+HISTORY_WINDOW:s+HISTORY_WINDOW+HORIZON, 0]
            y_true = np.sum(future, axis=1)
            p5 = p_at_k(comb, y_true, k=5)
            if p5 is not None:
                p5s_minmax.append(p5)
        mean_p5_minmax = float(np.mean(p5s_minmax)) if p5s_minmax else None
        print(f"Dia {d}: P@5 híbrido (min-max)={mean_p5_minmax:.3f} em {len(p5s_minmax)} janelas")
        model_path = RANKING_DIR / f"ranking_model_day{d}.pth"
        if not model_path.exists():
            print(f"Dia {d}: modelo não encontrado, pulando...")
            continue
        scaler = scalers_by_day[d]
        class M(torch.nn.Module):
            def __init__(self, input_dim=25):
                super().__init__()
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 64),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(32),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(32, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                )
            def forward(self, x):
                return self.fc(x).squeeze()
        m = M(input_dim=25).to(DEVICE)
        state = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and 'model_state' in state:
            # Converter chaves 'net.' para 'fc.' para compatibilidade
            converted_state = {}
            for key, value in state['model_state'].items():
                new_key = key.replace('net.', 'fc.')
                converted_state[new_key] = value
            m.load_state_dict(converted_state)
        else:
            # Caso antigo, tentar converter também
            converted_state = {}
            for key, value in state.items():
                new_key = key.replace('net.', 'fc.')
                converted_state[new_key] = value
            m.load_state_dict(converted_state)
        m.eval()

        # Avaliar últimas janelas desse dia da semana
        HISTORY_WINDOW=30; HORIZON=7
        total_days = node_features.shape[1]
        valid_range = total_days - HISTORY_WINDOW - HORIZON + 1
        indices = [i for i in range(valid_range-30, valid_range) if dates[i+HISTORY_WINDOW].weekday()==d]
        p5s = []
        for s in indices:
            window = node_features[:, s:s+HISTORY_WINDOW, 0]
            feats = extract_features_clean(window)
            feats_in = scaler.transform(feats)
            with torch.no_grad():
                out = m(torch.FloatTensor(feats_in).to(DEVICE)).cpu().numpy().ravel()
            # ST-GCN (simulado: soma dos últimos 7 dias)
            stgcn_scores = np.sum(window[:,-7:], axis=1)
            # Normalização por percentil
            stgcn_norm = normalize_percentile(stgcn_scores)
            rank_norm = normalize_percentile(out)
            print("ST-GCN norm (percentil):", stgcn_norm)
            print("Ranking norm (percentil):", rank_norm)
            comb = w_stgcn * stgcn_norm + w_rank * rank_norm
            # Target: soma dos próximos 7 dias
            future = node_features[:, s+HISTORY_WINDOW:s+HISTORY_WINDOW+HORIZON, 0]
            y_true = np.sum(future, axis=1)
            p5 = p_at_k(comb, y_true, k=5)
            if p5 is not None:
                p5s.append(p5)
        mean_p5 = float(np.mean(p5s)) if p5s else None
        print(f"Dia {d}: P@5 híbrido (percentil)={mean_p5:.3f} em {len(p5s)} janelas")
        results.append({'day':d,'mean_p5':mean_p5,'n':len(p5s)})

    print("Resumo híbrido:")
    for r in results:
        print(f"Dia {r['day']}: P@5 híbrido={r['mean_p5']:.3f} ({r['n']} janelas)")

if __name__ == '__main__':
    main(num_windows=30, w_stgcn=0.6, w_rank=0.4)