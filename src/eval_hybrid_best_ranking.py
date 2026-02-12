import os
import pickle
import numpy as np
import torch
from pathlib import Path

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
    features = np.zeros((num_nodes, 15))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(num_nodes):
            ts = ts_window[i, :]
            features[i, 0] = np.mean(ts)
            features[i, 1] = np.std(ts)
            features[i, 2] = np.max(ts)
            features[i, 3] = np.min(ts)
            features[i, 4] = np.sum(ts > 0) / len(ts) if len(ts) > 0 else 0
            features[i, 5] = np.sum(ts) / len(ts) if len(ts) > 0 else 0
            if len(ts) > 5:
                features[i, 6] = np.mean(ts[-5:]) - np.mean(ts[:5])
            if len(ts) > 1:
                features[i, 7] = np.mean(np.abs(np.diff(ts)))
            features[i, 8] = np.mean(ts[-3:]) if len(ts) >= 3 else 0
            features[i, 9] = np.mean(ts[-7:]) if len(ts) >= 7 else 0
            features[i, 10] = np.mean(ts[-14:]) if len(ts) >= 14 else 0
            if len(ts) > 1:
                mean_val = np.mean(ts)
                if mean_val > 1e-6:
                    features[i, 11] = np.std(ts) / mean_val
            features[i, 12] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0:
                features[i, 13] = (max_val - np.min(ts)) / max_val
    return np.nan_to_num(features)

def normalize(v):
    v = v.astype(float)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

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

    # Carregar scaler global (usado no grid search)
    scaler_path = RANKING_DIR / 'ranking_scaler.pkl'
    with open(scaler_path,'rb') as f: scaler = pickle.load(f)

    # Carregar modelos por dia com melhores hiperparâmetros
    results = []
    for d in range(7):
        model_path = RANKING_DIR / f"ranking_model_day{d}_selected.pth"
        if not model_path.exists():
            print(f"Dia {d}: modelo não encontrado, pulando...")
            continue
        params = BEST_PARAMS[d]
        class M(torch.nn.Module):
            def __init__(self, input_dim=15):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim,128), torch.nn.ReLU(), torch.nn.BatchNorm1d(128), torch.nn.Dropout(params['dropout']),
                    torch.nn.Linear(128,64), torch.nn.ReLU(), torch.nn.Linear(64,1)
                )
            def forward(self,x): return self.net(x)
        m = M().to(DEVICE)
        m.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
        m.eval()

        # Avaliar últimas janelas desse dia da semana
        HISTORY_WINDOW=30; HORIZON=7
        total_days = node_features.shape[1]
        valid_range = total_days - HISTORY_WINDOW - HORIZON + 1
        indices = [i for i in range(valid_range-30, valid_range) if dates[i+HISTORY_WINDOW].weekday()==d]
        p5s = []
        for s in indices:
            window = node_features[:, s:s+HISTORY_WINDOW, 0]
            feats = extract_features_v3(window)
            feats_in = scaler.transform(feats)
            with torch.no_grad():
                out = m(torch.FloatTensor(feats_in).to(DEVICE)).cpu().numpy().ravel()
            # ST-GCN (simulado: soma dos últimos 7 dias)
            stgcn_scores = np.sum(window[:,-7:], axis=1)
            stgcn_norm = normalize(stgcn_scores)
            rank_norm = normalize(out)
            comb = w_stgcn * stgcn_norm + w_rank * rank_norm
            # Target: soma dos próximos 7 dias
            future = node_features[:, s+HISTORY_WINDOW:s+HISTORY_WINDOW+HORIZON, 0]
            y_true = np.sum(future, axis=1)
            p5 = p_at_k(comb, y_true, k=5)
            if p5 is not None:
                p5s.append(p5)
        mean_p5 = float(np.mean(p5s)) if p5s else None
        print(f"Dia {d}: P@5 híbrido={mean_p5:.3f} em {len(p5s)} janelas")
        results.append({'day':d,'mean_p5':mean_p5,'n':len(p5s)})

    print("Resumo híbrido:")
    for r in results:
        print(f"Dia {r['day']}: P@5 híbrido={r['mean_p5']:.3f} ({r['n']} janelas)")

if __name__ == '__main__':
    main(num_windows=30, w_stgcn=0.6, w_rank=0.4)