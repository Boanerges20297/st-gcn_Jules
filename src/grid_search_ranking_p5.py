import os
import pickle
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / 'data' / 'processed' / 'processed_graph_data.pkl'
RANKING_DIR = ROOT / 'models' / 'ranking_by_day'
OUT_PATH = ROOT / 'outputs' / 'grid_search_ranking_p5.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train_and_eval(pos_weight, dropout, epochs, day, X_train, y_train, X_val, y_val):
    class M(torch.nn.Module):
        def __init__(self, input_dim=15):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim,128), torch.nn.ReLU(), torch.nn.BatchNorm1d(128), torch.nn.Dropout(dropout),
                torch.nn.Linear(128,64), torch.nn.ReLU(), torch.nn.Linear(64,1)
            )
        def forward(self,x): return self.net(x)
    m = M().to(DEVICE)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)
    for epoch in range(epochs):
        m.train()
        optimizer.zero_grad()
        out = m(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
    m.eval()
    with torch.no_grad():
        out_val = m(X_val_t)
        val_pred = torch.sigmoid(out_val).cpu().numpy()
        p5 = p_at_k(val_pred, y_val, k=5)
    return float(p5) if p5 is not None else None

def main():
    with open(DATA_FILE, 'rb') as f: data = pickle.load(f)
    node_features = data['node_features'][:, :, 0]
    total_days = node_features.shape[1]
    window = 30
    horizon = 7
    # Coletar janelas para todos os dias
    results = []
    for day in range(7):
        X_list, y_list = [], []
        for t in range(0, total_days - window - horizon, 2):
            if (t // 1) % 7 != day:
                continue
            window_data = node_features[:, t : t+window]
            feats = extract_features_v3(window_data)
            future_data = node_features[:, t+window : t+window+horizon]
            target = np.sum(future_data, axis=1)
            if target.max() > 0: target = target / target.max()
            X_list.append(feats)
            y_list.append(target)
        if not X_list:
            continue
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_all, test_size=0.2, random_state=42)
        # Grid search
        for pos_weight in [2, 5, 10]:
            for dropout in [0.1, 0.2, 0.3]:
                for epochs in [100, 200]:
                    p5 = train_and_eval(pos_weight, dropout, epochs, day, X_train, y_train, X_val, y_val)
                    results.append({'day':day,'pos_weight':pos_weight,'dropout':dropout,'epochs':epochs,'p5':p5})
                    print(f"Day {day} pos_weight={pos_weight} dropout={dropout} epochs={epochs} => P@5={p5}")
    # Salvar resultados
    with open(OUT_PATH,'w') as f:
        import json
        json.dump(results, f, indent=2)
    print(f"Resultados salvos em {OUT_PATH}")

if __name__ == '__main__':
    main()