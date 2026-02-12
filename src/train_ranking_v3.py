import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
OUTPUT_MODEL_DIR = os.path.join(ROOT, 'models', 'ranking_by_day')
SCALER_PATH = os.path.join(OUTPUT_MODEL_DIR, 'ranking_scaler.pkl') # ARQUIVO CRÍTICO
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RankingModelV3(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128, dropout=0.3):
        super(RankingModelV3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

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
                features[i, 11] = np.mean(np.abs(np.diff(ts)))
                mean_val = np.mean(ts)
                if mean_val > 1e-6:
                    features[i, 12] = np.std(ts) / mean_val
            
            features[i, 13] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0: 
                features[i, 14] = (max_val - np.min(ts)) / max_val

    return np.nan_to_num(features)

def train_ranking_v3():
    print(f"🚀 Iniciando Treinamento do Ranking V3 (COM SCALER)...")
    
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    node_features = data['node_features'][:, :, 0] 
    total_days = node_features.shape[1]
    
    X_list = []
    y_list = []
    window = 30
    horizon = 7
    
    for t in range(0, total_days - window - horizon, 2):
        window_data = node_features[:, t : t+window]
        feats = extract_features_v3(window_data)
        future_data = node_features[:, t+window : t+window+horizon]
        target = np.sum(future_data, axis=1)
        if target.max() > 0: target = target / target.max()
        X_list.append(feats)
        y_list.append(target)
        
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    
    # --- AQUI ESTAVA O ERRO: Precisamos salvar esse scaler ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    # Salvar o Scaler para usar na validação
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler salvo em: {SCALER_PATH}")
    # ---------------------------------------------------------
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_all, test_size=0.2, random_state=42)
    
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)
    
    # Configurações por env
    EPOCHS = int(os.environ.get('RANKING_EPOCHS', 150))
    LR = float(os.environ.get('RANKING_LR', 0.001))
    DROPOUT = float(os.environ.get('RANKING_DROPOUT', 0.2))
    LOSS_TYPE = os.environ.get('RANKING_LOSS', 'mse') # 'mse' ou 'bce'
    POS_WEIGHT = float(os.environ.get('RANKING_POS_WEIGHT', 2.0))

    model = RankingModelV3(dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if LOSS_TYPE == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE))
        print(f"Usando BCEWithLogitsLoss com pos_weight={POS_WEIGHT}")
    else:
        criterion = nn.MSELoss()
        print("Usando MSELoss")

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

    print(f"🏋️ Treinando... EPOCHS={EPOCHS} LR={LR} DROPOUT={DROPOUT} LOSS={LOSS_TYPE}")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        if LOSS_TYPE == 'bce':
            loss = criterion(output, y_train_t)
        else:
            loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

        # Avaliação
        if (epoch+1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                out_val = model(X_val_t)
                if LOSS_TYPE == 'bce':
                    val_pred = torch.sigmoid(out_val).cpu().numpy()
                else:
                    val_pred = out_val.cpu().numpy()
                p5 = p_at_k(val_pred, y_val, k=5)
                val_loss = criterion(out_val, y_val_t).item()
                print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f} P@5={p5}")

    for day in range(7):
        path = os.path.join(OUTPUT_MODEL_DIR, f"ranking_model_day{day}_selected.pth")
        torch.save(model.state_dict(), path)

    print(f"✅ Modelos salvos em {OUTPUT_MODEL_DIR}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    train_ranking_v3()