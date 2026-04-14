import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# Configurações do Benchmark
WINDOW = 30
EPOCHS = 10
FEATURES_A = 1 # Ocorrências
FEATURES_B = 1 # Apreensões (Energia)
TOTAL_FEATURES = FEATURES_A + FEATURES_B
P_TOP = 20
N_NODES = 121 # Valor padrão inicial
DROPOUT = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=1):
        super(SimpleGATLayer, self).__init__()
        self.heads = heads
        self.head_dim = out_features // heads
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, adj):
        # x: [B, N, F], adj: [N, N]
        b, n, f = x.size()
        h = self.lin(x).view(b, n, self.heads, self.head_dim)
        
        # Atenção otimizada: a(Wh_i, Wh_j) = a1*Wh_i + a2*Wh_j
        # Divide o parâmetro de atenção em dois vetores
        a1, a2 = self.attn.chunk(2, dim=-1)
        
        # Calcula scores de cada nó individualmente
        s1 = (h * a1).sum(dim=-1) # [B, N, heads]
        s2 = (h * a2).sum(dim=-1) # [B, N, heads]
        
        # Broadcast da soma para obter todos os pares (i, j)
        # s1.unsqueeze(2) é [B, N, 1, heads], s2.unsqueeze(1) é [B, 1, N, heads]
        e = s1.unsqueeze(2) + s2.unsqueeze(1) # [B, N, N, heads]
        e = F.leaky_relu(e)
        
        # Máscara de adjacência
        zero_vec = -9e15 * torch.ones_like(e)
        # adj deve ser [N, N], expandimos para [B, N, N, heads]
        mask = adj.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, self.heads)
        attention = torch.where(mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        
        # Agregação: [B, N, N, heads] * [B, N, heads, dim]
        h_prime = torch.einsum('bnnh,bnhd->bnhd', attention, h)
        return h_prime.reshape(b, n, -1)

class STGATPura(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGATPura, self).__init__()
        self.gat1 = SimpleGATLayer(in_channels, 64)
        self.dropout = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x, adj):
        b, n, t, f = x.size()
        spatial_out = []
        for i in range(t):
            h = self.gat1(x[:, :, i, :], adj)
            spatial_out.append(h)
        x = torch.stack(spatial_out, dim=2)
        x = x.view(b * n, t, -1)
        x = self.dropout(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0)).view(b, n)

class GCNPura(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNPura, self).__init__()
        self.lin1 = nn.Linear(in_channels * WINDOW, 128)
        self.lin2 = nn.Linear(128, 64)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x, adj):
        b, n, t, f = x.size()
        x = x.reshape(b, n, t * f)
        # GCN manual: A_hat * X * W
        # Normalização simples do adj
        d = torch.diag(adj.sum(1))
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
        
        x = F.relu(self.lin1(adj_norm @ x))
        x = F.relu(self.lin2(adj_norm @ x))
        return self.fc(x).view(b, n)

class LSTMPura(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSTMPura, self).__init__()
        self.lstm = nn.LSTM(in_channels, 64, batch_first=True)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x, adj=None):
        # Foco apenas no tempo por nó
        b, n, t, f = x.size()
        x = x.view(b * n, t, f)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n.squeeze(0))
        return out.view(b, n)

class TCNPura(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TCNPura, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x, adj=None):
        b, n, t, f = x.size()
        x = x.view(b * n, f, t) # [Batch*Nodes, Features, Time]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        out = self.fc(x)
        return out.view(b, n)

class TemporalTransformerPura(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalTransformerPura, self).__init__()
        self.emb = nn.Linear(in_channels, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x, adj=None):
        b, n, t, f = x.size()
        x = x.view(b * n, t, f)
        x = self.emb(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Global average pooling over time
        return self.fc(x).view(b, n)

class ChebNetPura(nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super(ChebNetPura, self).__init__()
        self.K = K
        self.lins = nn.ModuleList([nn.Linear(in_channels * WINDOW, 64) for _ in range(K + 1)])
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x, adj):
        b, n, t, f = x.size()
        x = x.reshape(b, n, -1)
        
        # L = I - D^-1/2 A D^-1/2 (Simplificado para o teste)
        d = torch.diag(adj.sum(1))
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        L = torch.eye(n).to(x.device) - d_inv_sqrt @ adj @ d_inv_sqrt
        
        # Polinômios de Chebyshev
        # T0 = X, T1 = LX
        res = self.lins[0](x)
        if self.K > 0:
            T0 = x
            T1 = L @ x
            res = res + self.lins[1](T1)
            for k in range(2, self.K + 1):
                T2 = 2 * L @ T1 - T0
                res = res + self.lins[k](T2)
                T0, T1 = T1, T2
        
        return self.fc(F.relu(res)).view(b, n)

class StaticBaseline(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StaticBaseline, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj=None):
        b, n, t, f = x.size()
        # Exponential smoothing manual
        weights = torch.exp(torch.arange(t).to(x.device) * torch.log(1 - torch.clamp(self.alpha, 0, 1)))
        weights = weights.flip(0).view(1, 1, t, 1)
        x_smooth = (x * weights).sum(dim=2) / weights.sum()
        return self.fc(x_smooth).view(b, n)

class GraphDiffusionPura(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphDiffusionPura, self).__init__()
        self.lin = nn.Linear(in_channels * WINDOW, 64)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x, adj):
        b, n, t, f = x.size()
        # Simula difusão simples via média de vizinhos
        x_flat = x.view(b, n, -1)
        # Propagação simplificada (1 passo)
        x_diff = adj @ x_flat
        x_diff = F.relu(self.lin(x_diff))
        out = self.fc(x_diff)
        return out.view(b, n).squeeze(-1)

def precision_at_k(preds, targets, k=20):
    # preds: [B, N], targets: [B, N]
    batch_p = []
    for i in range(preds.size(0)):
        _, top_idx = torch.topk(preds[i], k)
        correct = targets[i][top_idx].sum()
        batch_p.append(correct / (targets[i].sum() + 1e-6))
    return np.mean(batch_p)

def load_real_data():
    print("Carregando arquivos CSV...")
    path_a = r'c:\Users\STI01\Desktop\Projetos\Report Preview\data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    path_b = r'c:\Users\STI01\Desktop\Projetos\Report Preview\data\raw\ocorrencias_tropa.csv'
    
    # Arquivo A: Ocorrências
    df_a = pd.read_csv(path_a, low_memory=False)
    df_a = df_a.dropna(subset=['data', 'bairro'])
    df_a['data'] = pd.to_datetime(df_a['data']).dt.date
    
    # Arquivo B: Apreensões (Tropa)
    df_b = pd.read_csv(path_b, low_memory=False)
    # Tenta encontrar a coluna correta de data se data_registro falhar ou for esparsa
    date_col_b = 'data_registro' if 'data_registro' in df_b.columns else 'data'
    df_b = df_b.dropna(subset=[date_col_b])
    df_b['data_clean'] = pd.to_datetime(df_b[date_col_b]).dt.date
    
    # Mapeamento de Bairros (Foco Fortaleza conforme GEMINI.md)
    # Lista de bairros extraída do arquivo de centros latlong
    json_path = r'c:\Users\STI01\Desktop\Projetos\Report Preview\data\raw\bairros_centros_latlong.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        fortaleza_bairros = json.load(f).keys()
    
    # Filtra e conta para pegar os top N bairros
    active_bairros = df_a[df_a['bairro'].str.upper().isin(fortaleza_bairros)]['bairro'].value_counts().head(50).index.tolist()
    
    bairros = sorted(active_bairros)
    bairro_to_idx = {b: i for i, b in enumerate(bairros)}
    n_nodes = len(bairros)
    
    print(f"Benchmark limitado aos {n_nodes} bairros mais ativos de Fortaleza para eficiência local.")
    
    # Criar Timeseries Matrix [N, Dias]
    min_date = min(df_a['data'].min(), df_b['data_clean'].min())
    max_date = max(df_a['data'].max(), df_b['data_clean'].max())
    dias = (max_date - min_date).days + 1
    
    matrix_a = np.zeros((n_nodes, dias, FEATURES_A))
    matrix_b = np.zeros((n_nodes, dias, FEATURES_B))
    
    print(f"Processando {dias} dias para {n_nodes} bairros...")
    
    # Agregação Ocorrências (Simplificada: contagem por dia/bairro)
    counts_a = df_a.groupby(['data', 'bairro']).size().reset_index(name='count')
    for _, row in counts_a.iterrows():
        if row['bairro'] in bairro_to_idx:
            d_idx = (row['data'] - min_date).days
            b_idx = bairro_to_idx[row['bairro']]
            matrix_a[b_idx, d_idx, 0] = row['count']
            
    # Agregação Apreensões (Energia)
    counts_b = df_b.groupby(['data_clean', 'local_ocorrencia']).size().reset_index(name='count')
    for _, row in counts_b.iterrows():
        # Usamos local_ocorrencia como proxy para bairro se disponível, ou mapeamento
        # Para este benchmark, tentaremos casar nomes de bairros
        b_name = str(row['local_ocorrencia']).upper()
        if b_name in bairro_to_idx:
            d_idx = (row['data_clean'] - min_date).days
            b_idx = bairro_to_idx[b_name]
            matrix_b[b_idx, d_idx, 0] = row['count']

    # Criar Tensores [Batch, N, T, F]
    # Simplificação: pegamos os últimos WINDOW dias como feature e o dia seguinte como label
    X = []
    y = []
    
    for d in range(WINDOW, dias - 1):
        feat_a = matrix_a[:, d-WINDOW:d, :]
        feat_b = matrix_b[:, d-WINDOW:d, :]
        feat = np.concatenate([feat_a, feat_b], axis=-1)
        X.append(feat)
        
        target = (matrix_a[:, d, 0] > 0).astype(float) # Predição binária (ocorreu ou não)
        y.append(target)
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32), n_nodes

from torch.utils.data import TensorDataset, DataLoader

import argparse

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='Lista de modelos para rodar')
    args = parser.parse_args()

    print(f"--- INICIANDO BENCHMARK SENTINELA (Fase Otimizada) ---", flush=True)
    
    X_all, y_all, n_nodes_real = load_real_data()
    global N_NODES
    N_NODES = n_nodes_real
    
    # Split simples
    split = int(len(X_all) * 0.8)
    train_ds = TensorDataset(X_all[:split], y_all[:split])
    val_ds = TensorDataset(X_all[split:], y_all[split:])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    print(f"Dataset: {len(train_ds)} samples | Val: {len(val_ds)} samples", flush=True)
    
    # Matriz de adjacência densa
    adj = torch.zeros((n_nodes_real, n_nodes_real)).to(device)
    for i in range(n_nodes_real):
        adj[i, (i + 1) % n_nodes_real] = 1.0
        adj[(i + 1) % n_nodes_real, i] = 1.0
    adj += torch.eye(n_nodes_real).to(device)

    all_models = {
        "ST-GAT": STGATPura(TOTAL_FEATURES, 1),
        "GCN": GCNPura(TOTAL_FEATURES, 1),
        "LSTM": LSTMPura(TOTAL_FEATURES, 1),
        "TCN": TCNPura(TOTAL_FEATURES, 1),
        "Transformer": TemporalTransformerPura(TOTAL_FEATURES, 1),
        "ChebNet": ChebNetPura(TOTAL_FEATURES, 1, K=2),
        "Baseline": StaticBaseline(TOTAL_FEATURES, 1),
        "Diffusion": GraphDiffusionPura(TOTAL_FEATURES, 1)
    }

    if args.models:
        selected_models = {k: v for k, v in all_models.items() if k.lower() in [m.lower() for m in args.models]}
    else:
        selected_models = all_models

    results = {}

    for name, model in selected_models.items():
        try:
            print(f"\n> Treinando {name}...", flush=True)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Aumentado para Scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                                           steps_per_epoch=len(train_loader), 
                                                           epochs=EPOCHS)
            criterion = nn.BCEWithLogitsLoss()
            
            history = []
            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb, adj)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                history.append(avg_loss)
                print(f"  Época {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}", flush=True)
            
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    preds = torch.sigmoid(model(xb, adj))
                    all_preds.append(preds.cpu())
                    all_targets.append(yb)
            
            p20 = precision_at_k(torch.cat(all_preds), torch.cat(all_targets), k=P_TOP)
            results[name] = {"history": history, "p20": p20}
            print(f"  [RESULTADO] P@{P_TOP} (Val): {p20:.4f}", flush=True)
        except Exception as e:
            print(f"  [ERRO] Falha no modelo {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\n" + "="*40, flush=True)
    print("RESUMO FINAL DO BENCHMARK", flush=True)
    print("="*40, flush=True)
    for name, res in results.items():
        print(f"{name:15} | P@20: {res['p20']:.4f} | Loss Inicial: {res['history'][0]:.4f} -> Final: {res['history'][-1]:.4f}", flush=True)

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"ERRO FATAL NO SCRIPT: {e}", flush=True)
