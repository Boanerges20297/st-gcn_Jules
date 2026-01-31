#!/usr/bin/env python3
"""Treina/valida modelo LSTM para toda Fortaleza: treino 2023-2024, valida 2025-2026.

Gera por-bairro métricas (MSE, MAE, RMSE, R2), CSV com previsões por janela,
e plots PNG com média verdade vs previsão por bairro.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math

ROOT = Path(__file__).parents[2]
OUT_DIR = Path(__file__).parent

META = ROOT / 'data' / 'processed' / 'metadata_producao_v2.json'
NF = ROOT / 'data' / 'processed' / 'graph_data' / 'node_features.npy'
DATES_PKL = ROOT / 'data' / 'processed' / 'graph_data' / 'dates.pkl'
FAC = ROOT / 'data' / 'processed' / 'analise_movimentacao_faccoes.json'

# params
H = 120
P = 30
EPOCHS = 6
BATCH = 128
DEVICE = torch.device('cpu')

print('Loading metadata and node features...')
meta = json.load(open(META,'r',encoding='utf-8'))
bairros = meta.get('bairros_normalizados', [])

arr = np.load(NF, allow_pickle=True)
dates = None
if DATES_PKL.exists():
    import pickle
    dates = pickle.load(open(DATES_PKL,'rb'))
    # if arr is (T,N,F) transpose
    if arr.ndim==3 and arr.shape[0]==len(dates):
        arr = np.transpose(arr,(1,0,2))

N,T,F = arr.shape
print('node_features shape', arr.shape)

# select all bairros from metadata (assumes metadata indices align with arr)
name_map = {b:i for i,b in enumerate(bairros)}
idxs = [i for i in range(len(bairros))]
if not idxs:
    raise SystemExit('No bairros found in metadata')
dynamic = arr[idxs,:,:]

# factions one-hot
faction_data = json.load(open(FAC,'r',encoding='utf-8')) if FAC.exists() else {}
all_factions = set()
for v in faction_data.values():
    for f in v.get('facoes_envolvidas',[]):
        all_factions.add(f)
all_factions = sorted(list(all_factions))
K = len(all_factions)
fac_index = {f:i for i,f in enumerate(all_factions)}
faction_onehot = np.zeros((len(bairros), K), dtype=float)
for i,b in enumerate(bairros):
    info = faction_data.get(b, {})
    for f in info.get('facoes_envolvidas',[]):
        if f in fac_index:
            faction_onehot[i, fac_index[f]] = 1.0
faction_sel = faction_onehot[idxs]

# compute train/val start indices
if dates is None:
    raise SystemExit('dates.pkl required for year-slicing')
dates_list = list(dates)
train_start = next(i for i,d in enumerate(dates_list) if d.year >= 2023)
train_end = max(i for i,d in enumerate(dates_list) if d.year <= 2024)
val_start = next(i for i,d in enumerate(dates_list) if d.year >= 2025)
val_end = max(i for i,d in enumerate(dates_list) if d.year <= 2026)

train_starts = [i for i in range(train_start, train_end - H - P + 2)]
val_starts = [i for i in range(val_start, val_end - H - P + 2)]
print('train samples:', len(train_starts)*len(idxs), 'val samples:', len(val_starts)*len(idxs))

class TSMany(Dataset):
    def __init__(self, dyn, starts, history, horizon, faction=None, scaler=None):
        self.dyn = dyn
        self.starts = starts
        self.history = history
        self.horizon = horizon
        self.nodes = dyn.shape[0]
        self.faction = faction
        self.scaler = scaler
        self.samples = [(n,s) for s in starts for n in range(self.nodes)]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        n,s = self.samples[idx]
        x = self.dyn[n, s:s+self.history, :].astype(float)
        y = self.dyn[n, s+self.history:s+self.history+self.horizon, 0].astype(float)
        if self.scaler is not None:
            x = (x - self.scaler['mean'])/(self.scaler['std']+1e-9)
        if self.faction is not None:
            fvec = self.faction[n]
            fmat = np.tile(fvec.reshape(1,-1),(self.history,1))
            x = np.concatenate([x, fmat], axis=1)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# scaler from train
train_dyn = dynamic[:, train_start:train_end+1, :]
dyn_mean = np.mean(train_dyn.reshape(-1, train_dyn.shape[-1]), axis=0)
dyn_std = np.std(train_dyn.reshape(-1, train_dyn.shape[-1]), axis=0)
scaler = {'mean': dyn_mean, 'std': dyn_std}

train_ds = TSMany(dynamic, train_starts, H, P, faction=faction_sel if K>0 else None, scaler=scaler)
val_ds = TSMany(dynamic, val_starts, H, P, faction=faction_sel if K>0 else None, scaler=scaler)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, in_dim, hid=64, out=P):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, batch_first=True)
        self.fc = nn.Linear(hid, out)
    def forward(self,x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:])

in_dim = F + (K if K>0 else 0)
model = LSTMModel(in_dim=in_dim).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for ep in range(EPOCHS):
    model.train()
    total=0.0
    for xb,yb in train_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()*xb.size(0)
    print(f'Epoch {ep+1}/{EPOCHS} train_loss={total/len(train_ds):.6f}')

# Evaluate on validation set and collect per-bairro metrics
model.eval()
from collections import defaultdict
metrics = {}
pred_rows = []
with torch.no_grad():
    per_node_preds = defaultdict(list)
    per_node_trues = defaultdict(list)
    for idx in range(len(val_ds)):
        xb,yb = val_ds[idx]
        xb = xb.unsqueeze(0).to(DEVICE)
        pred = model(xb).cpu().numpy().ravel()
        true = yb.numpy().ravel()
        n,s = val_ds.samples[idx]
        per_node_preds[n].append(pred)
        per_node_trues[n].append(true)
        start_date = dates[s+H].strftime('%Y-%m-%d')
        pred_rows.append({'start_date':start_date,'bairro':bairros[n],'pred':pred.tolist(),'true':true.tolist()})

def compute_stats(y_true_arr, y_pred_arr):
    # arrays shape (S,P)
    y_true = np.vstack(y_true_arr) if len(y_true_arr)>0 else np.empty((0,P))
    y_pred = np.vstack(y_pred_arr) if len(y_pred_arr)>0 else np.empty((0,P))
    if y_true.size==0:
        return None
    mse = float(((y_true - y_pred)**2).mean())
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = math.sqrt(mse)
    # R2 per horizon flattened
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    r2 = float(1 - ss_res/(ss_tot+1e-9))
    return {'mse':mse,'mae':mae,'rmse':rmse,'r2':r2}

for n in range(len(idxs)):
    stats = compute_stats(per_node_trues[n], per_node_preds[n])
    if stats is None:
        continue
    metrics[bairros[n]] = stats
    # plot average
    avg_true = np.vstack(per_node_trues[n]).mean(axis=0)
    avg_pred = np.vstack(per_node_preds[n]).mean(axis=0)
    plt.figure(); x=np.arange(P)
    plt.plot(x, avg_true, label='true'); plt.plot(x, avg_pred, label='pred')
    plt.title(f'{bairros[n]}'); plt.legend(); plt.tight_layout()
    pdir = OUT_DIR / 'plots'
    pdir.mkdir(parents=True, exist_ok=True)
    # sanitize filename
    safe = ''.join(c if (c.isalnum() or c in (' ', '-', '_')) else '_' for c in bairros[n]).strip()
    safe = safe.replace(' ', '_')
    fname = pdir / f'{safe}_val_pred.png'
    plt.savefig(fname); plt.close()

# save report
report = {'params':{'history':H,'horizon':P,'epochs':EPOCHS,'batch':BATCH,'train_years':'2023-2024','val_years':'2025-2026'}, 'metrics':metrics}
with open(OUT_DIR / 'fortaleza_report.json','w',encoding='utf-8') as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)

# save CSV of predictions
import csv
with open(OUT_DIR / 'fortaleza_predictions_val.csv','w',newline='',encoding='utf-8') as fh:
    w = csv.writer(fh)
    header = ['start_date','bairro'] + [f'true_{i}' for i in range(P)] + [f'pred_{i}' for i in range(P)]
    w.writerow(header)
    for r in pred_rows:
        w.writerow([r['start_date'], r['bairro']] + r['true'] + r['pred'])

print('Saved report:', OUT_DIR / 'fortaleza_report.json')
print('Saved predictions CSV:', OUT_DIR / 'fortaleza_predictions_val.csv')
print('Saved plots to', OUT_DIR / 'plots')
