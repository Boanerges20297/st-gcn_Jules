#!/usr/bin/env python3
"""Treino LSTM para CVLI com facções one-hot, normalização e plots/CSV de saída."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
CONF = Path(__file__).parent / 'config.json'

with CONF.open('r', encoding='utf-8') as f:
    cfg = json.load(f)

NEI_NAMES = [n.upper() for n in cfg['neighborhoods']]
EPOCHS = int(cfg.get('epochs',5))
H = int(cfg.get('history_window',120))
P = int(cfg.get('predict_horizon',30))
DEVICE = torch.device(cfg.get('device','cpu'))

META = ROOT / 'data' / 'processed' / 'metadata_producao_v2.json'
NF = ROOT / 'data' / 'processed' / 'graph_data' / 'node_features.npy'
DATES_PKL = ROOT / 'data' / 'processed' / 'graph_data' / 'dates.pkl'
FAC = ROOT / 'data' / 'processed' / 'analise_movimentacao_faccoes.json'

print('Loading metadata and node features...')
meta = json.load(open(META, 'r', encoding='utf-8'))
bairros = meta.get('bairros_normalizados', [])
name_map = {b.upper(): i for i,b in enumerate(bairros)}

arr = np.load(NF, allow_pickle=True)
# handle orientation using dates
dates = None
if DATES_PKL.exists():
    import pickle
    dates = pickle.load(open(DATES_PKL,'rb'))
if arr.ndim==3 and dates is not None and arr.shape[0] == len(dates):
    arr = np.transpose(arr, (1,0,2))

N,T,F = arr.shape
print('node_features shape', arr.shape)

# pick neighborhoods
idxs = []
for n in NEI_NAMES:
    if n in name_map:
        idxs.append(name_map[n])
    else:
        print('Warning: neighborhood not found in metadata:', n)
if not idxs:
    raise SystemExit('No neighborhoods found to train on')
print('Using neighborhood indices:', idxs)

# dynamic features (use all temporal features available, e.g., CVLI and CVP)
dynamic = arr[idxs, :, :]
# dynamic shape (nodes, T, F)

# build faction one-hot
faction_data = json.load(open(FAC, 'r', encoding='utf-8')) if FAC.exists() else {}
all_factions = set()
for v in faction_data.values():
    for f in v.get('facoes_envolvidas', []):
        all_factions.add(f)
all_factions = sorted(list(all_factions))
fac_index = {f:i for i,f in enumerate(all_factions)}
K = len(all_factions)
faction_onehot = np.zeros((len(bairros), K), dtype=float)
for i,b in enumerate(bairros):
    info = faction_data.get(b, {})
    for f in info.get('facoes_envolvidas', []):
        if f in fac_index:
            faction_onehot[i, fac_index[f]] = 1.0
faction_sel = faction_onehot[idxs]

# prepare train/test ranges
if dates is not None:
    dates_list = list(dates)
    s_year = int(cfg.get('start_year', 2022))
    e_year = int(cfg.get('end_year', 2025))
    train_start = next(i for i,d in enumerate(dates_list) if d.year >= s_year)
    train_end = max(i for i,d in enumerate(dates_list) if d.year <= e_year)
    test_start = next((i for i,d in enumerate(dates_list) if d.year >= 2026), None)
else:
    train_start, train_end, test_start = 0, T-1, None

train_idxs = list(range(train_start, train_end - H - P + 2))
test_idxs = []
if test_start is not None:
    test_idxs = [i for i in range(test_start, T - H - P + 1)]

print(f'train range indices: {train_start}..{train_end}, samples {len(train_idxs)}')
print(f'test start index: {test_start}, samples {len(test_idxs)}')

# Build datasets
class TSMany(Dataset):
    def __init__(self, dyn, starts, history, horizon, faction=None, scaler=None):
        # dyn: (nodes, T, F)
        self.dyn = dyn
        self.starts = starts
        self.history = history
        self.horizon = horizon
        self.nodes = dyn.shape[0]
        self.faction = faction
        self.scaler = scaler
        self.samples = []
        for s in starts:
            for n in range(self.nodes):
                self.samples.append((n,s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n,s = self.samples[idx]
        x = self.dyn[n, s:s+self.history, :].astype(float)  # (H,F)
        y = self.dyn[n, s+self.history:s+self.history+self.horizon, 0].astype(float)  # predict CVLI only
        # scale dynamic features
        if self.scaler is not None:
            x = (x - self.scaler['mean']) / (self.scaler['std'] + 1e-9)
        # append faction one-hot as extra constant features across time
        if self.faction is not None:
            fvec = self.faction[n]
            fmat = np.tile(fvec.reshape(1,-1), (self.history,1))
            x = np.concatenate([x, fmat], axis=1)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# compute scaler from training dynamic features
train_dyn = dynamic[:, train_start:train_end+1, :]
dyn_mean = np.mean(train_dyn.reshape(-1, train_dyn.shape[-1]), axis=0)
dyn_std = np.std(train_dyn.reshape(-1, train_dyn.shape[-1]), axis=0)
scaler = {'mean': dyn_mean, 'std': dyn_std}

train_ds = TSMany(dynamic, train_idxs, H, P, faction=faction_sel if K>0 else None, scaler=scaler)
test_ds = TSMany(dynamic, test_idxs, H, P, faction=faction_sel if K>0 else None, scaler=scaler) if test_idxs else None

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# model
class LSTMModel(nn.Module):
    def __init__(self, in_dim, hid=64, out=P, nlayers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=nlayers, batch_first=True)
        self.fc = nn.Linear(hid, out)

    def forward(self,x):
        # x: (B, H, in_dim)
        out, (h,c) = self.lstm(x)
        # take last timestep output
        last = out[:, -1, :]
        return self.fc(last)

in_dim = F + (K if K>0 else 0)
model = LSTMModel(in_dim=in_dim).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print('Training', EPOCHS, 'epochs; train samples:', len(train_ds))
for ep in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    print(f'Epoch {ep+1}/{EPOCHS} loss={total_loss/len(train_ds):.6f}')

# Evaluation: generate predictions for each test start and node
results = {}
os_plot_dir = Path(__file__).parent / 'plots'
os_plot_dir.mkdir(exist_ok=True)
csv_rows = []
if test_ds is not None and len(test_ds)>0:
    model.eval()
    with torch.no_grad():
        # collect preds grouped by node
        per_node_preds = {i: [] for i in range(len(idxs))}
        per_node_trues = {i: [] for i in range(len(idxs))}
        for idx in range(len(test_ds)):
            xb, yb = test_ds[idx]
            xb = xb.unsqueeze(0).to(DEVICE)
            pred = model(xb).cpu().numpy().ravel()
            true = yb.numpy().ravel()
            # determine node and start
            n,s = test_ds.samples[idx]
            per_node_preds[n].append(pred)
            per_node_trues[n].append(true)
            # csv row with start date
            start_date = dates[s+self.history].strftime('%Y-%m-%d') if dates is not None else str(s)
            row = {'start_date': start_date, 'bairro': bairros[idxs[n]], 'pred': pred.tolist(), 'true': true.tolist()}
            csv_rows.append(row)

    # aggregate per-node MSE and plot average true vs pred
    for n in range(len(idxs)):
        preds = np.array(per_node_preds[n]) if per_node_preds[n] else np.empty((0,P))
        trues = np.array(per_node_trues[n]) if per_node_trues[n] else np.empty((0,P))
        if preds.size==0:
            continue
        mse = float(((preds - trues)**2).mean())
        avg_pred = preds.mean(axis=0)
        avg_true = trues.mean(axis=0)
        results[bairros[idxs[n]]] = {'mse': mse, 'avg_true': avg_true.tolist(), 'avg_pred': avg_pred.tolist()}
        # plot
        plt.figure()
        x = np.arange(P)
        plt.plot(x, avg_true, label='true')
        plt.plot(x, avg_pred, label='pred')
        plt.title(f'{bairros[idxs[n]]} avg true vs pred (P={P})')
        plt.xlabel('horizon days')
        plt.ylabel('CVLI')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os_plot_dir / f'{bairros[idxs[n]]}_pred_vs_true.png')
        plt.close()

    # write CSV
    import csv
    csv_path = os_plot_dir / 'predictions_test.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        header = ['start_date','bairro'] + [f'true_{i}' for i in range(P)] + [f'pred_{i}' for i in range(P)]
        w.writerow(header)
        for r in csv_rows:
            w.writerow([r['start_date'], r['bairro']] + r['true'] + r['pred'])
else:
    # fallback: no full test windows (common when 2026 data is short).
    # Use last available history window and predict horizon; compare overlap with available true days.
    print('No full test windows found — using final available window for evaluation')
    import csv
    csv_path = os_plot_dir / 'predictions_test_fallback.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        header = ['start_date','bairro'] + [f'true_{i}' for i in range(P)] + [f'pred_{i}' for i in range(P)]
        w.writerow(header)
        model.eval()
        with torch.no_grad():
            for n in range(len(idxs)):
                series = dynamic[n, :, :]
                avail_T = series.shape[0]
                # get last H timesteps (pad left with zeros if needed)
                if avail_T >= H:
                    x_raw = series[-H:, :]
                else:
                    pad = np.zeros((H - avail_T, series.shape[1]), dtype=float)
                    x_raw = np.vstack([pad, series[:, :]])
                x = (x_raw - scaler['mean']) / (scaler['std'] + 1e-9)
                if faction_sel is not None:
                    fvec = faction_sel[n]
                    fmat = np.tile(fvec.reshape(1,-1), (H,1))
                    x = np.concatenate([x, fmat], axis=1)
                xb = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                pred = model(xb).cpu().numpy().ravel()
                # true available tail
                true_avail = dynamic[n, -P:, 0] if avail_T >= P else dynamic[n, -avail_T:, 0]
                # pad true to length P with NaN for missing
                true_padded = np.full((P,), np.nan)
                true_padded[:len(true_avail)] = true_avail
                start_date = dates[-(len(true_avail))].strftime('%Y-%m-%d') if dates is not None else 'final'
                row = [start_date, bairros[idxs[n]]] + true_padded.tolist() + pred.tolist()
                w.writerow(row)
            print('Wrote fallback CSV to', csv_path)

# save results
OUT = Path(__file__).parent / 'train_results.json'
with OUT.open('w', encoding='utf-8') as fh:
    json.dump({'config':cfg, 'results':results}, fh, ensure_ascii=False, indent=2)

print('Wrote results to', OUT)
print('Saved plots/csv to', os_plot_dir)
