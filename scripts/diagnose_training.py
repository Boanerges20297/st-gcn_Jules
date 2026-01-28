import os
import time
import numpy as np
import torch
from src.train_separate import NodeFeatureWindowDataset
from src.model import STGCN

ROOT = os.path.dirname(os.path.dirname(__file__))
graph_nf = os.path.join(ROOT, 'data', 'processed', 'graph_data', 'node_features.npy')
adj_np = os.path.join(ROOT, 'data', 'processed', 'graph_data', 'adj_matrix.npy')

print(f"graph_nf exists: {os.path.exists(graph_nf)}")
if os.path.exists(graph_nf):
    print(f"node_features filesize: {os.path.getsize(graph_nf)} bytes")
    arr = np.load(graph_nf, mmap_mode='r')
    print("np.memmap shape, dtype:", arr.shape, arr.dtype)
else:
    raise SystemExit('node_features.npy not found')

if not os.path.exists(adj_np):
    raise SystemExit('adj_matrix.npy not found')

print('\nInstantiating NodeFeatureWindowDataset...')
dataset = NodeFeatureWindowDataset(graph_nf, history_window=90, horizon=7, feature_idx=0)
print('Dataset samples:', len(dataset))
print('Dataset N,T,C:', dataset.N, dataset.T, dataset.C)

# measure single sample access
print('\nSampling single item...')
t0 = time.time()
xs, ys = dataset[0]
t1 = time.time()
print('single __getitem__ time:', t1-t0)
print('sample shapes:', xs.shape, ys.shape)

# batch via DataLoader
from torch.utils.data import DataLoader
print('\nLoading first batch via DataLoader...')
dl = DataLoader(dataset, batch_size=4, shuffle=False)
t0 = time.time()
bx, by = next(iter(dl))
t1 = time.time()
print('batch load time:', t1-t0)
print('batch shapes:', bx.shape, by.shape)

# quick model forward
print('\nBuilding STGCN model and doing forward...')
model = STGCN(num_nodes=dataset.N, in_channels=1, time_steps=90, num_classes=1)
model.eval()

# load adj and normalize (same logic as train script)
adj_arr = np.load(adj_np, allow_pickle=True)
adj_t = torch.FloatTensor(adj_arr)
rowsum = adj_t.sum(1)
d_inv_sqrt = torch.pow(rowsum, -0.5)
d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt)
# make list of adjacencies matching expected num_graphs
norm_adj_list = [norm_adj, norm_adj]

with torch.no_grad():
    bx_t = bx
    t0 = time.time()
    out = model(bx_t, norm_adj_list)
    t1 = time.time()
    print('model forward time:', t1-t0)
    print('model out shape:', out.shape)

print('\nDiagnostic complete.')
