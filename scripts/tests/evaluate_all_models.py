#!/usr/bin/env python3
"""
Avalia todos os modelos na pasta `models/` e gera um CSV resumo.
Opcional: `--prune --threshold <metric>` remove modelos com desempenho abaixo do limiar (interativo por padrão).

Usage:
    python scripts/evaluate_all_models.py [--prune] [--threshold 0.12] [--out results.csv]

Notas:
 - Reusa parte da lógica de `scripts/evaluate_model_v3.py` para ST-GCN.
 - Para modelos de ranking em pickle, tenta carregar e usar `src.ranking_model_v2.RankingModel`.
 - Não deleta nada sem `--prune`.
"""

import os
import sys
import pickle
import joblib
import argparse
import numpy as np
import torch
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import STGCN
from src.ranking_model_v2 import RankingModel

MODELS_DIR = os.path.join(ROOT, 'models')
DATA_FILE = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')

# --- Utilities ---

def normalize_adj(adj_np):
    adj_t = torch.FloatTensor(adj_np)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt)


def precision_at_k_numpy(pred, target, k=5):
    # pred/target: 1D arrays of length num_nodes
    if target.max() == 0:
        return np.nan
    k = min(k, len(pred))
    pred_topk = np.argsort(pred)[-k:]
    true_topk = np.argsort(target)[-k:]
    hits = len(set(pred_topk) & set(true_topk))
    denom = min(k, (target > 0).sum())
    if denom == 0:
        return np.nan
    return hits / denom


def ndcg_at_k(pred, target, k=5):
    # pred and target are 1D arrays
    k = min(k, len(pred))
    idx_pred = np.argsort(pred)[::-1][:k]
    idx_true = np.argsort(target)[::-1][:k]
    # DCG
    dcg = 0.0
    for i, idx in enumerate(idx_pred):
        rel = target[idx]
        dcg += (2**rel - 1) / np.log2(i + 2)
    # IDCG
    idcg = 0.0
    for i, idx in enumerate(idx_true):
        rel = target[idx]
        idcg += (2**rel - 1) / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else np.nan


# --- Evaluators ---

def evaluate_stgcn(model_path, data_pack, history_window=30, batch_size=32):
    node_features = data_pack['node_features']
    adj_geo = data_pack['adj_geo'] if 'adj_geo' in data_pack else data_pack.get('adj_matrix')
    # Avoid ambiguous truth-value checks on numpy arrays
    if 'adj_conflict' in data_pack and data_pack['adj_conflict'] is not None:
        adj_faction = data_pack['adj_conflict']
    elif 'adj_faction' in data_pack and data_pack['adj_faction'] is not None:
        adj_faction = data_pack['adj_faction']
    else:
        adj_faction = adj_geo

    # Prepare windows similar to evaluate_model_v3
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(node_features, history_window, axis=1)
    X = windows[:, :-1, :, :]
    Y = node_features[:, history_window:, 0:1]

    X = X.transpose(1, 2, 0, 3)
    Y = Y.transpose(1, 0, 2)

    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]

    norm_adj_geo = normalize_adj(adj_geo).to(device)
    norm_adj_faction = normalize_adj(adj_faction).to(device)
    norm_adj_list = [norm_adj_geo, norm_adj_faction]

    model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=history_window, num_classes=1, num_graphs=2)
    model.to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        return {'error': f'failed to load: {e}'}

    model.eval()

    all_preds = []
    all_targets = []
    all_p5 = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            bx = torch.FloatTensor(X_test[i:batch_end]).to(device)
            by = torch.FloatTensor(Y_test[i:batch_end]).to(device)
            out = model(bx, norm_adj_list)
            preds = out.detach().cpu().numpy()
            targs = by.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targs)

            # compute p@5 per sample
            for j in range(preds.shape[0]):
                p5 = precision_at_k_numpy(preds[j,:,0], targs[j,:,0], k=5)
                if not np.isnan(p5):
                    all_p5.append(p5)

    if len(all_preds) == 0:
        return {'error': 'no predictions'}

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mean_p5 = np.nanmean(all_p5) if all_p5 else np.nan
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))

    return {'type':'stgcn', 'p5': float(mean_p5), 'mae': float(mae), 'rmse': float(rmse)}


def evaluate_ranking_model(file_path, data_pack):
    # Use simple protocol: per sample aggregate features and compare ranking to target
    node_features = data_pack['node_features']
    # We'll use the last available window to compute one prediction per node
    # Targets: next-day sums (use last day as ground truth slice if available)
    num_nodes, num_timesteps, num_features = node_features.shape
    if num_timesteps < 2:
        return {'error': 'not enough timesteps'}

    # Input X: use last time-step features per node
    X = node_features[:, -1, :]
    # Target: next day counts (if available) otherwise use last day as proxy
    Y = node_features[:, -1, 0]

    # Try torch.load first for .pth/.pt files or complex objects
    try:
        if file_path.endswith('.pth') or file_path.endswith('.pt'):
            obj = torch.load(file_path, map_location='cpu')
        else:
            # Try joblib first (common for sklearn pipelines), then pickle
            try:
                obj = joblib.load(file_path)
            except Exception:
                with open(file_path, 'rb') as f:
                    obj = pickle.load(f)
    except Exception as e:
        return {'error': f'load failed: {e}'}

    # Case 1: torch state_dict or object
    if isinstance(obj, dict) and any((isinstance(k, str) and (k.startswith('module.') or k in ['net', 'state_dict', 'model_state_dict'])) for k in obj.keys()):
        # instantiate RankingModel and load state
        input_dim = X.shape[1]
        model = RankingModel(input_dim=input_dim)
        try:
            # handle different keys
            if 'state_dict' in obj:
                state = obj['state_dict']
            elif 'model_state_dict' in obj:
                state = obj['model_state_dict']
            else:
                state = obj
            # If state is a full torch model object, try to extract state_dict
            if hasattr(state, 'keys'):
                try:
                    model.load_state_dict(state)
                except Exception:
                    # Maybe the dict is nested under 'net' or similar
                    for candidate in ['net', 'model', 'state_dict', 'model_state_dict']:
                        if candidate in state:
                            model.load_state_dict(state[candidate])
                            break
            else:
                # Unexpected format
                return {'error': 'state_dict format not recognized'}

            model.eval()
            with torch.no_grad():
                scores = model(torch.FloatTensor(X)).numpy()
            ndcg5 = ndcg_at_k(scores.flatten(), Y.flatten(), k=5)
            p5 = precision_at_k_numpy(scores.flatten(), Y.flatten(), k=5)
            return {'type':'ranking', 'ndcg5': float(ndcg5) if not np.isnan(ndcg5) else None, 'p5': float(p5) if not np.isnan(p5) else None}
        except Exception as e:
            return {'error': f'failed to use state_dict: {e}'}

    # Case 2: sklearn / pipeline object with predict or predict_proba
    if hasattr(obj, 'predict'):
        try:
            scores = obj.predict(X)
            ndcg5 = ndcg_at_k(scores.flatten(), Y.flatten(), k=5)
            p5 = precision_at_k_numpy(scores.flatten(), Y.flatten(), k=5)
            return {'type':'ranking', 'ndcg5': float(ndcg5) if not np.isnan(ndcg5) else None, 'p5': float(p5) if not np.isnan(p5) else None}
        except Exception as e:
            return {'error': f'predict failed: {e}'}
    # Fallback: if it's a numpy array or list of scores
    if isinstance(obj, (np.ndarray, list)):
        try:
            scores = np.array(obj).flatten()
            ndcg5 = ndcg_at_k(scores.flatten(), Y.flatten(), k=5)
            p5 = precision_at_k_numpy(scores.flatten(), Y.flatten(), k=5)
            return {'type':'ranking', 'ndcg5': float(ndcg5) if not np.isnan(ndcg5) else None, 'p5': float(p5) if not np.isnan(p5) else None}
        except Exception as e:
            return {'error': f'unable to evaluate array-like object: {e}'}

    return {'error': f'unknown object type: {type(obj)}'}


# --- Main flow ---

def discover_models(models_dir=MODELS_DIR):
    files = sorted([f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir,f))])
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune', action='store_true', help='Remover modelos abaixo do limiar (usar --threshold)')
    parser.add_argument('--threshold', type=float, default=0.12, help='Limiar mínimo para P@5 (ST-GCN) ou P@5 (ranking)')
    parser.add_argument('--out', type=str, default='model_evaluation_results.csv')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR)
    args = parser.parse_args()

    if not os.path.exists(DATA_FILE):
        print('Erro: dados processados não encontrados em', DATA_FILE)
        return

    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)

    models = discover_models(args.models_dir)
    results = []

    for m in models:
        path = os.path.join(args.models_dir, m)
        print('Evaluating', m)
        if m.endswith('.pth') and 'stgcn' in m.lower():
            r = evaluate_stgcn(path, data_pack)
        elif m.endswith('.pth') and 'ranking' in m.lower():
            # maybe a torch state_dict for ranking
            r = evaluate_ranking_model(path, data_pack)
        elif m.endswith('.pkl') or m.endswith('.pickle'):
            r = evaluate_ranking_model(path, data_pack)
        else:
            r = {'error': 'unknown model type for evaluation'}

        row = {'model': m}
        row.update(r)
        results.append(row)
        print(' ->', r)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print('\nSaved results to', args.out)

    # Prune logic (interactive)
    if args.prune:
        to_delete = []
        for idx, row in df.iterrows():
            score = None
            if row.get('type') == 'stgcn':
                score = row.get('p5')
            else:
                score = row.get('p5') if not pd.isna(row.get('p5')) else row.get('ndcg5')
            if score is None or (isinstance(score, float) and score < args.threshold):
                to_delete.append(row['model'])

        if not to_delete:
            print('Nenhum modelo abaixo do limiar encontrado.')
        else:
            print('Modelos candidatos à remoção:')
            for m in to_delete:
                print('  -', m)
            confirm = input('Deseja remover esses arquivos? (yes/no) ')
            if confirm.lower() in ['y', 'yes']:
                for m in to_delete:
                    p = os.path.join(args.models_dir, m)
                    try:
                        os.remove(p)
                        print('Removido', m)
                    except Exception as e:
                        print('Falha ao remover', m, e)
            else:
                print('Nenhuma ação de remoção executada.')

if __name__ == '__main__':
    main()
