import os
import pickle
import numpy as np
from pathlib import Path

# Ensure src import works
import sys
sys.path.insert(0, '.')

from src.ranking_model_v2 import RankingModel, RankingTrainerV2
from src.ranking_features import extract_ranking_features


def dcg_at_k(ranking, labels, k=5):
    dcg = 0.0
    for i in range(min(k, len(ranking))):
        node_id = int(ranking[i])
        relevance = float(labels[node_id])
        dcg += relevance / np.log2(i + 2)
    return dcg


def ndcg_at_k(pred_ranking, true_ranking, labels, k=5):
    dcg_pred = dcg_at_k(pred_ranking, labels, k=k)
    dcg_ideal = dcg_at_k(true_ranking, labels, k=k)
    if dcg_ideal == 0:
        return 0.0
    return dcg_pred / dcg_ideal


def infer_hidden_dim_from_state(state_dict, input_dim=26):
    # Look for first linear weight with second dim == input_dim
    for k, v in state_dict.items():
        if 'weight' in k:
            shape = v.shape
            if len(shape) == 2 and shape[1] == input_dim:
                return shape[0]
    # Fallback
    return 128


def load_model_from_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise RuntimeError('Unexpected pickle format: not a dict')

    state = data.get('model_state')
    config = data.get('config')
    scaler = data.get('trainer_scaler')

    if config and 'hidden_dim' in config:
        hidden = config['hidden_dim']
    else:
        hidden = infer_hidden_dim_from_state(state, input_dim=26)

    model = RankingModel(input_dim=26, hidden_dim=hidden)
    model.load_state_dict(state)

    trainer = RankingTrainerV2(model, device='cpu', lr=0.001)
    if scaler is not None:
        trainer.scaler = scaler

    reported = {
        'config': config,
        'has_history': 'history' in data,
        'best_val_p5': data.get('best_val_p5'),
        'eval_p5': data.get('eval_p5')
    }

    return model, trainer, reported


def evaluate(model, trainer, X, Y, name):
    ranking, scores = trainer.predict(X)
    true_ranking = np.argsort(-Y)
    ndcg5 = ndcg_at_k(ranking, true_ranking, Y, k=5)
    ndcg10 = ndcg_at_k(ranking, true_ranking, Y, k=10)
    overlap5 = len(set(ranking[:5]) & set(true_ranking[:5]))
    p_at_5 = overlap5 / 5.0
    from scipy.stats import spearmanr
    pred_rankings = np.zeros(len(X))
    pred_rankings[ranking] = np.arange(len(X))[::-1]
    true_ranks = np.zeros(len(X))
    true_ranks[true_ranking] = np.arange(len(X))[::-1]
    corr, _ = spearmanr(pred_rankings, true_ranks)

    print(f"\n[EVAL] {name}")
    print(f"  NDCG@5: {ndcg5:.4f}")
    print(f"  NDCG@10: {ndcg10:.4f}")
    print(f"  P@5: {p_at_5:.4f}")
    print(f"  Spearman: {corr:.4f}")
    print(f"  Top-5 Pred: {list(map(int, ranking[:5]))}")
    print(f"  Top-5 Real: {list(map(int, true_ranking[:5]))}")

    return {
        'ndcg5': ndcg5,
        'ndcg10': ndcg10,
        'p_at_5': p_at_5,
        'spearman': corr,
    }


def main():
    # Load data
    pkl = Path('data') / 'processed' / 'processed_graph_data.pkl'
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    node_features = data['node_features']
    dates = data.get('dates')
    if dates is None:
        # synthesize daily dates if not present
        from datetime import datetime, timedelta
        n_timesteps = node_features.shape[1]
        start = datetime(2022, 1, 1)
        dates = [start + timedelta(days=i) for i in range(n_timesteps)]

    X, Y = extract_ranking_features(node_features, dates)
    print(f"Features shape: {X.shape}, Target shape: {Y.shape}")

    targets = [
        Path('models') / 'ranking_model_v2.pkl',
        Path('models') / 'backup' / 'ranking_model_best_Config_02_SmallLR.pkl'
    ]

    for t in targets:
        if not t.exists():
            print(f"Missing model file: {t}")
            continue
        print(f"\nLoading {t}")
        model, trainer, info = load_model_from_pickle(t)
        print('Reported meta:', info)
        evaluate(model, trainer, X, Y, t.name)

if __name__ == '__main__':
    main()
