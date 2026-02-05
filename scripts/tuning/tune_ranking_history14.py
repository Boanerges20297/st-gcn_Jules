import itertools
import pickle
import time
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_ranking_v2 import load_data, extract_features_and_targets, train_ranking_model_v2
from src.ranking_model_v2 import RankingModel, RankingTrainerV2

OUT_DIR = Path('models') / 'tuning_history14'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Small grid
hidden_dims = [256, 512]
lrs = [0.01, 0.001]
weight_decays = [1e-4, 1e-3]
dropouts = [0.3]

best = {'p5': -1, 'cfg': None, 'path': None}

node_features, dates, full = load_data()
if node_features is None:
    raise SystemExit('no data')

# features extraction uses history=14 inside the train script; we reuse extraction here
X, Y = extract_features_and_targets(node_features, dates, horizon_days=7)

for hd, lr, wd, dr in itertools.product(hidden_dims, lrs, weight_decays, dropouts):
    cfg = {'hidden_dim': hd, 'lr': lr, 'weight_decay': wd, 'dropout_main': dr}
    print('\n[GRID] Testing', cfg)
    try:
        # instantiate model/trainer and call train_ranking_model_v2 with injection
        model = RankingModel(input_dim=X.shape[1], hidden_dim=hd, dropout_main=dr, dropout_small=max(0.1, dr/2))
        trainer = RankingTrainerV2(model, device='cpu', lr=lr, weight_decay=wd)

        # train for few epochs
        model, trainer, history, best_p5 = train_ranking_model_v2(X, Y, epochs=8, batch_size=319, device='cpu', model=model, trainer=trainer)
    except Exception as e:
        print('Error during training', e)
        continue

    # Save candidate
    ts = int(time.time())
    outp = OUT_DIR / f'ranking_tune_hd{hd}_lr{lr}_wd{wd}_{ts}.pkl'
    with open(outp, 'wb') as f:
        pickle.dump({'config': cfg, 'history': history, 'best_val_p5': best_p5}, f)

    print('[GRID] Result best_val_p5=', best_p5)
    if best_p5 is not None and best_p5 > best['p5']:
        best.update({'p5': best_p5, 'cfg': cfg, 'path': str(outp)})

print('\nGrid complete. Best:', best)
