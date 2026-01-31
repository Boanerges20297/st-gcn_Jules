import json, statistics
from pathlib import Path

p = Path('tests/novo_modelo/fortaleza_report.json')
j = json.load(p.open(encoding='utf-8'))
metrics = j['metrics']
mses = [v['mse'] for v in metrics.values()]
maes = [v['mae'] for v in metrics.values()]
rmses = [v['rmse'] for v in metrics.values()]
r2s = [v['r2'] for v in metrics.values()]
print('n_bairros', len(metrics))
print('mean_mse', statistics.mean(mses))
print('median_mse', statistics.median(mses))
print('mean_mae', statistics.mean(maes))
print('mean_rmse', statistics.mean(rmses))
print('mean_r2', statistics.mean(r2s))
worst = sorted(metrics.items(), key=lambda kv: kv[1]['mse'], reverse=True)[:10]
print('\nTop 10 bairros by MSE:')
for name, vals in worst:
    print(name, vals['mse'], vals['mae'], vals['rmse'], vals['r2'])
