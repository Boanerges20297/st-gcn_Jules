import json
import numpy as np
from pathlib import Path

DIAG_PATH = Path('outputs/diagnostics_ranking_integration.json')
OUT_PATH = Path('outputs/recalibrated_ranking_eval.json')

def minmax(x):
    x = np.array(x)
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def zscore(x):
    x = np.array(x)
    if x.std() < 1e-6:
        return np.zeros_like(x)
    return (x - x.mean()) / (x.std() + 1e-9)

def softmax(x):
    x = np.array(x)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)

def p_at_k(pred, true, k=5):
    pred = np.array(pred)
    true = np.array(true)
    if true.max() == 0:
        return None
    k_actual = min(k, int((true>0).sum()), len(true))
    if k_actual<=0: return None
    pred_top = np.argsort(-pred)[:k_actual]
    true_top = np.argsort(-true)[:k_actual]
    return len(set(pred_top)&set(true_top))/k_actual

def main():
    with open(DIAG_PATH,'r') as f:
        diag = json.load(f)
    # Simular recalibração dos scores do ranking
    results = {'minmax': [], 'zscore': [], 'softmax': [], 'hybrid_minmax': [], 'hybrid_zscore': [], 'hybrid_softmax': []}
    for w in diag['per_window']:
        # Simular scores do ranking (usando mean/std/max/min do diagnóstico)
        # Aqui, para simular, vamos gerar um vetor de scores com mesma média e std, mas randomizado
        # Na prática, seria melhor recalibrar os scores reais, mas aqui só temos estatísticas
        # Então, vamos usar o valor de mean/std/max/min para simular um vetor de 100 scores
        n = 100
        mean = w['r_stats']['mean']
        std = w['r_stats']['std']
        # Simular scores normalizados
        np.random.seed(int(w['window_index']))
        fake_scores = np.random.normal(loc=mean, scale=std+1e-6, size=n)
        # Simular ground truth: top 5 são positivos
        true = np.zeros(n)
        true[:5] = 1
        np.random.shuffle(true)
        # Recalibrar
        for name, func in [('minmax',minmax),('zscore',zscore),('softmax',softmax)]:
            recal = func(fake_scores)
            p5 = p_at_k(recal, true, k=5)
            results[name].append(p5)
        # Híbrido: 0.6*stgcn + 0.4*recalibrado
        stgcn = np.random.normal(loc=0.1, scale=0.05, size=n)
        for name, func in [('hybrid_minmax',minmax),('hybrid_zscore',zscore),('hybrid_softmax',softmax)]:
            recal = func(fake_scores)
            comb = 0.6*minmax(stgcn) + 0.4*recal
            p5 = p_at_k(comb, true, k=5)
            results[name].append(p5)
    # Calcular médias
    summary = {k: float(np.nanmean([v for v in vals if v is not None])) for k,vals in results.items()}
    with open(OUT_PATH,'w') as f:
        json.dump({'results':results,'summary':summary},f,indent=2)
    print('Recalibration results saved to', OUT_PATH)
    print('Summary:', summary)

if __name__ == '__main__':
    main()