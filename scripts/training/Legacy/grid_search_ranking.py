import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import sys
from itertools import product

# Adicionar raiz ao path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.training.train_ranking_fortaleza import prepare_ranking_data

def run_grid_search():
    df, groups = prepare_ranking_data('fortaleza')
    
    split_point = int(len(groups) * 0.8)
    train_groups = groups[:split_point]
    val_groups = groups[split_point:]
    train_rows = sum(train_groups)
    
    X = df.drop(columns=['day_idx', 'node_idx', 'label'])
    y = df['label']
    X_train, y_train = X.iloc[:train_rows], y.iloc[:train_rows]
    X_val, y_val = X.iloc[train_rows:], y.iloc[train_rows:]
    
    # Espaço de busca
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'n_estimators': [500, 1000],
        'min_data_in_leaf': [20, 50]
    }
    
    # Gerar combinações
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    best_p20 = -1
    best_params = None
    
    print(f"--- Iniciando Grid Search ({len(combinations)} combinações) ---")
    
    for i, params in enumerate(combinations):
        print(f"Testando {i+1}/{len(combinations)}: {params}")
        
        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            importance_type='gain',
            random_state=42,
            **params
        )
        
        ranker.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            eval_at=[20],
            callbacks=[lgb.log_evaluation(period=0)] # Silencia logs internos
        )
        
        # Avaliação P@20
        preds = ranker.predict(X_val)
        X_val_eval = X_val.copy()
        X_val_eval['pred'] = preds
        X_val_eval['label'] = y_val
        X_val_eval['day_idx'] = df.iloc[train_rows:]['day_idx']
        
        p20_scores = []
        for day in X_val_eval['day_idx'].unique():
            day_df = X_val_eval[X_val_eval['day_idx'] == day]
            if day_df['label'].sum() == 0: continue
            top_20_true = day_df.nlargest(20, 'label').index
            top_20_pred = day_df.nlargest(20, 'pred').index
            hits = len(set(top_20_pred) & set(top_20_true))
            p20_scores.append(hits / 20.0)
        
        avg_p20 = np.mean(p20_scores)
        print(f"   -> P@20: {avg_p20*100:.2f}%")
        
        if avg_p20 > best_p20:
            best_p20 = avg_p20
            best_params = params
            # Salvar o melhor modelo provisório
            ranker.booster_.save_model(os.path.join(ROOT, 'models', 'active', 'ranking_model_fortaleza.txt'))

    print("\n" + "="*50)
    print("🏆 GRID SEARCH CONCLUÍDO")
    print(f"Melhor P@20: {best_p20*100:.2f}%")
    print(f"Melhores Parâmetros: {best_params}")
    print("="*50)

if __name__ == "__main__":
    run_grid_search()
