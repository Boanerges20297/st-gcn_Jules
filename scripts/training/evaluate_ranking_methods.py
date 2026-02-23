import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import os
import sys

# Adicionar raiz ao path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.training.train_ranking_fortaleza import prepare_ranking_data

def evaluate_methods():
    df, groups = prepare_ranking_data('fortaleza')
    
    # Split temporal
    split_point = int(len(groups) * 0.8)
    train_rows = sum(groups[:split_point])
    
    df_train = df.iloc[:train_rows]
    df_test = df.iloc[train_rows:]
    
    print(f"\n--- Avaliando Métodos de Ranking ({len(df_test['day_idx'].unique())} dias de teste) ---")
    
    results = {}

    # --- MÉTODO 1: RANKING BRUTO (Média Histórica) ---
    # Usamos a média de crimes do treino para prever o teste
    bruto_map = df_train.groupby('node_idx')['label'].mean().to_dict()
    df_test['pred_bruto'] = df_test['node_idx'].map(bruto_map).fillna(0)
    
    # --- MÉTODO 2: BLEND HEURÍSTICO (Histórico + SMA + Trend) ---
    # Score = 0.5*Hist + 0.3*SMA_7d + 0.2*Trend
    df_test['pred_blend'] = (0.5 * df_test['hist_mean']) + \
                            (0.3 * df_test['cvli_7d_sma']) + \
                            (0.2 * df_test['trend'])

    # --- MÉTODO 3: RANDOM FOREST (Regressão) ---
    X_train = df_train.drop(columns=['day_idx', 'node_idx', 'label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['day_idx', 'node_idx', 'label', 'pred_bruto', 'pred_blend'])
    
    print("Treinando Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    df_test['pred_rf'] = rf.predict(X_test)

    # --- AVALIAÇÃO DE PRECISÃO TOP 20 ---
    methods = ['pred_bruto', 'pred_blend', 'pred_rf']
    
    for m in methods:
        p20_scores = []
        for day in df_test['day_idx'].unique():
            day_df = df_test[df_test['day_idx'] == day]
            if day_df['label'].sum() == 0: continue
            
            top_20_true = day_df.nlargest(20, 'label').index
            top_20_pred = day_df.nlargest(20, m).index
            
            hits = len(set(top_20_pred) & set(top_20_true))
            p20_scores.append(hits / 20.0)
        
        results[m] = np.mean(p20_scores)
        print(f"✅ {m.upper()}: {results[m]*100:.2f}%")

    print("\n" + "="*50)
    best_m = max(results, key=results.get)
    print(f"🏆 VENCEDOR: {best_m.upper()} ({results[best_m]*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    evaluate_methods()
