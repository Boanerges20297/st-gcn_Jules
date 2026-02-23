import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import sys
from datetime import datetime

# Adicionar raiz ao path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def prepare_ranking_data(region_key='fortaleza'):
    path = os.path.join(ROOT, 'data', 'processed', f'processed_{region_key}.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['node_features'] # (Nodes, Days, Channels)
    dates = pd.to_datetime(data['dates'])
    nodes_gdf = data['nodes_gdf']
    
    # 1. Filtro Temporal: 2024 e 2025
    mask_2024_2025 = (dates >= '2024-01-01') & (dates <= '2025-12-31')
    target_dates = dates[mask_2024_2025]
    start_idx = np.where(mask_2024_2025)[0][0]
    end_idx = np.where(mask_2024_2025)[0][-1]
    
    num_nodes = features.shape[0]
    
    rows = []
    groups = []
    
    print(f"--- Preparando dados para Ranking ({len(target_dates)} dias) ---")
    
    # Label: soma dos próximos 7 dias (alvo do ranking)
    # Features: média móvel, tendência, tensão, facção
    for t_idx in range(start_idx, end_idx - 7):
        current_date = dates[t_idx]
        
        # Grupo: Todos os bairros deste dia t
        for n_idx in range(num_nodes):
            # Features base
            cvli_7d_sma = features[n_idx, t_idx, 24]
            cvli_14d_sma = pd.Series(features[n_idx, :t_idx+1, 0]).tail(14).mean()
            tension = features[n_idx, t_idx, 2]
            
            # Tendência (Seman atual vs Anterior)
            last_7 = features[n_idx, t_idx-7:t_idx, 0].sum()
            prev_7 = features[n_idx, t_idx-14:t_idx-7, 0].sum()
            trend = last_7 - prev_7
            
            # Média histórica do bairro até hoje
            hist_mean = features[n_idx, :t_idx+1, 0].mean()
            
            # Label para Ranking: Crimes nos próximos 7 dias
            label = features[n_idx, t_idx+1:t_idx+8, 0].sum()
            
            rows.append({
                'day_idx': t_idx,
                'node_idx': n_idx,
                'cvli_7d_sma': cvli_7d_sma,
                'cvli_14d_sma': cvli_14d_sma,
                'tension': tension,
                'trend': trend,
                'hist_mean': hist_mean,
                'dow': current_date.dayofweek,
                'month': current_date.month,
                'label': int(label)
            })
        
        groups.append(num_nodes)
        
    df = pd.DataFrame(rows)
    return df, groups

def train_ranking():
    df, groups = prepare_ranking_data('fortaleza')
    
    # Split: 80% treino, 20% validação (temporal)
    split_point = int(len(groups) * 0.8)
    train_groups = groups[:split_point]
    val_groups = groups[split_point:]
    
    train_rows = sum(train_groups)
    
    X = df.drop(columns=['day_idx', 'node_idx', 'label'])
    y = df['label']
    
    X_train, y_train = X.iloc[:train_rows], y.iloc[:train_rows]
    X_val, y_val = X.iloc[train_rows:], y.iloc[train_rows:]
    
    print(f"Treino: {len(X_train)} linhas | Val: {len(X_val)} linhas")
    
    # Configuração LightGBM LambdaRank
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        importance_type='gain',
        label_gain=np.arange(y.max() + 1), # Dá mais peso para quem tem mais crimes
        random_state=42
    )
    
    print("--- Iniciando Treinamento LambdaRank ---")
    ranker.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        eval_at=[5, 10, 20]
    )
    
    # 4. Avaliar Precisão no Top 20 (Métrica do Usuário)
    print("\n--- Avaliação de Eficiência (Top 20) ---")
    preds = ranker.predict(X_val)
    X_val_eval = X_val.copy()
    X_val_eval['pred'] = preds
    X_val_eval['label'] = y_val
    X_val_eval['day_idx'] = df.iloc[train_rows:]['day_idx']
    
    p20_scores = []
    for day in X_val_eval['day_idx'].unique():
        day_df = X_val_eval[X_val_eval['day_idx'] == day]
        top_20_true = day_df.nlargest(20, 'label').index
        top_20_pred = day_df.nlargest(20, 'pred').index
        
        # Se o dia não teve crimes, ignorar na média de precisão
        if day_df['label'].sum() == 0: continue
            
        hits = len(set(top_20_pred) & set(top_20_true))
        p20_scores.append(hits / 20.0)
    
    avg_p20 = np.mean(p20_scores) if p20_scores else 0
    print(f"✅ Média P@20 (Validação): {avg_p20*100:.2f}%")
    
    # 5. Salvar Modelo
    model_dir = os.path.join(ROOT, 'models', 'active')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'ranking_model_fortaleza.txt')
    ranker.booster_.save_model(model_path)
    
    print(f"📦 Modelo salvo em: {model_path}")

if __name__ == "__main__":
    train_ranking()
