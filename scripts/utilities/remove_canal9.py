"""
Remover canal 9 - voltar para 8 canais
"""
import pickle
import numpy as np

with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    d = pickle.load(f)

nf = d['node_features']
print(f"Shape atual: {nf.shape}")

# Se tem 9 canais, remover canal 9 (index 8)
if nf.shape[2] == 9:
    nf_8 = nf[:, :, :8]
    d['node_features'] = nf_8
    d['feature_names'] = ['CVLI', 'CVP', 'TENSION_INDEX', 'DOW_SIN', 'DOW_COS', 'MONTH_SIN', 'MONTH_COS', 'IS_WEEKEND']
    
    with open('data/processed/processed_graph_data.pkl', 'wb') as f:
        pickle.dump(d, f)
    
    print(f"[OK] Revertido para 8 canais: {nf_8.shape}")
else:
    print(f"[OK] Ja tem {nf.shape[2]} canais")
