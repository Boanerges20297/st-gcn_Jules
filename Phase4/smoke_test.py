import os
import sys

# Altera para o diretório raiz do projeto
project_root = r"C:\Users\Boanerges\Desktop\Projetos\st-gcn_jules"
os.chdir(project_root)
sys.path.append(project_root)

print("Iniciando teste de fumaça da Phase 5 (Escala Massiva)...")

try:
    import torch
    print(f"PyTorch Versão: {torch.__version__}")
    
    # Testa importação do modelo
    from Phase4.model_v4 import DeepSTGAT
    print("Sucesso: Arquitetura DeepSTGAT carregada.")
    
    # Testa importação do Dataset
    from Phase4.train_v4 import LazyCrimeDataset
    print("Sucesso: LazyCrimeDataset carregado.")
    
    # Tenta carregar os dados reais para validar o shape
    import pickle
    DATA_FILE = 'data/processed/processed_graph_data.pkl'
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    node_features = data['node_features']
    print(f"Sucesso: Dados carregados. Shape: {node_features.shape}")
    print(f"Total de dias detectados: {node_features.shape[1]}")
    
    print("\n--- Tudo pronto para o treino massivo ---")
    
except Exception as e:
    print(f"ERRO NO SETUP: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
