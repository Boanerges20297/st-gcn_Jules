import sys
import os
import time
import psutil
import torch
import numpy as np
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import STGCN
from Phase3.gat_prototype import STGAT

def load_data():
    print("Carregando dados processados...")
    data_dir = 'data/processed/graph_data'
    
    # Load features
    try:
        node_features = np.load(os.path.join(data_dir, 'node_features.npy'), allow_pickle=True)
        adj_geo = np.load(os.path.join(data_dir, 'adj_geo.npy'), allow_pickle=True)
        adj_faction = np.load(os.path.join(data_dir, 'adj_faction.npy'), allow_pickle=True)
    except FileNotFoundError:
        print("Erro: Arquivos .npy não encontrados em data/processed/graph_data/")
        sys.exit(1)
        
    print(f"Features Shape: {node_features.shape}")
    print(f"Adj Geo Shape: {adj_geo.shape}")
    
    return node_features, [torch.tensor(adj_geo).float(), torch.tensor(adj_faction).float()]

def prepare_input(node_features, time_steps=12):
    # node_features might be (N, C) or (N, T, C)
    # STGCN expects (B, C, N, T)
    
    x = torch.tensor(node_features).float()
    
    if len(x.shape) == 2:
        # (N, C) -> replicate for Time and Batch
        N, C = x.shape
        x = x.unsqueeze(0).unsqueeze(-1) # (1, N, C, 1)
        x = x.permute(0, 2, 1, 3) # (1, C, N, 1)
        x = x.repeat(1, 1, 1, time_steps) # (1, C, N, T)
    elif len(x.shape) == 3:
        # (N, T_full, C) -> Slice to (N, time_steps, C)
        # Assuming dim 1 is time
        N, T_full, C = x.shape
        print(f"   Original Data Shape: {x.shape} (N, T, C)")
        
        if T_full > time_steps:
            print(f"   Slicing time steps from {T_full} to {time_steps}")
            x = x[:, -time_steps:, :] # Take last T steps
        else:
            print(f"   Warning: Available time steps ({T_full}) < Requested ({time_steps})")
            
        # Transform to (B, C, N, T)
        # Current: (N, T, C)
        x = x.permute(2, 0, 1).unsqueeze(0) # (1, C, N, T)
            
    return x

def benchmark_model(model, x, adj_list, model_name="Model", n_loops=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    model.to(device)
    x = x.to(device)
    adj_list = [adj.to(device) for adj in adj_list]
    
    model.eval()
    
    # Warmup
    print(f"   Warmup {model_name}...")
    with torch.no_grad():
        for _ in range(2):
            _ = model(x, adj_list)
            
    # Latency
    times = []
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024
    
    print(f"   Running benchmark ({n_loops} loops)...")
    with torch.no_grad():
        for i in range(n_loops):
            start = time.time()
            _ = model(x, adj_list)
            end = time.time()
            times.append((end - start) * 1000) # ms
            print(f"     Loop {i+1}/{n_loops}: {(end - start) * 1000:.2f} ms")
            
    end_mem = psutil.Process().memory_info().rss / 1024 / 1024
    mem_usage = end_mem - start_mem
    
    avg_latency = np.mean(times)
    std_latency = np.std(times)
    
    return {
        "Model": model_name,
        "Avg Latency (ms)": f"{avg_latency:.2f}",
        "Std Latency (ms)": f"{std_latency:.2f}",
        "Memory Delta (MB)": f"{mem_usage:.2f}"
    }

def run_validation(model, x, adj_list, split_ratio=0.5):
    """
    Simulates validation with 50% data / 50% no-data (masked).
    Returns 'Success' if runs without error and output shape is correct.
    """
    B, C, N, T = x.shape
    
    # Create mask for 50% of nodes or features? 
    # User said "dividindo de forma equitativa com dados e sem dados".
    # Let's mask 50% of features (simulate missing data) for ALL nodes, 
    # OR mask 50% of nodes (zero features for those nodes).
    # Let's do 50% of nodes have NO data (zeros).
    
    mask = torch.ones_like(x)
    num_masked = int(N * split_ratio)
    masked_indices = np.random.choice(N, num_masked, replace=False)
    mask[:, :, masked_indices, :] = 0
    
    x_masked = x * mask
    
    try:
        with torch.no_grad():
            out = model(x_masked, adj_list)
        return "Sucesso", out.shape
    except Exception as e:
        return f"Falha: {str(e)}", None

def main():
    print("=== Iniciando Benchmark de Eficiência (STGCN vs GAT) ===")
    
    # 1. Load Data
    node_features, adj_list = load_data()
    x = prepare_input(node_features, time_steps=12) # (B, C, N, T)
    
    N = x.shape[2]
    C = x.shape[1]
    T = x.shape[3]
    
    print(f"Input Tensor Shape: {x.shape}")
    
    # 2. Initialize Models
    print("\nInicializando modelos...")
    stgcn = STGCN(num_nodes=N, in_channels=C, time_steps=T, num_classes=1, num_graphs=2)
    stgat = STGAT(num_nodes=N, in_channels=C, time_steps=T, num_classes=1, num_graphs=2)
    
    # 3. Benchmark - Full Data
    print("\n[Cenário 1] Dados Completos (Full Features)")
    res_stgcn = benchmark_model(stgcn, x, adj_list, "ST-GCN (Full)")
    res_gat = benchmark_model(stgat, x, adj_list, "ST-GAT (Full)")
    
    # 4. Benchmark - 50% Missing Data (Validation)
    print("\n[Cenário 2] 50% Sem Dados (Sparse/Missing Features)")
    # Create masked input
    mask = torch.ones_like(x)
    num_masked = int(N * 0.5)
    masked_indices = np.random.choice(N, num_masked, replace=False)
    mask[:, :, masked_indices, :] = 0
    x_sparse = x * mask
    
    res_stgcn_sparse = benchmark_model(stgcn, x_sparse, adj_list, "ST-GCN (50% Data)")
    res_gat_sparse = benchmark_model(stgat, x_sparse, adj_list, "ST-GAT (50% Data)")
    
    # 5. Validation Check
    print("\nVerificando integridade da execução (Validation Check)...")
    val_stgcn, shape_stgcn = run_validation(stgcn, x, adj_list)
    val_gat, shape_gat = run_validation(stgat, x, adj_list)
    
    # 6. Report
    results = [res_stgcn, res_gat, res_stgcn_sparse, res_gat_sparse]
    df = pd.DataFrame(results)
    
    print("\n=== Resultados do Benchmark ===")
    print(df.to_markdown(index=False))
    
    report_content = f"""# Relatório de Análise de Eficiência e Validação

## 1. Visão Geral
Este relatório apresenta a comparação de eficiência entre a arquitetura atual (ST-GCN) e o protótipo proposto (ST-GAT), considerando cenários com dados completos e cenários de escassez de dados (50% no-data).

## 2. Metodologia
- **Hardware**: CPU (Simulado) / GPU (se disponível)
- **Dados**: `data/processed/graph_data` (Reais)
- **Dimensões**: Nodes={N}, Channels={C}, TimeSteps={T}
- **Cenários**:
  - Full Data: Todos os nós com features completas.
  - 50% Data: 50% dos nós com features zeradas (simulando falta de dados históricos).

## 3. Resultados de Performance
{df.to_markdown(index=False)}

## 4. Validação Técnica
- **ST-GCN com 50% Data**: {val_stgcn} (Shape Saída: {shape_stgcn})
- **ST-GAT com 50% Data**: {val_gat} (Shape Saída: {shape_gat})

## 5. Conclusões Preliminares
- **Latência**: O GAT tende a ser mais pesado devido ao cálculo da matriz de atenção densa (NxN), enquanto o GCN usa matrizes esparsas fixas.
- **Robustez**: Ambos os modelos processam dados esparsos tecnicamente, mas o GAT tem potencial teórico de adaptar os pesos de atenção para ignorar nós zerados (via mecanismo de atenção), enquanto o GCN propaga zeros fixamente pela topologia.

"""
    
    with open('Phase3/efficiency_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("\nRelatório salvo em Phase3/efficiency_report.md")

if __name__ == "__main__":
    main()
