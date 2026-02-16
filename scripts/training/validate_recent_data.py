"""
Script de validação do modelo com dados recentes (após 02/02/2026)

Este script carrega os dados novos, processa-os, faz predições com o modelo
atual e avalia o desempenho em dados não vistos.
"""

import pandas as pd
import numpy as np
import torch
import pickle
import json
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stgat import STGAT
from src.model import STGCN

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status.json')
PROCESSED_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_STGAT = os.path.join(BASE_DIR, 'models', 'st_gat_production.pth')
MODEL_STGCN = os.path.join(BASE_DIR, 'models', 'stgcn_model_v2.pth')
NODES_FILE = os.path.join(BASE_DIR, 'outputs', 'nodes_with_faction_assigned.geojson')

HISTORY_WINDOW = 12  # Dias de histórico para fazer predição (ajustado para o modelo treinado)
HORIZON = 7  # Dias à frente que o modelo prediz

# Usar ST-GAT (modelo de produção) ou STGCN?
USE_STGAT = True

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def load_recent_data():
    """Carrega dados recentes do arquivo JSON"""
    print(f"\n{'='*80}")
    print("CARREGANDO DADOS RECENTES")
    print(f"{'='*80}")
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extrair registros de dados
    records = []
    for item in data:
        if item.get('type') == 'table' and 'data' in item:
            records = item['data']
            break
    
    df = pd.DataFrame(records)
    df['data'] = pd.to_datetime(df['data'])
    
    print(f"✓ Total de registros carregados: {len(df)}")
    print(f"✓ Período: {df['data'].min().date()} até {df['data'].max().date()}")
    print(f"\nDistribuição por tipo:")
    print(df['tipo'].value_counts())
    
    return df

def load_graph_structure():
    """Carrega estrutura do grafo e mapeamento de nós"""
    print(f"\n{'='*80}")
    print("CARREGANDO ESTRUTURA DO GRAFO")
    print(f"{'='*80}")
    
    # Carregar dados processados
    with open(PROCESSED_DATA, 'rb') as f:
        graph_data = pickle.load(f)
    
    # Carregar geojson dos nós
    import geopandas as gpd
    nodes_gdf = gpd.read_file(NODES_FILE)
    
    print(f"✓ Número de nós: {len(nodes_gdf)}")
    print(f"✓ Shape dos dados: {graph_data['node_features'].shape}")
    
    return graph_data, nodes_gdf

def map_events_to_nodes(df, nodes_gdf):
    """Mapeia eventos criminais para os nós do grafo"""
    print(f"\n{'='*80}")
    print("MAPEANDO EVENTOS PARA NÓS")
    print(f"{'='*80}")
    
    from shapely.geometry import Point
    
    # Filtrar apenas CVLIs
    cvli_df = df[df['tipo'] == 'cvli'].copy()
    print(f"✓ Total de CVLIs: {len(cvli_df)}")
    
    # Criar geometria dos eventos
    cvli_df['geometry'] = cvli_df.apply(
        lambda row: Point(float(row['longitude']), float(row['latitude'])),
        axis=1
    )
    
    # Mapear para nós (por proximidade)
    event_counts = defaultdict(lambda: defaultdict(int))
    
    for idx, event in cvli_df.iterrows():
        event_point = event['geometry']
        event_date = event['data'].date()
        
        # Encontrar nó mais próximo
        min_dist = float('inf')
        closest_node_idx = None
        
        for node_idx, node in nodes_gdf.iterrows():
            dist = event_point.distance(node['geometry'])
            if dist < min_dist:
                min_dist = dist
                closest_node_idx = node_idx
        
        if closest_node_idx is not None:
            event_counts[event_date][closest_node_idx] += 1
    
    print(f"✓ Eventos mapeados para {len(set(n for d in event_counts.values() for n in d.keys()))} nós diferentes")
    
    return event_counts, cvli_df

def prepare_validation_data(graph_data, event_counts, nodes_gdf):
    """Prepara dados para validação"""
    print(f"\n{'='*80}")
    print("PREPARANDO DADOS PARA VALIDAÇÃO")
    print(f"{'='*80}")
    
    # Pegar features históricas
    node_features = graph_data['node_features']  # shape: (num_nodes, num_timesteps, num_features)
    num_nodes = node_features.shape[0]
    
    # Pegar as últimas 30 observações para fazer predição
    X_historical = node_features[:, -HISTORY_WINDOW:, :]
    
    print(f"✓ Shape dos dados históricos: {X_historical.shape}")
    print(f"✓ Número de nós: {num_nodes}")
    
    # Preparar ground truth dos dados novos
    # Pegar todas as datas disponíveis
    all_dates = sorted(event_counts.keys())
    print(f"✓ Datas com eventos: {all_dates}")
    
    # Criar matriz de ground truth
    Y_true = np.zeros((num_nodes, len(all_dates)))
    
    for date_idx, date in enumerate(all_dates):
        for node_idx, count in event_counts[date].items():
            Y_true[node_idx, date_idx] = count
    
    print(f"✓ Shape do ground truth: {Y_true.shape}")
    print(f"✓ Total de eventos no período: {Y_true.sum()}")
    
    return X_historical, Y_true, all_dates

def load_model(model_path, graph_data, use_stgat=True):
    """Carrega modelo treinado"""
    print(f"\n{'='*80}")
    print(f"CARREGANDO MODELO {'ST-GAT' if use_stgat else 'ST-GCN'}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    if use_stgat:
        # Parâmetros do ST-GAT
        model = STGAT(
            num_nodes=graph_data['node_features'].shape[0],
            in_channels=graph_data['node_features'].shape[2],
            time_steps=HISTORY_WINDOW,
            num_classes=1,
            num_graphs=2,
            dropout=0.3
        )
    else:
        # Parâmetros do ST-GCN
        model = STGCN(
            num_nodes=graph_data['node_features'].shape[0],
            in_channels=graph_data['node_features'].shape[2],
            out_channels=1
        )
    
    # Carregar pesos
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Modelo carregado de: {model_path}")
    
    return model, device

def make_predictions(model, X_historical, graph_data, device, use_stgat=True):
    """Faz predições com o modelo"""
    print(f"\n{'='*80}")
    print("FAZENDO PREDIÇÕES")
    print(f"{'='*80}")
    
    with torch.no_grad():
        # Preparar dados
        X = torch.FloatTensor(np.transpose(X_historical, (2, 0, 1))).unsqueeze(0).to(device)
        
        print(f"✓ Shape do input: {X.shape}")
        
        if use_stgat:
            # ST-GAT usa lista de matrizes de adjacência
            # Usar matrizes geográfica e de conflito
            adj_geo = graph_data.get('adj_geo')
            adj_conflict = graph_data.get('adj_conflict')
            
            if adj_geo is None or adj_conflict is None:
                raise ValueError("Matrizes de adjacência (adj_geo, adj_conflict) não encontradas nos dados do grafo")
            
            # Normalizar matrizes de adjacência
            def normalize_adj(adj_matrix):
                adj_tensor = torch.FloatTensor(adj_matrix)
                deg = adj_tensor.sum(dim=1)
                deg_inv_sqrt = torch.pow(deg, -0.5)
                deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
                norm_adj = deg_inv_sqrt.unsqueeze(1) * adj_tensor * deg_inv_sqrt.unsqueeze(0)
                return norm_adj
            
            norm_adj_geo = normalize_adj(adj_geo)
            norm_adj_conflict = normalize_adj(adj_conflict)
            
            # Criar lista com duas matrizes
            adj_list = [norm_adj_geo.to(device), norm_adj_conflict.to(device)]
            
            # Predição
            pred = model(X, adj_list)
        else:
            # ST-GCN usa edge_index
            edge_index_t = torch.LongTensor(graph_data['edge_index']).to(device)
            pred = model(X, edge_index_t)
        
        pred = pred.squeeze().cpu().numpy()
        
        print(f"✓ Shape da predição: {pred.shape}")
        print(f"✓ Predição total de eventos: {pred.sum():.2f}")
    
    return pred

def calculate_metrics(Y_pred, Y_true, all_dates):
    """Calcula métricas de desempenho"""
    print(f"\n{'='*80}")
    print("CALCULANDO MÉTRICAS DE DESEMPENHO")
    print(f"{'='*80}")
    
    # Se temos predição para 7 dias mas só temos alguns dias de dados reais,
    # usar apenas os dias disponíveis
    num_days_available = min(Y_pred.shape[0] if len(Y_pred.shape) > 1 else 1, Y_true.shape[1])
    
    # Se Y_pred é 1D (predição agregada), expandir para comparar
    if len(Y_pred.shape) == 1:
        # Predição é agregada para todos os 7 dias
        # Comparar com soma de todos os dias disponíveis
        Y_pred_total = Y_pred
        Y_true_total = Y_true.sum(axis=1)
        
        # MAE
        mae = np.mean(np.abs(Y_pred_total - Y_true_total))
        
        # RMSE
        rmse = np.sqrt(np.mean((Y_pred_total - Y_true_total) ** 2))
        
        # MAPE (evitar divisão por zero)
        mask = Y_true_total > 0
        mape = np.mean(np.abs((Y_true_total[mask] - Y_pred_total[mask]) / Y_true_total[mask])) * 100 if mask.any() else 0
        
        # Precisão em top-k
        k = 20
        top_k_pred = np.argsort(Y_pred_total)[-k:]
        top_k_true = np.argsort(Y_true_total)[-k:]
        precision_at_k = len(set(top_k_pred) & set(top_k_true)) / k
        
        print(f"\n{'='*60}")
        print("MÉTRICAS GERAIS (Predição Agregada)")
        print(f"{'='*60}")
        print(f"MAE:                {mae:.4f}")
        print(f"RMSE:               {rmse:.4f}")
        print(f"MAPE:               {mape:.2f}%")
        print(f"Precision@{k}:       {precision_at_k:.4f}")
        print(f"\nTotal Previsto:     {Y_pred_total.sum():.2f}")
        print(f"Total Real:         {Y_true_total.sum():.0f}")
        print(f"Erro Total:         {Y_pred_total.sum() - Y_true_total.sum():.2f}")
        
    else:
        # Predição diária disponível
        # Tomar apenas os dias que temos dados reais
        Y_pred_subset = Y_pred[:num_days_available, :] if Y_pred.shape[0] > num_days_available else Y_pred
        
        # Calcular métricas
        mae_per_day = []
        rmse_per_day = []
        precision_at_k_per_day = []
        
        for day in range(num_days_available):
            pred_day = Y_pred_subset[day, :]
            true_day = Y_true[:, day]
            
            mae_day = np.mean(np.abs(pred_day - true_day))
            rmse_day = np.sqrt(np.mean((pred_day - true_day) ** 2))
            
            # Top-k
            k = 20
            top_k_pred = np.argsort(pred_day)[-k:]
            top_k_true = np.argsort(true_day)[-k:]
            p_at_k = len(set(top_k_pred) & set(top_k_true)) / k
            
            mae_per_day.append(mae_day)
            rmse_per_day.append(rmse_day)
            precision_at_k_per_day.append(p_at_k)
        
        print(f"\n{'='*60}")
        print("MÉTRICAS POR DIA")
        print(f"{'='*60}")
        for day in range(num_days_available):
            print(f"\nDia {all_dates[day]}:")
            print(f"  MAE:          {mae_per_day[day]:.4f}")
            print(f"  RMSE:         {rmse_per_day[day]:.4f}")
            print(f"  Precision@20: {precision_at_k_per_day[day]:.4f}")
            print(f"  Total Prev:   {Y_pred_subset[day, :].sum():.2f}")
            print(f"  Total Real:   {Y_true[:, day].sum():.0f}")
        
        print(f"\n{'='*60}")
        print("MÉDIAS GERAIS")
        print(f"{'='*60}")
        print(f"MAE Média:           {np.mean(mae_per_day):.4f}")
        print(f"RMSE Média:          {np.mean(rmse_per_day):.4f}")
        print(f"Precision@20 Média:  {np.mean(precision_at_k_per_day):.4f}")
    
    # Análise de distribuição espacial
    print(f"\n{'='*60}")
    print("ANÁLISE ESPACIAL")
    print(f"{'='*60}")
    
    Y_true_total = Y_true.sum(axis=1)
    Y_pred_total = Y_pred.sum(axis=0) if len(Y_pred.shape) > 1 else Y_pred
    
    nodes_with_events_true = (Y_true_total > 0).sum()
    nodes_with_events_pred = (Y_pred_total > 0.5).sum()  # threshold
    
    print(f"Nós com eventos reais:      {nodes_with_events_true}")
    print(f"Nós com eventos previstos:  {nodes_with_events_pred}")
    
    # Top nós
    print(f"\n{'='*60}")
    print("TOP 10 NÓS MAIS CRÍTICOS")
    print(f"{'='*60}")
    
    top_10_true = np.argsort(Y_true_total)[-10:][::-1]
    
    print(f"\n{'Rank':<6} {'Nó':<8} {'Real':<8} {'Previsto':<12} {'Erro':<8}")
    print("-" * 50)
    for rank, node_idx in enumerate(top_10_true, 1):
        real = Y_true_total[node_idx]
        pred = Y_pred_total[node_idx]
        erro = pred - real
        print(f"{rank:<6} {node_idx:<8} {real:<8.0f} {pred:<12.2f} {erro:<+8.2f}")
    
    return {
        'mae': np.mean(mae_per_day) if len(Y_pred.shape) > 1 else mae,
        'rmse': np.mean(rmse_per_day) if len(Y_pred.shape) > 1 else rmse,
        'precision_at_k': np.mean(precision_at_k_per_day) if len(Y_pred.shape) > 1 else precision_at_k,
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'#'*80}")
    print("VALIDAÇÃO DO MODELO COM DADOS RECENTES")
    print(f"Data de validação: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{'#'*80}")
    
    # 1. Carregar dados recentes
    df_recent = load_recent_data()
    
    # 2. Carregar estrutura do grafo
    graph_data, nodes_gdf = load_graph_structure()
    
    # 3. Mapear eventos para nós
    event_counts, cvli_df = map_events_to_nodes(df_recent, nodes_gdf)
    
    # 4. Preparar dados de validação
    X_historical, Y_true, all_dates = prepare_validation_data(graph_data, event_counts, nodes_gdf)
    
    # 5. Carregar modelo
    model_path = MODEL_STGAT if USE_STGAT else MODEL_STGCN
    if not os.path.exists(model_path):
        print(f"\n❌ ERRO: Modelo não encontrado em {model_path}")
        return
    
    model, device = load_model(model_path, graph_data, use_stgat=USE_STGAT)
    
    # 6. Fazer predições
    Y_pred = make_predictions(model, X_historical, graph_data, device, use_stgat=USE_STGAT)
    
    # 7. Calcular métricas
    metrics = calculate_metrics(Y_pred, Y_true, all_dates)
    
    # 8. Salvar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'ST-GAT' if USE_STGAT else 'ST-GCN',
        'validation_period': {
            'start': str(all_dates[0]),
            'end': str(all_dates[-1]),
            'days': len(all_dates)
        },
        'metrics': metrics,
        'total_events': {
            'predicted': float(Y_pred.sum()),
            'actual': int(Y_true.sum())
        },
        'data_summary': {
            'total_records': len(df_recent),
            'cvli_count': len(cvli_df),
            'cvp_count': len(df_recent[df_recent['tipo'] == 'cvp'])
        }
    }
    
    output_file = os.path.join(BASE_DIR, 'reports', f'validation_recent_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Resultados salvos em: {output_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
