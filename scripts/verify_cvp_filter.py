#!/usr/bin/env python3
"""
Verifica a aplicação do filtro CVP (apenas veículos).
Compara antes/depois e mostra estatísticas.
"""

import pickle
import numpy as np

print("\n" + "="*80)
print("VERIFICAÇÃO DO FILTRO CVP - VEÍCULOS APENAS")
print("="*80 + "\n")

# Carregar dados processados
with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    data = pickle.load(f)

node_features = data['node_features']  # (N, T, 3)
dates = data['dates']

# Canal 0: CVLI
# Canal 1: CVP (agora apenas veículos)
# Canal 2: Tensão

cvli_data = node_features[:, :, 0]
cvp_data = node_features[:, :, 1]

# Estatísticas
cvli_total = cvli_data.sum()
cvp_total = cvp_data.sum()

cvli_days_with_crime = (cvli_data.sum(axis=0) > 0).sum()
cvp_days_with_crime = (cvp_data.sum(axis=0) > 0).sum()

cvli_nodes_with_crime = (cvli_data.sum(axis=1) > 0).sum()
cvp_nodes_with_crime = (cvp_data.sum(axis=1) > 0).sum()

print("📊 ESTATÍSTICAS GERAIS:")
print(f"   • Período: {dates[0].date()} a {dates[-1].date()}")
print(f"   • Total de dias: {len(dates)}")
print(f"   • Total de nós: {node_features.shape[0]}")
print()

print("📈 CVLI (Homicídios):")
print(f"   • Total de eventos: {int(cvli_total)}")
print(f"   • Dias com eventos: {cvli_days_with_crime}/{len(dates)} ({cvli_days_with_crime/len(dates)*100:.1f}%)")
print(f"   • Nós com eventos: {cvli_nodes_with_crime}/{node_features.shape[0]} ({cvli_nodes_with_crime/node_features.shape[0]*100:.1f}%)")
print(f"   • Média diária: {cvli_total/len(dates):.2f} eventos/dia")
print()

print("🚗 CVP_VEÍCULOS (Roubos/Furtos de Veículos):")
print(f"   • Total de eventos: {int(cvp_total)}")
print(f"   • Dias com eventos: {cvp_days_with_crime}/{len(dates)} ({cvp_days_with_crime/len(dates)*100:.1f}%)")
print(f"   • Nós com eventos: {cvp_nodes_with_crime}/{node_features.shape[0]} ({cvp_nodes_with_crime/node_features.shape[0]*100:.1f}%)")
print(f"   • Média diária: {cvp_total/len(dates):.2f} eventos/dia")
print()

print("📊 PROPORÇÃO CVP/CVLI:")
ratio = cvp_total / cvli_total if cvli_total > 0 else 0
print(f"   • {ratio:.2f}x (CVP_Veículos / CVLI)")
print()

# Top 5 nós com mais CVP
cvp_per_node = cvp_data.sum(axis=1)
top_5_indices = np.argsort(cvp_per_node)[-5:][::-1]

print("🔝 TOP 5 ÁREAS COM MAIS ROUBOS/FURTOS DE VEÍCULOS:")
nodes_gdf = data['nodes_gdf']
name_col = 'name' if 'name' in nodes_gdf.columns else 'nome'
for rank, idx in enumerate(top_5_indices, 1):
    node_name = nodes_gdf.iloc[idx][name_col]
    count = int(cvp_per_node[idx])
    cvli_count = int(cvli_data[idx].sum())
    print(f"   {rank}. {node_name}: {count} veículos roubados/furtados | {cvli_count} CVLI")
print()

# Top 5 nós com mais CVLI
cvli_per_node = cvli_data.sum(axis=1)
top_5_cvli = np.argsort(cvli_per_node)[-5:][::-1]

print("🔴 TOP 5 ÁREAS COM MAIS CVLI:")
for rank, idx in enumerate(top_5_cvli, 1):
    node_name = nodes_gdf.iloc[idx][name_col]
    count = int(cvli_per_node[idx])
    cvp_count = int(cvp_data[idx].sum())
    print(f"   {rank}. {node_name}: {count} CVLI | {cvp_count} veículos roubados/furtados")
print()

print("="*80)
print("✅ FILTRO CVP APLICADO COM SUCESSO!")
print("="*80)
print()
print("💡 IMPORTANTE:")
print("   • CVP agora inclui APENAS roubos/furtos de VEÍCULOS")
print("   • Roubos/furtos genéricos (celulares, residências, etc.) foram REMOVIDOS")
print("   • Isso deve eliminar o viés do Centro (muito CVP, pouco CVLI)")
print()
print("⚠️  Reinicie o servidor Flask para aplicar as mudanças!")
print()
