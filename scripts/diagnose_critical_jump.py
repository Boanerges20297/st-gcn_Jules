#!/usr/bin/env python3
"""
Diagnóstico: Comparar dados antes/depois do filtro CVP
Investigar por que 21 áreas críticas → 145 áreas críticas
"""

import pickle
import numpy as np

print("\n" + "="*80)
print("DIAGNÓSTICO: OSCILAÇÃO DE ÁREAS CRÍTICAS")
print("="*80 + "\n")

# Carregar dados processados
with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    data = pickle.load(f)

node_features = data['node_features']  # (N, T, 3)
nodes_gdf = data['nodes_gdf']

# Filtrar apenas Fortaleza
fortaleza_mask = nodes_gdf['node_type'] == 'bairro'
fortaleza_indices = np.where(fortaleza_mask)[0]

print(f"📊 OVERVIEW:")
print(f"   • Total de nós: {len(nodes_gdf)}")
print(f"   • Nós Fortaleza (bairros): {len(fortaleza_indices)}")
print(f"   • Período: {data['dates'][0].date()} a {data['dates'][-1].date()}")
print()

# Canal 0: CVLI, Canal 1: CVP, Canal 2: Tensão
cvli_data = node_features[:, :, 0]
cvp_data = node_features[:, :, 1]

# Focar nos últimos 14 dias (janela do modelo)
last_14_days = cvli_data[:, -14:]
cvli_recent = last_14_days[fortaleza_indices, :]
cvp_recent = cvp_data[fortaleza_indices, -14:]

# Estatísticas CVLI
cvli_sum_14d = cvli_recent.sum(axis=1)
nodes_with_cvli = (cvli_sum_14d > 0).sum()
total_cvli_14d = cvli_sum_14d.sum()

print("🔴 CVLI (Últimos 14 dias - Fortaleza):")
print(f"   • Total de homicídios: {int(total_cvli_14d)}")
print(f"   • Bairros com CVLI: {nodes_with_cvli}/{len(fortaleza_indices)} ({nodes_with_cvli/len(fortaleza_indices)*100:.1f}%)")
print(f"   • Máximo em um bairro: {int(cvli_sum_14d.max())}")
print(f"   • Média por bairro: {cvli_sum_14d.mean():.2f}")
print()

# Estatísticas CVP
cvp_sum_14d = cvp_recent.sum(axis=1)
nodes_with_cvp = (cvp_sum_14d > 0).sum()
total_cvp_14d = cvp_sum_14d.sum()

print("🚗 CVP_VEÍCULOS (Últimos 14 dias - Fortaleza):")
print(f"   • Total de roubos/furtos: {int(total_cvp_14d)}")
print(f"   • Bairros com CVP: {nodes_with_cvp}/{len(fortaleza_indices)} ({nodes_with_cvp/len(fortaleza_indices)*100:.1f}%)")
print(f"   • Máximo em um bairro: {int(cvp_sum_14d.max())}")
print(f"   • Média por bairro: {cvp_sum_14d.mean():.2f}")
print()

# Distribuição de atividade
print("📊 DISTRIBUIÇÃO DE ATIVIDADE CRIMINAL (CVLI):")
cvli_ranges = [
    ("Sem eventos", (cvli_sum_14d == 0).sum()),
    ("1 evento", (cvli_sum_14d == 1).sum()),
    ("2-3 eventos", ((cvli_sum_14d >= 2) & (cvli_sum_14d <= 3)).sum()),
    ("4-5 eventos", ((cvli_sum_14d >= 4) & (cvli_sum_14d <= 5)).sum()),
    ("6+ eventos", (cvli_sum_14d >= 6).sum()),
]

for label, count in cvli_ranges:
    pct = count / len(fortaleza_indices) * 100
    print(f"   • {label}: {count} bairros ({pct:.1f}%)")
print()

# Verificar se há zeros em excesso no CVP
zero_cvp_pct = (cvp_sum_14d == 0).sum() / len(fortaleza_indices) * 100
print(f"⚠️  ALERTA: {zero_cvp_pct:.1f}% dos bairros SEM atividade CVP (veículos)")
print(f"   Isso pode indicar filtro muito restritivo!")
print()

# Top 10 bairros com mais CVLI
top10_cvli_indices = np.argsort(cvli_sum_14d)[-10:][::-1]
print("🔝 TOP 10 BAIRROS COM MAIS CVLI (14 dias):")
for rank, idx in enumerate(top10_cvli_indices, 1):
    global_idx = fortaleza_indices[idx]
    bairro = nodes_gdf.iloc[global_idx]['name']
    cvli = int(cvli_sum_14d[idx])
    cvp = int(cvp_sum_14d[idx])
    print(f"   {rank:2d}. {bairro:30s} - {cvli} CVLI | {cvp} CVP_Veículos")
print()

# Verificar toda a série temporal CVP
cvp_total_all_time = cvp_data[fortaleza_indices, :].sum()
cvli_total_all_time = cvli_data[fortaleza_indices, :].sum()

print("📈 TOTAIS HISTÓRICOS (Todo o período):")
print(f"   • CVLI Fortaleza: {int(cvli_total_all_time)}")
print(f"   • CVP_Veículos Fortaleza: {int(cvp_total_all_time)}")
print(f"   • Proporção CVP/CVLI: {cvp_total_all_time/cvli_total_all_time:.2f}x")
print()

# Se CVP for muito baixo, pode ter um problema
if cvp_total_all_time < cvli_total_all_time * 0.5:
    print("❌ PROBLEMA DETECTADO:")
    print("   CVP_Veículos está MUITO BAIXO em relação a CVLI!")
    print("   Filtro pode estar removendo dados demais.")
    print("   Modelo pode estar superestimando risco por falta de contexto CVP.")
else:
    print("✅ Proporção CVP/CVLI parece razoável")

print()
print("="*80)
