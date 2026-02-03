"""
Diagnóstico rápido do problema de estagnação do modelo
"""
import pickle
import numpy as np
import pandas as pd

print("="*60)
print("DIAGNÓSTICO DE ESTAGNAÇÃO DO MODELO")
print("="*60)

# Carregar dados
with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    data = pickle.load(f)

node_features = data['node_features']
dates = data['dates']

print(f"\n📊 ESTRUTURA DOS DADOS:")
print(f"  Shape: {node_features.shape}")  # (nodes, days, channels)
print(f"  Datas: {dates[0]} até {dates[-1]}")
print(f"  Total dias: {len(dates)}")

# Análise dos 3 canais
print(f"\n📈 ANÁLISE POR CANAL:")
for ch in range(3):
    canal_data = node_features[:, :, ch]
    print(f"\nCanal {ch}:")
    print(f"  Min: {canal_data.min():.6f}")
    print(f"  Max: {canal_data.max():.6f}")
    print(f"  Mean: {canal_data.mean():.6f}")
    print(f"  Std: {canal_data.std():.6f}")
    print(f"  Zeros: {(canal_data == 0).sum()} / {canal_data.size} ({(canal_data == 0).sum()/canal_data.size*100:.1f}%)")
    print(f"  Valores únicos: {len(np.unique(canal_data))}")

# Análise de variância temporal
print(f"\n⏰ VARIÂNCIA TEMPORAL (últimos 30 dias):")
recent = node_features[:, -30:, :]
for ch in range(3):
    var_per_day = np.var(recent[:, :, ch], axis=0)
    print(f"  Canal {ch} - Variância média por dia: {var_per_day.mean():.6f}")

# Verificar desbalanceamento CVLI
cvli = node_features[:, :, 0]
cvli_positive = (cvli > 0).sum()
cvli_total = cvli.size
print(f"\n⚖️ DESBALANCEAMENTO:")
print(f"  CVLI > 0: {cvli_positive} / {cvli_total} ({cvli_positive/cvli_total*100:.2f}%)")
print(f"  Ratio: 1:{cvli_total/cvli_positive:.1f}")

# Distribuição de valores CVLI
print(f"\n📊 DISTRIBUIÇÃO CVLI (valores > 0):")
cvli_nonzero = cvli[cvli > 0]
if len(cvli_nonzero) > 0:
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"  P{p}: {np.percentile(cvli_nonzero, p):.4f}")

# Verificar se há padrão nos últimos dias
print(f"\n🔍 ÚLTIMOS 14 DIAS (janela do modelo):")
window = node_features[:, -14:, :]
for ch in range(3):
    ch_data = window[:, :, ch]
    print(f"  Canal {ch}: sum={ch_data.sum():.2f}, mean={ch_data.mean():.6f}, max={ch_data.max():.4f}")

print(f"\n{'='*60}")
print("POSSÍVEIS PROBLEMAS:")
print("="*60)

# Diagnósticos
if node_features[:, :, 0].std() < 0.01:
    print("❌ Canal 0 (CVLI) tem variância muito baixa - dados muito homogêneos")

if (node_features == 0).sum() / node_features.size > 0.95:
    print("❌ Mais de 95% dos valores são zero - dados muito esparsos")

if len(np.unique(node_features[:, :, 0])) < 10:
    print("❌ Canal CVLI tem poucos valores únicos - pouca granularidade")

if node_features.max() <= 1.0 and node_features.min() >= 0.0:
    print("✓ Dados normalizados em [0,1]")
else:
    print("⚠️ Dados fora do range esperado [0,1]")

print("\n💡 RECOMENDAÇÕES:")
print("  1. Se variância muito baixa → dados normalizados demais")
print("  2. Se muito esparso → usar weighted loss com peso maior")
print("  3. Se poucos valores únicos → problema na agregação dos dados")
