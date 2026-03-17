import pandas as pd
import numpy as np

data = pd.read_pickle('data/processed/processed_interior.pkl')
print('=== INTERIOR DATA ANALYSIS ===')
print(f'Nodes: {len(data["nodes_gdf"])}')
print(f'Node features shape: {data["node_features"].shape}')

x = data['node_features']
crimes_30d = x[:, -30:, 0]
print(f'\nCrimes ultimos 30 dias:')
print(f'  Min: {crimes_30d.sum(axis=1).min():.0f}')
print(f'  Max: {crimes_30d.sum(axis=1).max():.0f}')
print(f'  Mean: {crimes_30d.sum(axis=1).mean():.2f}')
print(f'  Total: {crimes_30d.sum():.0f}')

crimes_7d = x[:, -7:, 0]
print(f'\nCrimes ultimos 7 dias:')
print(f'  Total: {crimes_7d.sum():.0f}')
print(f'  Nos com crime: {(crimes_7d.sum(axis=1) > 0).sum()}/{len(data["nodes_gdf"])}')

# Simular cold_streak como no treino e inferencia
print(f'\nAnalise Cold Streak (simulado):')
cold_streak = np.zeros(len(data['nodes_gdf']))
for t in range(60, x.shape[1]):
    crimes_today = x[:, t, 0]
    cold_streak = np.where(crimes_today > 0, 0, cold_streak + 1)

print(f'  Cold streak range: {cold_streak.min():.0f} to {cold_streak.max():.0f}')
print(f'  Mean cold streak: {cold_streak.mean():.2f}')
print(f'  Nos with 7+ dias frio: {(cold_streak >= 7).sum()}')
print(f'  Nos with 14+ dias frio: {(cold_streak >= 14).sum()}')
print(f'  Nos with 21+ dias frio: {(cold_streak >= 21).sum()}')

# Top nodes com maior cold_streak
top_cold_idx = np.argsort(cold_streak)[-5:]
print(f'\nTop 5 nos mais frios:')
for idx in top_cold_idx[::-1]:
    name = data['nodes_gdf'].iloc[idx]['name']
    cs = cold_streak[idx]
    print(f'  {name}: {cs:.0f} dias sem crime')
