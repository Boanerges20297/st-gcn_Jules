import pickle
import pandas as pd
import numpy as np
import os
import sys

# Configuração de saída UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_data():
    path = 'data/processed/processed_graph_data_global.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_top_by_cardinal():
    data = load_data()
    nodes = data['nodes_gdf']
    features = data['node_features']
    
    # CVLI total por nó (Canal 0)
    nodes['total_cvli'] = features[:, :, 0].sum(axis=1)
    
    # Centróide Geométrico
    centroid_lat = nodes.geometry.centroid.y.mean()
    centroid_lon = nodes.geometry.centroid.x.mean()
    
    # Filtros Cardinais
    masks = {
        "NORTE": nodes.geometry.centroid.y >= centroid_lat,
        "SUL": nodes.geometry.centroid.y < centroid_lat,
        "LESTE": nodes.geometry.centroid.x >= centroid_lon,
        "OESTE": nodes.geometry.centroid.x < centroid_lon
    }
    
    results = {}
    for region, mask in masks.items():
        # Pega os Top 3 de cada região
        top_3 = nodes[mask].sort_values(by='total_cvli', ascending=False).head(3)
        results[region] = top_3[['name', 'total_cvli']]

    # Gerar Saída Markdown
    lines = []
    lines.append("# 🏆 Top 3 Localidades por Região Cardinal (Visão Estratégica)\n")
    lines.append(f"**Referência Central (Centróide):** Lat {centroid_lat:.4f}, Lon {centroid_lon:.4f}\n")
    lines.append("Este relatório identifica os 'Hotspots' absolutos em cada direção do estado.\n")
    
    for region, df in results.items():
        lines.append(f"### 📍 Região {region}")
        lines.append("| Posição | Localidade (Município/Bairro) | Total CVLI (Período) |")
        lines.append("|---|---|---|")
        for i, (idx, row) in enumerate(df.iterrows(), 1):
            lines.append(f"| {i}º | **{row['name']}** | {int(row['total_cvli'])} |")
        lines.append("")

    report_path = 'docs/TOP3_REGIONAL_HOTSPOTS.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"Relatório gerado em: {report_path}")

if __name__ == "__main__":
    analyze_top_by_cardinal()
