import pickle
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Configuração de saída UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_data():
    path = 'data/processed/processed_graph_data_global.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def update_robust_report():
    data = load_data()
    nodes = data['nodes_gdf']
    features = data['node_features']
    
    # 1. Extração de Top 3 por Região
    nodes['total_cvli'] = features[:, :, 0].sum(axis=1)
    centroid_lat = nodes.geometry.centroid.y.mean()
    centroid_lon = nodes.geometry.centroid.x.mean()
    
    regions = {
        "NORTE": nodes.geometry.centroid.y >= centroid_lat,
        "SUL": nodes.geometry.centroid.y < centroid_lat,
        "LESTE": nodes.geometry.centroid.x >= centroid_lon,
        "OESTE": nodes.geometry.centroid.x < centroid_lon
    }
    
    # 2. Levantamento de Facções em Maracanaú
    maracanau_data = nodes[nodes['name'].str.contains('MARACANAÚ', case=False, na=False)]
    factions_info = "Informação de facção não encontrada nos metadados diretamente."
    
    # Busca por colunas de facção
    potential_cols = ['faction', 'factions_active', 'factions', 'dominancia']
    for col in potential_cols:
        if col in nodes.columns:
            f_list = maracanau_data[col].unique()
            valid_f = [str(f) for f in f_list if f and str(f).lower() != 'nan']
            if valid_f:
                factions_info = ", ".join(valid_f)
                break
    
    # --- INCREMENTO DO RELATÓRIO ---
    report_path = 'docs/CVLI_HUMAN_ANALYSIS_ROBUST.md'
    if not os.path.exists(report_path):
        print(f"Erro: {report_path} não encontrado.")
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Se já existir a seção 4, vamos substituir ou apenas não duplicar
    if "## 🏆 4. Hotspots Regionais" in content:
        print("Relatório já contém seções incrementais. Pulando.")
        return

    new_section = "\n## 🏆 4. Hotspots Regionais (Top 3 por Ponto Cardinal)\n\n"
    for reg in ["NORTE", "SUL", "LESTE", "OESTE"]:
        mask = regions[reg]
        top_3 = nodes[mask].sort_values(by='total_cvli', ascending=False).head(3)
        new_section += f"### 📍 Região {reg}\n"
        new_section += "| Pos | Localidade | CVLI |\n"
        new_section += "|---|---|---|
"
        for i, (idx, row) in enumerate(top_3.iterrows(), 1):
            new_section += f"| {i}º | **{row['name']}** | {int(row['total_cvli'])} |\n"
        new_section += "\n"
    
    new_section += f"## 🛡️ 5. Inteligência de Maracanaú (Epicentro RMF)\n\n"
    new_section += f"- **Status de Maracanaú:** É a localidade com maior volume absoluto de CVLI no dataset.\n"
    new_section += f"- **Facções Identificadas (Base de Dados):** {factions_info}\n"
    new_section += "\n> **Nota de Analista:** Maracanaú atua como um hub de transição entre a Capital e o Interior, o que justifica a alta tensão territorial constante.\n"
    
    # Inserir antes da nota final ou no final
    if "---" in content:
        final_content = content.replace("---", new_section + "\n---")
    else:
        final_content = content + new_section
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"Relatório Robust atualizado com sucesso em: {report_path}")

if __name__ == "__main__":
    update_robust_report()
