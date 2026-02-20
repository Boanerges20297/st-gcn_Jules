import pickle
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Configure standard output to UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_data():
    path = 'data/processed/processed_graph_data_global.pkl'
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        sys.exit(1)
        
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def generate_report():
    data = load_data()
    
    # Extract data components
    # node_features: (N, T, C) where C=29, C[0]=CVLI
    features = data['node_features'] 
    dates = pd.to_datetime(data['dates'])
    nodes = data['nodes_gdf']
    
    # CVLI data (Channel 0)
    cvli_matrix = features[:, :, 0] # (N, T)
    
    # 1. TEMPORAL ANALYSIS
    # Total CVLI by day across all nodes
    daily_total = cvli_matrix.sum(axis=0) # (T,)
    
    df_temp = pd.DataFrame({'date': dates, 'cvli': daily_total})
    df_temp['year'] = df_temp['date'].dt.year
    df_temp['month'] = df_temp['date'].dt.month
    df_temp['day_name'] = df_temp['date'].dt.day_name()
    df_temp['month_year'] = df_temp['date'].dt.to_period('M')
    
    # Totals by Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pt_days = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    dow_stats = df_temp.groupby('day_name')['cvli'].sum().reindex(day_order)
    
    # Totals by Month (Sazonalidade)
    pt_months = {
        1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Junho',
        7: 'Julho', 8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
    }
    month_stats = df_temp.groupby('month')['cvli'].sum()
    
    # Totals by Year
    year_stats = df_temp.groupby('year')['cvli'].sum()
    
    # Totals by Month-Year (History)
    history_stats = df_temp.groupby('month_year')['cvli'].sum()
    
    # 2. SPATIAL ANALYSIS
    # Calculate Total CVLI per Node (over all time)
    node_totals = cvli_matrix.sum(axis=1) # (N,)
    nodes['total_cvli'] = node_totals
    
    # Geometric Centroid (mean of coordinates)
    centroid_lat = nodes.geometry.centroid.y.mean()
    centroid_lon = nodes.geometry.centroid.x.mean()
    
    # Classify Cardinal Regions (using Latitude and Longitude comparison)
    # North/South based on Latitude, East/West based on Longitude
    # Note: A node can be both North and East (NE).
    
    norte_mask = nodes.geometry.centroid.y >= centroid_lat
    sul_mask = nodes.geometry.centroid.y < centroid_lat
    leste_mask = nodes.geometry.centroid.x >= centroid_lon
    oeste_mask = nodes.geometry.centroid.x < centroid_lon
    
    regional_totals = {
        "Norte (Acima do Centróide)": nodes[norte_mask]['total_cvli'].sum(),
        "Sul (Abaixo do Centróide)": nodes[sul_mask]['total_cvli'].sum(),
        "Leste (Direita do Centróide)": nodes[leste_mask]['total_cvli'].sum(),
        "Oeste (Esquerda do Centróide)": nodes[oeste_mask]['total_cvli'].sum()
    }
    
    # Detailed Quadrants
    ne_mask = norte_mask & leste_mask
    nw_mask = norte_mask & oeste_mask
    se_mask = sul_mask & leste_mask
    sw_mask = sul_mask & oeste_mask
    
    quadrant_totals = {
        "NORDÊSTE (NE)": nodes[ne_mask]['total_cvli'].sum(),
        "NORDOESTE (NW)": nodes[nw_mask]['total_cvli'].sum(),
        "SUDESTE (SE)": nodes[se_mask]['total_cvli'].sum(),
        "SUDOESTE (SW)": nodes[sw_mask]['total_cvli'].sum()
    }
    
    # --- GENERATE MARKDOWN CONTENT ---
    lines = []
    lines.append("# 📊 Relatório Estratégico CVLI - Análise Humana")
    lines.append(f"\n**Data de Geração:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    lines.append(f"**Dataset Period:** {dates.min().strftime('%d/%m/%Y')} até {dates.max().strftime('%d/%m/%Y')}")
    lines.append(f"**Total Geral de CVLI:** {int(daily_total.sum())}")
    
    lines.append("\n## 📅 1. Análise Temporal: O 'Quando'")
    
    lines.append("\n### 🗓️ 1.1 Total CVLI por Dia da Semana")
    lines.append("| Dia da Semana | Total CVLI | Média Diária |")
    lines.append("|---|---|---|")
    for day in day_order:
        total = int(dow_stats[day])
        pt_day = pt_days[day]
        count = (df_temp['day_name'] == day).sum()
        avg = total / count if count > 0 else 0
        lines.append(f"| {pt_day} | **{total}** | {avg:.2f} |")
        
    lines.append("\n### 🌙 1.2 Sazonalidade: Total por Mês")
    lines.append("| Mês | Total Acumulado |")
    lines.append("|---|---|")
    for month_num in range(1, 13):
        total = int(month_stats.get(month_num, 0))
        pt_month = pt_months[month_num]
        lines.append(f"| {pt_month} | **{total}** |")
        
    lines.append("\n### 📈 1.3 Histórico: Total por Ano")
    lines.append("| Ano | Total CVLI | Variação (%) |")
    lines.append("|---|---|---|")
    prev_total = None
    for year in sorted(year_stats.index):
        total = int(year_stats[year])
        variation = "-"
        if prev_total is not None and prev_total > 0:
            change = ((total - prev_total) / prev_total) * 100
            icon = "🔺" if change > 0 else "🔻"
            variation = f"{icon} {change:+.1f}%"
        lines.append(f"| {year} | **{total}** | {variation} |")
        prev_total = total
        
    lines.append("\n### 🎞️ 1.4 Séries Históricas (Últimos 24 Meses)")
    lines.append("| Mês/Ano | Total CVLI |")
    lines.append("|---|---|")
    # Show last 24 months
    for period in history_stats.index[-24:]:
        total = int(history_stats[period])
        lines.append(f"| {period} | **{total}** |")
        
    lines.append("\n## 📍 2. Análise Espacial: O 'Onde'")
    lines.append(f"\n**Centróide Geométrico da Rede:** Lat {centroid_lat:.4f}, Lon {centroid_lon:.4f}")
    
    lines.append("\n### 🧭 2.1 Visão Macro: Pontos Cardeais")
    lines.append("*Nota: As regiões Norte/Sul e Leste/Oeste se sobrepõem (ex: um bairro no Norte também é Leste ou Oeste).*")
    lines.append("\n| Região Cardinal | Descrição | Total CVLI |")
    lines.append("|---|---|---|")
    for region, total in regional_totals.items():
        lines.append(f"| **{region}** | Em relação ao centro | **{int(total)}** |")
        
    lines.append("\n### 🗺️ 2.2 Visão Granular: Quadrantes do Estado")
    lines.append("| Quadrante | Posição | Total CVLI | % do Total |")
    lines.append("|---|---|---|---|")
    grand_total = sum(quadrant_totals.values())
    for quad, total in quadrant_totals.items():
        pct = (total / grand_total * 100) if grand_total > 0 else 0
        lines.append(f"| **{quad}** | Norte/Sul e Leste/Oeste | **{int(total)}** | {pct:.1f}% |")
    
    lines.append("\n---")
    lines.append("\n*Relatório gerado automaticamente para análise humana.*")
    
    # Save to file
    os.makedirs('docs', exist_ok=True)
    report_path = 'docs/CVLI_HUMAN_ANALYSIS_ROBUST.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    print(f"Relatório gerado em: {report_path}")

if __name__ == "__main__":
    generate_report()
