import pickle
import pandas as pd
import numpy as np
import datetime
import os
import sys

# Configure output encoding
sys.stdout.reconfigure(encoding='utf-8')

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
    
    # Extract components
    features = data['node_features'] # (N, T, C)
    dates = pd.to_datetime(data['dates'])
    nodes = data['nodes_gdf']
    
    # CVLI is Channel 0
    cvli_matrix = features[:, :, 0] # (N, T)
    
    # --- TEMPORAL ANALYSIS ---
    # Sum across all nodes to get daily state total
    daily_total = cvli_matrix.sum(axis=0) # (T,)
    
    df = pd.DataFrame({'date': dates, 'cvli': daily_total})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    df['month_year'] = df['date'].dt.to_period('M')
    
    # 1. Total per Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    week_map = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    dow_stats = df.groupby('day_of_week')['cvli'].sum().reindex(day_order)
    
    # 2. Total per Month (Seasonality)
    month_stats = df.groupby('month')['cvli'].sum()
    month_map = {
        1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Junho',
        7: 'Julho', 8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
    }
    
    # 3. Total per Year
    year_stats = df.groupby('year')['cvli'].sum()
    
    # 4. Total per Month-Year (Trend)
    moy_stats = df.groupby('month_year')['cvli'].sum()
    
    # --- SPATIAL ANALYSIS (CENTROID) ---
    # Calculate Total CVLI per Node
    node_totals = cvli_matrix.sum(axis=1) # (N,)
    nodes['total_cvli'] = node_totals
    
    # Calculate Centroid
    centroid_lat = nodes.geometry.y.mean()
    centroid_lon = nodes.geometry.x.mean()
    
    # Classify Quadrants
    def classify_quadrant(row):
        lat = row.geometry.centroid.y
        lon = row.geometry.centroid.x
        
        is_north = lat >= centroid_lat
        is_east = lon >= centroid_lon
        
        if is_north and is_east: return "Nordeste (NE)"
        if is_north and not is_east: return "Noroeste (NW)"
        if not is_north and is_east: return "Sudeste (SE)"
        if not is_north and not is_east: return "Sudoeste (SW)"
        
    nodes['quadrant'] = nodes.apply(classify_quadrant, axis=1)
    quadrant_stats = nodes.groupby('quadrant')['total_cvli'].sum().sort_values(ascending=False)
    
    # --- WRITE REPORT ---
    os.makedirs('docs', exist_ok=True)
    with open('docs/CVLI_HUMAN_ANALYSIS.md', 'w', encoding='utf-8') as f:
        f.write("# 📊 Análise Humana de CVLI - Ceará\n\n")
        f.write(f"**Data da Análise:** {datetime.datetime.now().strftime('%d/%m/%Y')}\n")
        f.write(f"**Período:** {df['date'].min().strftime('%d/%m/%Y')} a {df['date'].max().strftime('%d/%m/%Y')}\n\n")
        
        f.write("## 1. Totais por Dia da Semana\n")
        f.write("| Dia da Semana | Total CVLI | Média Diária |\n")
        f.write("|---|---|---|
")
        for day in day_order:
            total = dow_stats.get(day, 0)
            pt_day = week_map[day]
            count = (df['day_of_week'] == day).sum()
            avg = total / count if count > 0 else 0
            f.write(f"| {pt_day} | **{int(total)}** | {avg:.1f} |\n")
        f.write("\n")
        
        f.write("## 2. Sazonalidade Mensal (Acumulado)\n")
        f.write("| Mês | Total Acumulado |\n")
        f.write("|---|---|
")
        for month_num in range(1, 13):
            total = month_stats.get(month_num, 0)
            pt_month = month_map[month_num]
            f.write(f"| {pt_month} | **{int(total)}** |\n")
        f.write("\n")
        
        f.write("## 3. Totais por Ano\n")
        f.write("| Ano | Total CVLI | Variação |\n")
        f.write("|---|---|---|
")
        prev = None
        for year, total in year_stats.items():
            delta = ""
            if prev:
                change = ((total - prev) / prev) * 100
                icon = "🔻" if change < 0 else "🔺"
                delta = f"{icon} {change:.1f}%"
            f.write(f"| {year} | **{int(total)}** | {delta} |\n")
            prev = total
        f.write("\n")
        
        f.write("## 4. Evolução Mês a Mês (Recente)\n")
        f.write("*(Últimos 12 meses do dataset)*\n\n")
        f.write("| Mês/Ano | Total CVLI |\n")
        f.write("|---|---|
")
        # Ensure we have data
        if not moy_stats.empty:
            for period in moy_stats.index[-12:]:
                total = moy_stats[period]
                f.write(f"| {period} | **{int(total)}** |\n")
        f.write("\n")
        
        f.write("## 5. Análise Geoespacial (Divisão por Centróide)\n")
        f.write(f"**Centro Geométrico do Estado (Nodes):** Lat {centroid_lat:.4f}, Lon {centroid_lon:.4f}\n\n")
        
        f.write("### Distribuição por Quadrante\n")
        f.write("| Quadrante | Posição Relativa | Total CVLI | % do Total |\n")
        f.write("|---|---|---|---|
")
        grand_total = quadrant_stats.sum()
        for quad, total in quadrant_stats.items():
            pct = (total / grand_total) * 100
            desc = ""
            if "NE" in quad: desc = "Acima e à Direita"
            if "NW" in quad: desc = "Acima e à Esquerda"
            if "SE" in quad: desc = "Abaixo e à Direita"
            if "SW" in quad: desc = "Abaixo e à Esquerda"
            
            f.write(f"| **{quad}** | {desc} | **{int(total)}** | **{pct:.1f}%** |\n")
            
        f.write("\n> **Nota:** Esta divisão é puramente geométrica baseada na média das coordenadas de todos os bairros/cidades monitorados.\n")

if __name__ == "__main__":
    generate_report()
    print("Relatório gerado em docs/CVLI_HUMAN_ANALYSIS.md")
