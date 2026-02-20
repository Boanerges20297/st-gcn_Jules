import pickle
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime

def load_cvli_data():
    path = 'data/processed/processed_graph_data_global.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    features = data['node_features']
    dates = pd.to_datetime(data['dates'])
    daily_total = features[:, :, 0].sum(axis=0)
    return pd.DataFrame({'date': dates, 'cvli': daily_total})

def load_weather_data():
    # We'll use the weather cache for Fortaleza (approx -3.72, -38.57)
    # Looking for a file near Fortaleza coordinates
    cache_files = glob.glob('weather_cache/precip_-3.72_-38.57*.json')
    if not cache_files:
        # Fallback to any file in Fortaleza region
        cache_files = glob.glob('weather_cache/precip_-3.7*.json')
    
    if not cache_files:
        return None
        
    all_precip = []
    for f_path in cache_files:
        with open(f_path, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'precip': data['daily']['precipitation_sum']
            })
            all_precip.append(df)
            
    # Merge and average precipitation if multiple files
    weather_df = pd.concat(all_precip).groupby('date')['precip'].mean().reset_index()
    return weather_df

def analyze():
    cvli_df = load_cvli_data()
    weather_df = load_weather_data()
    
    if weather_df is None:
        print("No weather data found in cache.")
        return
        
    merged = pd.merge(cvli_df, weather_df, on='date', how='inner')
    
    # Analysis 1: Dry vs Rainy
    merged['is_rainy'] = merged['precip'] > 2.0 # Threshold for "significant rain"
    stats = merged.groupby('is_rainy')['cvli'].agg(['mean', 'count', 'std']).reset_index()
    
    # Analysis 2: Rain Intensity
    def rain_label(p):
        if p == 0: return "Sem Chuva"
        if p <= 5: return "Chuva Leve (<5mm)"
        if p <= 20: return "Chuva Moderada (5-20mm)"
        return "Chuva Forte (>20mm)"
        
    merged['intensity'] = merged['precip'].apply(rain_label)
    intensity_stats = merged.groupby('intensity')['cvli'].agg(['mean', 'count']).reindex([
        "Sem Chuva", "Chuva Leve (<5mm)", "Chuva Moderada (5-20mm)", "Chuva Forte (>20mm)"
    ])

    # Analysis 3: Correlation
    correlation = merged['cvli'].corr(merged['precip'])
    
    # Output to Markdown
    with open('docs/WEATHER_CVLI_CORRELATION.md', 'w', encoding='utf-8') as f:
        f.write("# 🌦️ Correlação Clima vs CVLI (Ceará)\n\n")
        f.write(f"**Período analisado:** {merged['date'].min().strftime('%d/%m/%Y')} a {merged['date'].max().strftime('%d/%m/%Y')}\n")
        f.write(f"**Fonte Clima:** Open-Meteo (Precipitação Diária em Fortaleza/RMF)\n\n")
        
        f.write("## 1. Impacto da Chuva no Volume de CVLI\n")
        f.write("| Condição | Média Diária de CVLI | Amostra (Dias) |\n")
        f.write("|---|---|---|")
        dry_mean = stats[stats['is_rainy'] == False]['mean'].values[0]
        rain_mean = stats[stats['is_rainy'] == True]['mean'].values[0]
        f.write(f"| Dias Secos (<= 2mm) | **{dry_mean:.2f}** | {int(stats[stats['is_rainy'] == False]['count'].values[0])} |\n")
        f.write(f"| Dias Chuvosos (> 2mm) | **{rain_mean:.2f}** | {int(stats[stats['is_rainy'] == True]['count'].values[0])} |\n")
        
        diff = ((rain_mean - dry_mean) / dry_mean) * 100
        f.write(f"\n**Resultado:** Dias chuvosos apresentam uma variação de **{diff:+.1f}%** no volume de CVLI em relação a dias secos.\n\n")
        
        f.write("## 2. Análise por Intensidade de Precipitação\n")
        f.write("| Intensidade | Média CVLI | Frequência |\n")
        f.write("|---|---|---|")
        for label, row in intensity_stats.iterrows():
            if pd.isna(row['mean']): continue
            f.write(f"| {label} | **{row['mean']:.2f}** | {int(row['count'])} |\n")
            
        f.write("\n## 3. Coeficiente de Correlação de Pearson\n")
        f.write(f"O coeficiente de correlação entre precipitação (mm) e CVLI é: **{correlation:.4f}**\n\n")
        
        f.write("## 4. Contexto Sociológico (Pesquisa)\n")
        f.write("- **Temperatura:** Estudos no Brasil indicam que ondas de calor extremo (5°C acima da média) aumentam homicídios em média **10,6%**, devido à irritabilidade e maior circulação de pessoas em espaços públicos.\n")
        f.write("- **Chuva:** A chuva tende a reduzir crimes contra o patrimônio (furto/roubo) por limitar a mobilidade, mas seu impacto no CVLI é mais complexo, podendo estar ligado a conflitos em áreas de risco durante desastres ou redução de policiamento ostensivo em áreas periféricas de difícil acesso.\n")
        f.write("- **El Niño/La Niña:** O Ceará enfrentou El Niño forte em 2023/2024 (seca), o que historicamente pressiona a economia rural, podendo indiretamente afetar a criminalidade por fatores econômicos secundários.\n")

    print("Relatório de correlação clima gerado em docs/WEATHER_CVLI_CORRELATION.md")

if __name__ == "__main__":
    analyze()
