import pandas as pd
import json
import os
from datetime import datetime, timedelta

def generate_geo_streets_dynamic():
    # Caminhos oficiais do projeto
    path_occ = r'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    path_micronodes = r'data/raw/inteligencia/micronodos_faccoes_2026.csv'
    output_path = r'data/geo_streets_cache.json'
    
    if not os.path.exists(path_occ):
        print(f"❌ Arquivo de ocorrências não encontrado em {path_occ}")
        return

    print("📖 Lendo base ENRIQUECIDA para extração dinâmica (30 dias)...")
    
    # 1. Carregar dados brutos
    df = pd.read_csv(path_occ, low_memory=False)
    
    # 2. Filtrar apenas CVLIs e tratar datas
    df['tipo'] = df['tipo'].astype(str).str.lower()
    df_cvli_all = df[df['tipo'] == 'cvli'].copy()
    df_cvli_all['data'] = pd.to_datetime(df_cvli_all['data'], errors='coerce')
    df_cvli_all = df_cvli_all.dropna(subset=['latitude', 'longitude', 'data'])
    
    # Normalizar nomes de ruas e bairros (remover nulos e tratar como string)
    df_cvli_all['RuaUpper'] = df_cvli_all['name'].fillna('AREA SEM NOME').astype(str).str.upper().str.strip()
    df_cvli_all['BairroUpper'] = df_cvli_all['bairro'].fillna('DESCONHECIDO').astype(str).str.upper().str.strip()

    # 3. Calcular janela de 30 dias com base na data mais recente do arquivo
    max_date = df_cvli_all['data'].max()
    cutoff_date = max_date - timedelta(days=30)
    print(f"⏱️ Janela Dinâmica: {cutoff_date.date()} até {max_date.date()} (Últimos 30 dias)")

    # 4. Agrupar TODAS as ruas históricas para manter o mapa completo
    historical_grouped = df_cvli_all.groupby(['BairroUpper', 'RuaUpper']).agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    # 5. Filtrar e contar apenas crimes da janela recente (30 dias)
    df_recent = df_cvli_all[df_cvli_all['data'] >= cutoff_date]
    recent_counts = df_recent.groupby(['BairroUpper', 'RuaUpper']).size().reset_index(name='recent_count')

    # 6. Merge das informações: Coordenada média histórica + Contagem recente
    merged = pd.merge(historical_grouped, recent_counts, on=['BairroUpper', 'RuaUpper'], how='left').fillna(0)
    
    streets_data = []
    processed_keys = set()

    for _, row in merged.iterrows():
        key = (row['BairroUpper'], row['RuaUpper'])
        entry = {
            'rua': row['RuaUpper'],
            'bairro': row['BairroUpper'],
            'cidade': 'FORTALEZA',
            'lat': round(float(row['latitude']), 5),
            'lng': round(float(row['longitude']), 5),
            'ocorrencias': int(row['recent_count']), # Contagem dos últimos 30 dias
            'source': 'crime_data'
        }
        streets_data.append(entry)
        processed_keys.add(key)

    print(f"✅ {len(streets_data)} ruas históricas processadas ({len(recent_counts)} com crimes recentes).")

    # 7. Integrar Micronodos de Inteligência
    if os.path.exists(path_micronodes):
        print("🔍 Sincronizando micronodos de inteligência...")
        df_micro = pd.read_csv(path_micronodes, low_memory=False)
        for _, row in df_micro.iterrows():
            r_name = str(row['micronodo']).upper().strip()
            b_name = str(row['area_oficial']).upper().strip()
            key = (b_name, r_name)
            
            if key not in processed_keys:
                entry = {
                    'rua': r_name,
                    'bairro': b_name,
                    'cidade': 'FORTALEZA',
                    'lat': round(float(row['lat']), 5),
                    'lng': round(float(row['long']), 5),
                    'ocorrencias': 0, 
                    'faction': str(row['faction']).upper(),
                    'source': 'intelligence'
                }
                streets_data.append(entry)
                processed_keys.add(key)
            else:
                # Se já existe por crime, anexa a facção
                for s in streets_data:
                    if s['rua'] == r_name and s['bairro'] == b_name:
                        s['faction'] = str(row['faction']).upper()
                        s['source'] = 'hybrid'
                        break

    # 8. Ordenação: Crimes recentes primeiro
    streets_data.sort(key=lambda x: x['ocorrencias'], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(streets_data, f, ensure_ascii=False, indent=4)
        
    print(f"🚀 Mapeamento Dinâmico (30 dias) concluído: {len(streets_data)} registros salvos.")

if __name__ == "__main__":
    generate_geo_streets_dynamic()
