import pandas as pd
import os
import sys
import json
from datetime import datetime

# Adiciona o diretório raiz ao path para importar src
sys.path.append(os.getcwd())

from src.enrichment import (
    get_day_of_week_pt, 
    is_brazil_holiday, 
    is_cvp_hot_day, 
    get_weather_label, 
    CACHE_FILE
)

def enrich_csv(input_path):
    print(f"Lendo arquivo: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    
    if 'data' not in df.columns:
        print("Erro: Coluna 'data' não encontrada no CSV.")
        return

    # Garante que temos objeto datetime e remove linhas onde a data é impossível (se houver)
    df['dt_obj'] = pd.to_datetime(df['data'], errors='coerce')
    
    # Filtra linhas com data válida para o enriquecimento
    valid_mask = df['dt_obj'].notna()
    
    print("Iniciando enriquecimento com dados climáticos reais da API...")
    
    # 1. Dia da Semana
    df.loc[valid_mask, 'dia_semana'] = df.loc[valid_mask, 'dt_obj'].apply(get_day_of_week_pt)
    
    # 2. Feriado
    df.loc[valid_mask, 'eh_feriado'] = df.loc[valid_mask, 'dt_obj'].apply(is_brazil_holiday)
    
    # 3. Dias quentes CVP
    df.loc[valid_mask, 'dia_quente_cvp'] = df.loc[valid_mask, 'dt_obj'].apply(is_cvp_hot_day)
    
    # 4. Clima Real (Via Cache Populado)
    weather_cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            weather_cache = json.load(f)
    
    def get_precip(dt):
        if pd.isna(dt): return 0.0
        d_str = dt.strftime('%Y-%m-%d')
        return weather_cache.get(d_str, 0.0)

    df.loc[valid_mask, 'precipitacao_mm'] = df.loc[valid_mask, 'dt_obj'].apply(get_precip)
    df.loc[valid_mask, 'clima'] = df.loc[valid_mask, 'precipitacao_mm'].apply(get_weather_label)
    
    # Remove coluna auxiliar e salva
    df = df.drop(columns=['dt_obj'])
    
    # Preenche gaps se houver (em colunas novas)
    df['dia_semana'] = df['dia_semana'].fillna('')
    df['eh_feriado'] = df['eh_feriado'].fillna(False)
    df['dia_quente_cvp'] = df['dia_quente_cvp'].fillna(False)
    df['clima'] = df['clima'].fillna('Desconhecido')
    df['precipitacao_mm'] = df['precipitacao_mm'].fillna(0.0)
    
    output_path = input_path
    df.to_csv(output_path, index=False)
    print(f"Enriquecimento com API concluído! Arquivo salvo em: {output_path}")
    print(f"Total de registros processados: {len(df)}")

if __name__ == "__main__":
    target = r'data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    enrich_csv(target)
