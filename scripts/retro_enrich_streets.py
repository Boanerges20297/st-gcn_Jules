import pandas as pd
import numpy as np
import os
import json
import unicodedata
import re
import time
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- CONFIGURAÇÕES ---
BASE_DIR = os.getcwd()
OFFICIAL_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')

# Termos que indicam que o campo de rua está "sujo" com a natureza do crime
INVALID_TERMS = ['HOMICIDIO', 'BALA', 'FOGO', 'LESAO', 'MORTE', 'CADAVER', 'LATROCINIO', 'TIRO', 'EXECUCAO', 'ACHADO']

def normalize_text(text):
    if not text or pd.isna(text): return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

# Iniciar Geocoder (Nominatim exige um user_agent único)
geolocator = Nominatim(user_agent="report_preview_retro_180d")
# RateLimiter para respeitar a política de 1 requisição por segundo do Nominatim
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1.2)

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def retro_enrich_180d():
    print(f"--- INICIANDO ENRIQUECIMENTO RETROATIVO (ÚLTIMOS 180 DIAS) ---")
    
    if not os.path.exists(OFFICIAL_CSV):
        print("❌ Arquivo CSV não encontrado.")
        return

    # 1. Carregar Dados
    df = pd.read_csv(OFFICIAL_CSV, low_memory=False)
    df['data_dt'] = pd.to_datetime(df['data'], errors='coerce')
    
    # 2. Filtrar Janela de 180 dias
    cutoff = datetime.now() - timedelta(days=180)
    mask_date = df['data_dt'] >= cutoff
    
    # 3. Identificar registros que precisam de correção
    def needs_enrichment(row):
        street = str(row.get('name', '')).upper()
        lat = pd.to_numeric(row.get('latitude'), errors='coerce')
        lon = pd.to_numeric(row.get('longitude'), errors='coerce')
        
        # Ignora se não tiver GPS válido
        if pd.isna(lat) or pd.isna(lon) or lat == 0:
            return False
            
        # Precisa se estiver vazio ou for curto demais
        if not street or len(street.strip()) < 5:
            return True
            
        # Precisa se contiver termos de natureza em vez de endereço
        if any(term in street for term in INVALID_TERMS):
            return True
            
        return False

    mask_needs = df.apply(needs_enrichment, axis=1)
    
    # Dataset de trabalho: Últimos 180 dias E que precisam de correção
    df_work = df[mask_date & mask_needs].copy()
    
    total_to_fix = len(df_work)
    print(f"Total na janela 180d: {len(df[mask_date])} registros.")
    print(f"Registros identificados para correção: {total_to_fix}")

    if total_to_fix == 0:
        print("✅ Tudo atualizado na janela selecionada.")
        return

    # 4. Processar com Cache
    geo_cache = load_cache()
    processed_count = 0
    updated_count = 0

    print(f"Iniciando geolocalização reversa (Limite: 1 req/seg)...")
    print("Pressione Ctrl+C para parar e salvar o progresso atual.")

    try:
        # Iterar do mais recente para o mais antigo
        for idx, row in df_work.sort_values('data_dt', ascending=False).iterrows():
            lat, lon = float(row['latitude']), float(row['longitude'])
            
            # Chave do cache (4 casas decimais ~11m de precisão)
            cache_key = f"{round(lat, 4)}_{round(lon, 4)}"
            
            street_found = None
            if cache_key in geo_cache:
                street_found = geo_cache[cache_key]
            else:
                try:
                    # Chamada real à API
                    location = geolocator.reverse((lat, lon), timeout=5)
                    if location:
                        addr = location.raw.get('address', {})
                        # Tenta pegar a rua, senão o bairro/subúrbio
                        street_found = addr.get('road') or addr.get('suburb') or addr.get('neighbourhood')
                        if street_found:
                            street_found = street_found.upper()
                            geo_cache[cache_key] = street_found
                    # Delay para não ser bloqueado
                    time.sleep(0.3)
                except Exception as e:
                    print(f"\n⚠️ Erro na API ({lat}, {lon}): {e}")
                    time.sleep(2)
            
            if street_found:
                df.at[idx, 'name'] = street_found
                updated_count += 1
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Progresso: {processed_count}/{total_to_fix} | Atualizados: {updated_count}")
                save_cache(geo_cache)

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Salvando progresso...")

    # 5. Finalizar e Salvar CSV
    if updated_count > 0:
        # Remover coluna temporária de data antes de salvar
        if 'data_dt' in df.columns: df = df.drop(columns=['data_dt'])
        df.to_csv(OFFICIAL_CSV, index=False, encoding='utf-8')
        print(f"✅ Sucesso! {updated_count} nomes de ruas enriquecidos no CSV.")
    else:
        print("ℹ️ Nenhum dado novo pôde ser atualizado.")

    save_cache(geo_cache)
    print(f"Cache atualizado: {len(geo_cache)} entradas.")

if __name__ == "__main__":
    retro_enrich_180d()
