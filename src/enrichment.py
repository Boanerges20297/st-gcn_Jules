import os
import json
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

def get_day_of_week_pt(date_obj):
    days = {0: "Segunda-feira", 1: "Terça-feira", 2: "Quarta-feira", 3: "Quinta-feira", 4: "Sexta-feira", 5: "Sábado", 6: "Domingo"}
    return days.get(date_obj.weekday(), "")

def is_brazil_holiday(date_obj):
    holidays = {(1,1): "Ano Novo", (3,19): "São José", (3,25): "Data Magna", (4,21): "Tiradentes", (5,1): "Dia do Trabalho", (9,7): "Independência", (10,12): "Nossa Sra Aparecida", (11,2): "Finados", (11,15): "Proclamação", (12,25): "Natal", (2,16): "Carnaval", (2,17): "Carnaval", (4,3): "Sexta-feira Santa", (6,4): "Corpus Christi"}
    return (date_obj.month, date_obj.day) in holidays

def is_cvp_hot_day(date_obj):
    return 1 <= date_obj.day <= 10 or date_obj.day in [30, 31]

def get_weather_label(precip):
    if precip is None or precip < 0.1: return "Sem Chuva"
    if precip < 5: return "Nublado"
    if precip <= 20: return "Chuvoso"
    return "Chuva Intensa"

_weather_cache = {}
CACHE_FILE = 'data/weather_archive_cache.json'

def fetch_weather_from_api(lat, lon, start_date, end_date):
    """Busca dados reais da Open-Meteo API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "America/Sao_Paulo"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            daily = data.get('daily', {})
            times = daily.get('time', [])
            precips = daily.get('precipitation_sum', [])
            return dict(zip(times, precips))
    except Exception as e:
        print(f"Erro na API de Clima: {e}")
    return {}

def get_real_weather(date_obj, lat=-3.717, lon=-38.543):
    global _weather_cache
    date_str = date_obj.strftime('%Y-%m-%d')
    
    if date_str in _weather_cache:
        return _weather_cache[date_str]
    
    # Carrega cache do disco se existir
    if not _weather_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                _weather_cache.update(json.load(f))
        except: pass
    
    if date_str in _weather_cache:
        return _weather_cache[date_str]

    # Se não está no cache, buscamos um range (para eficiência)
    print(f"Buscando clima real para o período de {date_str}...")
    # Busca 30 dias ao redor para popular o cache
    start = (date_obj - timedelta(days=15)).strftime('%Y-%m-%d')
    end = (date_obj + timedelta(days=15)).strftime('%Y-%m-%d')
    
    # Limite pro futuro (não pode ser maior q hoje - 2 dias para archive)
    today = datetime.now().date()
    if date_obj >= today:
        # Para HOJE usamos forecast api
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&timezone=America/Sao_Paulo"
        try:
            res = requests.get(forecast_url).json()
            new_data = dict(zip(res['daily']['time'], res['daily']['precipitation_sum']))
            _weather_cache.update(new_data)
        except: pass
    else:
        new_data = fetch_weather_from_api(lat, lon, start, end)
        _weather_cache.update(new_data)
    
    # Salva cache atualizado
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(_weather_cache, f)
        
    return _weather_cache.get(date_str)

def enrich_event(event, base_dir=None):
    date_str = event.get('date', '')
    if not date_str: return event
    try:
        dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
        event['day_of_week'] = get_day_of_week_pt(dt)
        event['is_holiday'] = is_brazil_holiday(dt)
        event['is_cvp_hot_day'] = is_cvp_hot_day(dt)
        
        # Clima Real
        precip = get_real_weather(dt)
        event['clima'] = get_weather_label(precip)
        event['precipitation_mm'] = precip # Adiciona valor numérico para o modelo
    except Exception: pass
    return event
