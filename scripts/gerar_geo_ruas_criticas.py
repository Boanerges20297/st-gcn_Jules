import pandas as pd
import json
import os
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

def generate_geo_streets():
    path_occ = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    path_micronodes = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\inteligencia\micronodos_faccoes_2026.csv'
    output_path = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\geo_streets_cache.json'
    
    streets_data = []
    cache_coords = {}

    # 1. Carregar cache existente se houver para não reprocessar o geocoding demorado
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                streets_data = json.load(f)
                cache_coords = {(round(float(c['lat']), 3), round(float(c['lng']), 3)): c for c in streets_data}
                print(f"♻️ Cache existente carregado com {len(streets_data)} registros.")
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache: {e}")

    # 2. Carregar Micronodos da Inteligência (Alta Prioridade)
    if os.path.exists(path_micronodes):
        print(f"🔍 Carregando micronodos de {path_micronodes}...")
        try:
            df_micro = pd.read_csv(path_micronodes, low_memory=False)
            for _, row in df_micro.iterrows():
                try:
                    lat, lng = float(row['lat']), float(row['long'])
                    name = str(row['micronodo']).upper()
                    bairro = str(row['area_oficial']).upper()
                    faction = str(row['faction']).upper()
                    
                    lat_r, lng_r = round(lat, 3), round(lng, 3)
                    
                    if (lat_r, lng_r) not in cache_coords:
                        entry = {
                            'rua': name,
                            'bairro': bairro,
                            'cidade': '', # Deixar vazio ou extrair do nome se possível
                            'lat': lat_r,
                            'lng': lng_r,
                            'ocorrencias': 10, # Forçar criticidade alta para micronodos de inteligência
                            'faction': faction,
                            'source': 'intelligence'
                        }
                        streets_data.append(entry)
                        cache_coords[(lat_r, lng_r)] = entry
                except: continue
            print(f"✅ Micronodos processados. Total no cache: {len(streets_data)}")
            # Salvar imediatamente após processar micronodos
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(streets_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"❌ Erro ao processar micronodos: {e}")

    # 3. Processar CVLIs do CSV para Geocoding Reversa
    if os.path.exists(path_occ):
        print("📖 Carregando dados de CVLI para agrupamento geográfico...")
        df = pd.read_csv(path_occ, usecols=['cidade', 'bairro', 'tipo', 'latitude', 'longitude', 'data'], low_memory=False)
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        
        # Filtrar apenas CVLIs na janela de 14 dias
        df_cvli = df[df['tipo'].str.lower() == 'cvli'].copy()
        cutoff = df_cvli['data'].max() - pd.Timedelta(days=14)
        df_cvli = df_cvli[df_cvli['data'] >= cutoff]
        print(f"⏱️ Janela: últimos 14 dias ({cutoff.date()} a {df_cvli['data'].max().date()}) — {len(df_cvli)} CVLIs.")
        df_cvli['latitude'] = pd.to_numeric(df_cvli['latitude'], errors='coerce')
        df_cvli['longitude'] = pd.to_numeric(df_cvli['longitude'], errors='coerce')
        df_cvli = df_cvli.dropna(subset=['latitude', 'longitude'])
        
        # Arredondar para 3 casas decimais
        df_cvli['lat_r'] = df_cvli['latitude'].round(3)
        df_cvli['lng_r'] = df_cvli['longitude'].round(3)
        
        cluster_counts = df_cvli.groupby(['lat_r', 'lng_r']).size().reset_index(name='ocorrencias')
        critical_clusters = cluster_counts[cluster_counts['ocorrencias'] >= 1].sort_values('ocorrencias', ascending=False)
        
        print(f"⚠️ Identificados {len(critical_clusters)} clusters CVLI (>= 1 ocorrência).")
        
        # Import geopy
        try:
            from geopy.geocoders import Nominatim
        except ImportError:
            print("❌ Biblioteca geopy não encontrada.")
            return

        geolocator = Nominatim(user_agent="report_preview_geo_intel")
        
        count_processed = 0
        total = len(critical_clusters)
        new_geocoded = 0
        
        print("🌐 Iniciando geocodificação reversa para novos pontos...")
        for _, row in critical_clusters.iterrows():
            lat, lng = row['lat_r'], row['lng_r']
            count = row['ocorrencias']
            
            if (lat, lng) in cache_coords:
                # Se já existe (seja por geocoding anterior ou micronodo), atualiza ocorrências
                cache_coords[(lat, lng)]['ocorrencias'] = max(int(count), cache_coords[(lat, lng)].get('ocorrencias', 0))
                count_processed += 1
                continue
                
            # Limitar geocodificação para evitar bloqueio (só novos pontos)
            if new_geocoded > 2000: # Limite por execução para não travar
                print("⏸️ Limite de 2000 novos geocodings atingido nesta execução.")
                break

            try:
                time.sleep(1)
                location = geolocator.reverse(f"{lat}, {lng}", exactly_one=True)
                if location and location.raw and 'address' in location.raw:
                    addr = location.raw['address']
                    rua = addr.get('road', addr.get('pedestrian', addr.get('path', addr.get('suburb', ''))))
                    bairro = addr.get('suburb', addr.get('neighbourhood', addr.get('city_district', '')))
                    cidade = addr.get('city', addr.get('town', addr.get('municipality', '')))
                    
                    if rua and len(rua) > 2:
                        entry = {
                            'rua': rua.upper(),
                            'bairro': bairro.upper() if bairro else '',
                            'cidade': cidade.upper() if cidade else '',
                            'lat': lat,
                            'lng': lng,
                            'ocorrencias': int(count),
                            'source': 'geocoding'
                        }
                        streets_data.append(entry)
                        cache_coords[(lat, lng)] = entry
                        new_geocoded += 1
                        print(f"[{count_processed+1}/{total}] 📍 Mapeado: {rua} ({bairro}) -> {count}")
                count_processed += 1
                
                if new_geocoded % 50 == 0 and new_geocoded > 0:
                    streets_data.sort(key=lambda x: x.get('ocorrencias', 0), reverse=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(streets_data, f, ensure_ascii=False, indent=4)
                    print(f"💾 Salvamento intermediário: {len(streets_data)} registros.")
            except Exception as e:
                print(f"❌ Erro em {lat}, {lng}: {e}")
                time.sleep(2)

    # Ordenar por ocorrências
    streets_data.sort(key=lambda x: x.get('ocorrencias', 0), reverse=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(streets_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Cache consolidado final salvo com {len(streets_data)} localidades.")

if __name__ == "__main__":
    generate_geo_streets()
