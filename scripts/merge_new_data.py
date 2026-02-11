import json
import os
import time
import sys

# Tenta importar geopy para geolocalização
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    GEOPY_AVAILABLE = True
except ImportError:
    print("AVISO: Biblioteca 'geopy' não encontrada. A geolocalização será ignorada.")
    print("Para instalar: pip install geopy")
    GEOPY_AVAILABLE = False

# Definição de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
NEW_DATA_FILE = os.path.join(DATA_RAW_DIR, 'dados_status.json')
MAIN_DATA_FILE = os.path.join(DATA_RAW_DIR, 'dados_status_ocorrencias_gerais.json')

def load_json(filepath):
    """Carrega arquivo JSON."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler {filepath}: {e}")
        return None

def save_json(filepath, data):
    """Salva arquivo JSON."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Arquivo salvo com sucesso: {filepath}")
    except Exception as e:
        print(f"Erro ao salvar {filepath}: {e}")

def enrich_data(item):
    """
    Geolocaliza e enriquece o item com Bairro, Distrito, Lat/Lon.
    """
    if not GEOPY_AVAILABLE:
        return item

    # Se já tem Lat/Lon e Bairro, não precisa consultar
    if item.get('Latitude') and item.get('Longitude') and item.get('BairroOcor'):
        return item

    address = item.get('LocalOcor')
    city = item.get('CidadeOcor', 'Caucaia') # Default para Caucaia se não especificado
    
    if not address:
        return item

    # Limpa endereço para melhor busca (remove números de apto, etc se necessário, aqui uso simples)
    full_query = f"{address}, {city}, Ceará, Brazil"
    
    geolocator = Nominatim(user_agent="st_gcn_jules_merger")

    try:
        print(f"Geolocalizando: {full_query}...")
        location = geolocator.geocode(full_query, addressdetails=True, timeout=10)
        
        if location:
            # Atualiza coordenadas se não existirem
            if not item.get('Latitude'):
                item['Latitude'] = location.latitude
            if not item.get('Longitude'):
                item['Longitude'] = location.longitude
            
            address_details = location.raw.get('address', {})
            
            # Extrai Bairro
            if not item.get('BairroOcor'):
                bairro = address_details.get('suburb') or address_details.get('neighbourhood') or address_details.get('residential')
                if bairro:
                    item['BairroOcor'] = bairro
                    print(f" -> Bairro identificado: {bairro}")
            
            # Extrai Distrito (se houver)
            if not item.get('Distrito'):
                distrito = address_details.get('city_district')
                if distrito:
                    item['Distrito'] = distrito
                    print(f" -> Distrito identificado: {distrito}")

        else:
            print(f" -> Localização não encontrada.")
            
        # Respeita limites da API (1 req/s)
        time.sleep(1.1)

    except (GeocoderTimedOut, Exception) as e:
        print(f"Erro na geolocalização: {e}")

    return item

def main():
    print("=== Script de Merge e Enriquecimento de Dados ===")
    
    # 1. Carregar dados novos
    new_data = load_json(NEW_DATA_FILE)
    if not new_data:
        print(f"Arquivo de novos dados não encontrado ou vazio: {NEW_DATA_FILE}")
        return

    if not isinstance(new_data, list):
        new_data = [new_data]
    
    print(f"Novos registros encontrados: {len(new_data)}")

    # 2. Carregar dados existentes (Main)
    main_data = load_json(MAIN_DATA_FILE)
    if main_data is None:
        print("Arquivo principal não encontrado. Criando novo.")
        main_data = []
    
    # Lidar com estrutura complexa (header/database/table) se existir
    target_list = main_data
    if isinstance(main_data, list) and len(main_data) > 0:
        if isinstance(main_data[0], dict) and main_data[0].get('type') == 'header':
            for entry in main_data:
                if entry.get('type') == 'table' and 'data' in entry:
                    target_list = entry['data']
                    break
    
    # 3. Criar índice de duplicatas
    existing_keys = set()
    for item in target_list:
        key = item.get('Controle') or item.get('id')
        if key:
            existing_keys.add(str(key))

    # 4. Processar e Mergear
    merged_count = 0
    for item in new_data:
        key = item.get('Controle') or item.get('id')
        
        if key and str(key) in existing_keys:
            continue
            
        enriched_item = enrich_data(item)
        target_list.append(enriched_item)
        
        if key:
            existing_keys.add(str(key))
        merged_count = 1

    # 5. Salvar
    if merged_count > 0:
        save_json(MAIN_DATA_FILE, main_data)
        print(f"Sucesso! {merged_count} novos registros adicionados e enriquecidos.")
    else:
        print("Nenhum registro novo para adicionar.")

if __name__ == "__main__":
    main()
