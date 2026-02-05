"""
Script para processar novos dados, adicionar bairros via geocoding reverso,
mergear com base existente, filtrar período de treino e retreinar o modelo.

Workflow:
1. Carrega novos dados (dados_status_020226.json)
2. Adiciona bairros usando lat/long quando ausente
3. Mergeia com dados_status_ocorrencias_gerais.json
4. Filtra período 2024-2025 para treinamento
5. Executa retreino do modelo
"""

import json
import os
import sys
import subprocess
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminhos dos arquivos
NEW_DATA_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_020226.json')
MERGED_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais.json')
BAIRROS_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'AIS - CAPITAL.geojson')
BACKUP_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'backups')

def load_bairros_polygons():
    """Carrega polígonos de bairros de Fortaleza."""
    if not os.path.exists(BAIRROS_FILE):
        print(f"⚠️  Arquivo de bairros não encontrado: {BAIRROS_FILE}")
        return None
    
    try:
        gdf = gpd.read_file(BAIRROS_FILE)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        print(f"✓ Carregados {len(gdf)} polígonos de bairros")
        return gdf
    except Exception as e:
        print(f"❌ Erro ao carregar bairros: {e}")
        return None


def geocode_bairro(lat, lon, bairros_gdf):
    """Encontra o bairro baseado em lat/long usando polígonos."""
    if bairros_gdf is None or lat is None or lon is None:
        return None
    
    try:
        lat = float(lat)
        lon = float(lon)
        point = Point(lon, lat)
        
        # Busca espacial
        possible_matches = bairros_gdf[bairros_gdf.geometry.contains(point)]
        
        if len(possible_matches) > 0:
            # Retorna o primeiro match (pode haver sobreposições)
            for col in ['nome', 'name', 'NOME', 'NAME', 'bairro', 'BAIRRO']:
                if col in possible_matches.columns:
                    bairro = possible_matches.iloc[0][col]
                    if pd.notna(bairro) and str(bairro).strip():
                        return str(bairro).strip()
        
        return None
    except Exception as e:
        print(f"⚠️  Erro no geocoding: {e}")
        return None


def extract_occurrences_from_export(data):
    """Extrai ocorrências do formato de export do PHPMyAdmin."""
    occurrences = []
    
    for item in data:
        if isinstance(item, dict) and item.get('type') == 'table':
            table_data = item.get('data', [])
            if isinstance(table_data, list):
                occurrences.extend(table_data)
    
    return occurrences


def process_new_data(bairros_gdf):
    """Processa novos dados adicionando bairros onde faltam."""
    
    print(f"\n{'='*60}")
    print("STEP 1: Processando novos dados")
    print(f"{'='*60}\n")
    
    if not os.path.exists(NEW_DATA_FILE):
        print(f"❌ Arquivo não encontrado: {NEW_DATA_FILE}")
        return []
    
    with open(NEW_DATA_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extrai ocorrências do formato de export
    new_occurrences = extract_occurrences_from_export(raw_data)
    print(f"✓ Carregadas {len(new_occurrences)} novas ocorrências")
    
    # Adiciona bairros onde faltam
    updated_count = 0
    for occ in new_occurrences:
        if not occ.get('bairro') or occ['bairro'] == 'null' or not str(occ['bairro']).strip():
            lat = occ.get('latitude')
            lon = occ.get('longitude')
            
            if lat and lon:
                bairro = geocode_bairro(lat, lon, bairros_gdf)
                if bairro:
                    occ['bairro'] = bairro
                    updated_count += 1
                else:
                    # Fallback: usa cidade
                    occ['bairro'] = occ.get('cidade', 'Desconhecido')
    
    print(f"✓ Atualizados {updated_count} registros com bairros via geocoding")
    return new_occurrences


def merge_with_existing(new_occurrences):
    """Mergeia novos dados com base existente."""
    
    print(f"\n{'='*60}")
    print("STEP 2: Mergeando com base existente")
    print(f"{'='*60}\n")
    
    # Cria backup
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    if os.path.exists(MERGED_FILE):
        backup_path = os.path.join(BACKUP_DIR, f'dados_status_backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(MERGED_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Backup criado: {backup_path}")
        print(f"✓ Base existente: {len(existing_data)} registros")
    else:
        existing_data = []
        print("⚠️  Nenhuma base existente encontrada. Criando nova.")
    
    # Remove duplicatas por ID
    existing_ids = {str(occ.get('id')) for occ in existing_data}
    new_unique = [occ for occ in new_occurrences if str(occ.get('id')) not in existing_ids]
    
    print(f"✓ Novos registros únicos: {len(new_unique)}")
    
    # Merge
    merged_data = existing_data + new_unique
    
    # Ordena apenas registros válidos (evita erro com diferentes tipos)
    try:
        merged_data.sort(key=lambda x: (str(x.get('data', '')), str(x.get('hora', ''))))
    except Exception as e:
        print(f"⚠️  Não foi possível ordenar registros: {e}")
    
    # Salva
    with open(MERGED_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Total após merge: {len(merged_data)} registros")
    print(f"✓ Salvo em: {MERGED_FILE}")
    
    return merged_data


def filter_training_period(merged_data):
    """Filtra dados do período 2025-2026."""
    
    print(f"\n{'='*60}")
    print("STEP 3: Filtrando período de treinamento (2025-2026)")
    print(f"{'='*60}\n")
    
    # Filtra por data
    filtered = []
    for occ in merged_data:
        data_str = occ.get('data', '')
        if data_str:
            try:
                year = int(data_str[:4])
                if 2025 <= year <= 2026:
                    filtered.append(occ)
            except:
                pass
    
    print(f"✓ Registros 2025-2026: {len(filtered)}")
    
    # Estatísticas
    df = pd.DataFrame(filtered)
    if len(df) > 0:
        print(f"\n📊 Estatísticas do período de treino:")
        print(f"   - Período: {df['data'].min()} até {df['data'].max()}")
        print(f"   - Total CVLI: {len(df[df['tipo'] == 'cvli'])}")
        print(f"   - Cidades: {df['cidade'].nunique()}")
        print(f"   - Bairros únicos: {df['bairro'].nunique()}")
    
    return filtered


def retrain_model():
    """Executa o script de retreinamento do modelo."""
    
    print(f"\n{'='*60}")
    print("STEP 4: Retreinando modelo")
    print(f"{'='*60}\n")
    
    # Primeiro reprocessa os dados
    print("▶ Reprocessando dados...")
    data_processing_path = os.path.join(BASE_DIR, 'src', 'data_processing.py')
    
    result = subprocess.run(
        [sys.executable, data_processing_path],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )
    
    if result.returncode != 0:
        print(f"❌ Erro no processamento de dados:")
        print(result.stderr)
        return False
    
    print("✓ Dados reprocessados com sucesso!")
    
    # Agora treina o modelo
    print("\n▶ Iniciando treinamento do modelo...")
    train_path = os.path.join(BASE_DIR, 'src', 'train.py')
    
    result = subprocess.run(
        [sys.executable, train_path],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )
    
    if result.returncode != 0:
        print(f"❌ Erro no treinamento:")
        print(result.stderr)
        return False
    
    print("✓ Modelo retreinado com sucesso!")
    print(result.stdout)
    
    return True


def main():
    """Função principal."""
    
    print(f"\n{'#'*60}")
    print("# MERGE DE DADOS E RETREINAMENTO DE MODELO")
    print(f"{'#'*60}\n")
    
    # Carrega polígonos de bairros
    bairros_gdf = load_bairros_polygons()
    
    # 1. Processa novos dados
    new_occurrences = process_new_data(bairros_gdf)
    
    if not new_occurrences:
        print("❌ Nenhum dado novo para processar. Abortando.")
        return
    
    # 2. Mergeia com existente
    merged_data = merge_with_existing(new_occurrences)
    
    # 3. Filtra período de treino
    training_data = filter_training_period(merged_data)
    
    # 4. Retreina modelo
    print("\n🤔 Deseja retreinar o modelo agora? (s/n): ", end='')
    response = input().strip().lower()
    
    if response == 's':
        success = retrain_model()
        if success:
            print(f"\n{'='*60}")
            print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("⚠️  Merge concluído, mas erro no retreinamento.")
            print(f"{'='*60}\n")
    else:
        print("\n⏭️  Retreinamento pulado. Execute manualmente quando necessário:")
        print("   python src/data_processing.py")
        print("   python src/train.py")


if __name__ == '__main__':
    main()
