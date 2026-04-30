import pandas as pd
import json
import os

# Caminhos dos arquivos
JSON_FILE = 'data/raw/dados_status.json'
CSV_FILE = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'

def inject_ids():
    print(f"🚀 Iniciando injeção de IDs de {JSON_FILE} para {CSV_FILE}")
    
    if not os.path.exists(JSON_FILE) or not os.path.exists(CSV_FILE):
        print("❌ Erro: Arquivos necessários não encontrados.")
        return

    # 1. Carregar JSON (ignorar cabeçalhos do phpMyAdmin se houver)
    print("⏳ Carregando JSON...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        try:
            data_json = json.load(f)
            # Encontrar a lista de dados (geralmente o último elemento ou o que tem 'data')
            if isinstance(data_json, list):
                # Se for o formato exportado pelo phpMyAdmin, os dados estão após os metadados
                records = []
                for item in data_json:
                    if isinstance(item, dict) and 'data' in item and isinstance(item['data'], list):
                        records.extend(item['data'])
                    elif isinstance(item, dict) and 'id' in item:
                        records.append(item)
            else:
                records = data_json.get('data', [])
        except Exception as e:
            print(f"❌ Erro ao ler JSON: {e}")
            return

    df_json = pd.DataFrame(records)
    print(f"✅ JSON carregado: {len(df_json)} registros.")

    # Normalizar chaves para o merge
    df_json['data'] = pd.to_datetime(df_json['data']).dt.strftime('%Y-%m-%d')
    df_json['hora'] = df_json['hora'].astype(str)
    df_json['lat_join'] = pd.to_numeric(df_json['latitude'], errors='coerce').round(5)
    df_json['lng_join'] = pd.to_numeric(df_json['longitude'], errors='coerce').round(5)
    df_json['id_new'] = df_json['id'] # Guardar o ID correto

    # 2. Carregar CSV
    print("⏳ Carregando CSV Enriquecido...")
    df_csv = pd.read_csv(CSV_FILE, low_memory=False)
    print(f"✅ CSV carregado: {len(df_csv)} registros.")

    # Normalizar CSV
    df_csv['data_join'] = pd.to_datetime(df_csv['data']).dt.strftime('%Y-%m-%d')
    df_csv['hora_join'] = df_csv['hora'].astype(str)
    df_csv['lat_join'] = pd.to_numeric(df_csv['latitude'], errors='coerce').round(5)
    df_csv['lng_join'] = pd.to_numeric(df_csv['longitude'], errors='coerce').round(5)

    # 3. Merge (Mapeamento de ID)
    print("🔄 Cruzando dados para recuperação de IDs...")
    # Chave: data, hora, lat, lng, tipo
    # Nota: Alguns registros podem não ter hora exata, mas data/lat/lng/tipo costuma ser único o suficiente
    df_merged = pd.merge(
        df_csv, 
        df_json[['data', 'hora', 'lat_join', 'lng_join', 'tipo', 'id_new']], 
        left_on=['data_join', 'hora_join', 'lat_join', 'lng_join', 'tipo'],
        right_on=['data', 'hora', 'lat_join', 'lng_join', 'tipo'],
        how='left',
        suffixes=('', '_from_json')
    )

    # Atualizar a coluna 'id' onde estiver nula ou onde encontramos um match melhor
    count_before = df_merged['id'].isna().sum()
    df_merged['id'] = df_merged['id_new'].combine_first(df_merged['id'])
    count_after = df_merged['id'].isna().sum()
    
    print(f"📊 Recuperação concluída:")
    print(f"   - IDs nulos antes: {count_before}")
    print(f"   - IDs nulos depois: {count_after}")
    print(f"   - IDs injetados: {count_before - count_after}")

    # Limpar colunas auxiliares
    cols_to_drop = ['data_join', 'hora_join', 'lat_join', 'lng_join', 'id_new', 'data_from_json', 'hora_from_json']
    df_merged = df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns])

    # 4. Salvar
    print(f"💾 Salvando arquivo atualizado...")
    df_merged.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print("✅ Concluído com sucesso!")

if __name__ == "__main__":
    inject_ids()
