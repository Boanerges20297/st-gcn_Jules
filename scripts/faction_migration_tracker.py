import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime

def track_faction_migration(geojson_path, history_path):
    print(f"--- Rastreando migração de facções em {geojson_path} ---")
    
    if not os.path.exists(geojson_path):
        print(f"Erro: Arquivo GeoJSON não encontrado: {geojson_path}")
        return

    # 1. Carregar estado atual
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    current_state = {}
    for feat in data.get('features', []):
        props = feat.get('properties', {})
        node_id = props.get('id') or props.get('node_id') or props.get('micronodo')
        faction = props.get('faccao') or props.get('faccao_predominante') or 'NEUTRO'
        name = props.get('name') or props.get('area_oficial') or 'Desconhecido'
        municipio = props.get('municipio') or 'Desconhecido'
        
        if node_id:
            current_state[str(node_id)] = {
                'faction': faction.upper(),
                'name': name,
                'municipio': municipio
            }

    # 2. Carregar histórico anterior
    history = []
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    
    # Pegar o último snapshot conhecido
    last_snapshot = {}
    if history:
        # Reconstruir o último estado a partir dos logs ou buscar o último campo 'snapshot'
        # Para simplificar, vamos salvar um arquivo de 'ultimo_estado.json' separadamente ou ler o último log
        last_state_path = history_path.replace('.json', '_last_state.json')
        if os.path.exists(last_state_path):
            with open(last_state_path, 'r', encoding='utf-8') as f:
                last_snapshot = json.load(f)

    # 3. Comparar e Logar Mudanças
    changes = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for node_id, info in current_state.items():
        prev_info = last_snapshot.get(node_id)
        if prev_info and prev_info['faction'] != info['faction']:
            change = {
                'timestamp': timestamp,
                'node_id': node_id,
                'name': info['name'],
                'municipio': info['municipio'],
                'from': prev_info['faction'],
                'to': info['faction'],
                'type': 'migration'
            }
            changes.append(change)
            print(f"⚠️ MUDANÇA DETECTADA: {info['name']} ({info['municipio']}) -> {prev_info['faction']} para {info['to']}")

    # 4. Salvar Histórico e Novo Estado
    if changes:
        history.extend(changes)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    # Sempre atualiza o último estado para a próxima comparação
    last_state_path = history_path.replace('.json', '_last_state.json')
    with open(last_state_path, 'w', encoding='utf-8') as f:
        json.dump(current_state, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Rastreamento concluído. {len(changes)} novas migrações registradas.")

if __name__ == "__main__":
    GEO_PATH = os.path.join('data', 'raw', 'inteligencia', 'micronodos_faccoes_2026.geojson')
    HIST_PATH = os.path.join('data', 'faction_migration_history.json')
    track_faction_migration(GEO_PATH, HIST_PATH)
