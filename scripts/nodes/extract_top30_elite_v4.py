import os
import json
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_elite_v4():
    """
    Extrai o Top 30 de Micronodos da Inteligência V4 (Elite P10 - 500m).
    """
    base_dir = os.getcwd()
    
    # Caminhos de entrada
    ZONES_PATH = os.path.join(base_dir, 'tests', 'Sentinela', 'sentinela_v4_zones.geojson')
    RANKING_PATH = os.path.join(base_dir, 'tests', 'Sentinela', 'ranking_sentinela_v4.json')
    
    # Caminhos de saída (compatibilidade com o dashboard de screenshot)
    OUTPUT_PATH = os.path.join(base_dir, 'outputs', 'top30_elite_p10.geojson')
    # Package oficial para envio
    STATIC_EXPORT_PATH = os.path.join(base_dir, 'static_export', 'data', 'top30_elite_p10.geojson')
    # Backup tático
    STATIC_EXPORT_TACTICAL_PATH = os.path.join(base_dir, 'static_export', 'data_tactical', 'top30_elite_p10.geojson')

    if not os.path.exists(ZONES_PATH) or not os.path.exists(RANKING_PATH):
        logger.error("Arquivos do Sentinela V4 não encontrados.")
        return

    # 1. Carregar Geometrias
    with open(ZONES_PATH, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    # Criar mapa de zone_id -> geometry
    geom_map = {}
    for feat in geojson_data.get('features', []):
        zid = feat['properties'].get('zone_id')
        if zid:
            geom_map[zid] = feat['geometry']

    # 2. Carregar Ranking
    with open(RANKING_PATH, 'r', encoding='utf-8') as f:
        ranking_data = json.load(f)
    
    ranking_list = ranking_data.get('ranking', [])
    # O ranking já deve estar ordenado, mas vamos garantir e pegar os Top 30
    ranking_list.sort(key=lambda x: x.get('indice_risco', 0), reverse=True)
    top_30 = ranking_list[:30]

    # 3. Construir Novo GeoJSON
    output_features = []
    for rank_item in top_30:
        zone_id = rank_item.get('zone_id')
        if zone_id in geom_map:
            # Enriquecer propriedades para o dashboard
            properties = {
                'id': zone_id,
                'bairro': rank_item.get('bairro', 'N/A'),
                'indice_risco': rank_item.get('indice_risco', 0.0),
                'rank': rank_item.get('rank'),
                'status': rank_item.get('status', 'MODERADO'),
                'natureza': rank_item.get('natureza_critica', 'CVLI'),
                'gap_index': rank_item.get('gap_index', 0.0),
                'is_macro': False,
                'is_elite': True,
                'raio': '500m'
            }
            
            feat = {
                'type': 'Feature',
                'properties': properties,
                'geometry': geom_map[zone_id]
            }
            output_features.append(feat)

    output_geojson = {
        'type': 'FeatureCollection',
        'features': output_features,
        'metadata': {
            'total': len(output_features),
            'source': 'Sentinela V4 (Elite P10)',
            'timestamp': ranking_data.get('timestamp')
        }
    }

    # 4. Salvar
    def save_json(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Salvo em: {path}")

    save_json(OUTPUT_PATH, output_geojson)
    save_json(STATIC_EXPORT_PATH, output_geojson)
    save_json(STATIC_EXPORT_TACTICAL_PATH, output_geojson)

if __name__ == "__main__":
    extract_elite_v4()
