import os
import json
import logging
import math
from flask import Blueprint, jsonify, request
from src.core.orchestrator import normalize_name

logger = logging.getLogger(__name__)

def create_v4_api_blueprint(base_dir):
    """
    Cria blueprint com endpoints para inteligência granular Sentinela V4 (500m).
    """
    v4_bp = Blueprint('api_v4', __name__, url_prefix='/api/v4')

    # Caminhos de dados (assumindo tests/Sentinela como fonte atual)
    ZONES_GEOJSON_PATH = os.path.join(base_dir, 'tests', 'Sentinela', 'sentinela_v4_zones.geojson')
    RANKING_JSON_PATH = os.path.join(base_dir, 'tests', 'Sentinela', 'ranking_sentinela_v4.json')
    STREETS_CACHE_PATH = os.path.join(base_dir, 'data', 'geo_streets_cache.json')

    @v4_bp.route('/sentinella/zones', methods=['GET'])
    def get_v4_zones():
        """
        Retorna as zonas V4 com dados de risco acoplados.
        """
        try:
            region_filter = request.args.get('region', '').lower()
            tactical_bairro = request.args.get('tactical_bairro', '')
            
            # 1. Carregar Geometrias
            if not os.path.exists(ZONES_GEOJSON_PATH):
                logger.error(f"GeoJSON V4 não encontrado: {ZONES_GEOJSON_PATH}")
                return jsonify({"error": "Geometrias V4 indisponíveis"}), 404
            
            with open(ZONES_GEOJSON_PATH, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            # 2. Carregar Ranking/Risco
            ranking_map = {}
            if os.path.exists(RANKING_JSON_PATH):
                with open(RANKING_JSON_PATH, 'r', encoding='utf-8') as f:
                    ranking_data = json.load(f)
                    for item in ranking_data.get('ranking', []):
                        ranking_map[item['zone_id']] = item
            
            # 3. Merge e Filtro de Melhor Hotspot por Área
            features = geojson_data.get('features', [])
            best_by_area = {}
            
            norm_tactical = normalize_name(tactical_bairro) if tactical_bairro else None
            
            for feat in features:
                props = feat.get('properties', {})
                zone_id = props.get('zone_id')
                bairro_orig = props.get('bairro', '')
                bairro_norm = normalize_name(bairro_orig)
                
                # Acoplar dados do ranking
                rank_info = ranking_map.get(zone_id, {})
                props.update({
                    'status': rank_info.get('status', 'MODERADO'),
                    'indice_risco': rank_info.get('indice_risco', 0.0),
                    'score_hibrido': rank_info.get('indice_risco', 0.0), # V4 simplificado
                    'gat_score': rank_info.get('gap_index', 0.0),
                    'is_macro': False, # Zonas 500m são micro-zonas
                    'is_priority': rank_info.get('status') == 'CRITICO'
                })
                
                # Lógica de Restrição: Um hotspot por área (bairro ou cidade)
                # Mantemos o de maior risco.
                risk = props.get('indice_risco', 0.0)
                
                # Se for bairro tático, poderíamos mostrar mais? 
                # O usuário pediu restrição para evitar poluição, então aplicamos globalmente.
                if bairro_norm not in best_by_area or risk > best_by_area[bairro_norm]['properties']['indice_risco']:
                    best_by_area[bairro_norm] = feat

            # Converter de volta para lista e aplicar filtro de região se necessário
            reduced_features = list(best_by_area.values())
            filtered_features = []

            for feat in reduced_features:
                props = feat.get('properties', {})
                bairro_norm = normalize_name(props.get('bairro', ''))

                # Filtro de Bairro Tático (se ativo, garante que ele passe)
                if norm_tactical:
                    if norm_tactical == bairro_norm or norm_tactical in bairro_norm or bairro_norm in norm_tactical:
                        filtered_features.append(feat)
                        continue
                
                # Filtro genérico de região (pode ser expandido conforme necessidade)
                # Se não houver filtro tático, adicionamos todos os "melhores" de cada área
                filtered_features.append(feat)
                
            geojson_data['features'] = filtered_features
            logger.info(f"📍 Sentinela V4: Reduzido de {len(features)} para {len(filtered_features)} hotspots (1 por área)")
            return jsonify(geojson_data)
            
        except Exception as e:
            logger.error(f"Erro no endpoint v4/zones: {e}")
            return jsonify({"error": str(e)}), 500

    @v4_bp.route('/streets/nearby', methods=['GET'])
    def get_nearby_streets():
        """
        Retorna ruas próximas a uma coordenada.
        """
        try:
            lat = float(request.args.get('lat', 0))
            lng = float(request.args.get('lng', 0))
            radius = 0.0045 # Aproximadamente 500m em graus
            
            if not os.path.exists(STREETS_CACHE_PATH):
                return jsonify([])
                
            with open(STREETS_CACHE_PATH, 'r', encoding='utf-8') as f:
                streets = json.load(f)
                
            nearby = []
            for s in streets:
                s_lat = s.get('lat')
                s_lng = s.get('lng')
                if s_lat is None or s_lng is None: continue
                
                # Bounding Box rápido
                if abs(s_lat - lat) < radius and abs(s_lng - lng) < radius:
                    # Distância Euclidiana (suficiente para pequenas distâncias)
                    dist = math.sqrt((s_lat - lat)**2 + (s_lng - lng)**2)
                    if dist < radius:
                        nearby.append({
                            'rua': s.get('rua', 'Via não identificada'),
                            'ocorrencias': s.get('ocorrencias', 0),
                            'dist': dist
                        })
            
            # Ordenar por proximidade e pegar top 5
            nearby.sort(key=lambda x: x['dist'])
            return jsonify(nearby[:5])
            
        except Exception as e:
            logger.error(f"Erro no endpoint v4/streets: {e}")
            return jsonify({"error": str(e)}), 500

    return v4_bp
