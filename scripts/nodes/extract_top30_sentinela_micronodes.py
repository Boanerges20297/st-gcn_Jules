#!/usr/bin/env python3
"""Build dynamic micronode overlays for every plotted moderate/high/critical area."""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import app
from src.core.orchestrator import normalize_name

OUT_DIR = BASE_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MICRONODES_FILE = BASE_DIR / 'data' / 'raw' / 'inteligencia' / 'micronodos_faccoes_2026.geojson'
GEO_STREETS_FILE = BASE_DIR / 'data' / 'geo_streets_cache.json'


def ensure_runtime_ready():
    if app.orchestrator is None or app.nodes_gdf is None:
        print('🧠 Inicializando Motor de Inteligência Sentinela...')
        app.load_data_and_models()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_micro_score(score):
    return round(max(0.0, min(100.0, float(score))), 3)


def _centroid_from_geometry(geometry):
    if not isinstance(geometry, dict):
        return None, None

    geom_type = geometry.get('type')
    if geom_type == 'Polygon':
        coords = geometry.get('coordinates', [[]])[0]
    elif geom_type == 'MultiPolygon':
        coords = []
        for polygon in geometry.get('coordinates', []):
            coords.extend((polygon or [[]])[0])
    elif geom_type == 'Point':
        coords = [geometry.get('coordinates', [None, None])]
    else:
        return None, None

    coords = [point for point in coords if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not coords:
        return None, None

    lon = sum(point[0] for point in coords) / len(coords)
    lat = sum(point[1] for point in coords) / len(coords)
    return lon, lat


def _shoelace_area(ring):
    if len(ring) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(ring) - 1):
        x1, y1 = ring[idx][0], ring[idx][1]
        x2, y2 = ring[idx + 1][0], ring[idx + 1][1]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _geometry_area_proxy(geometry):
    if not isinstance(geometry, dict):
        return 0.0
    geom_type = geometry.get('type')
    if geom_type == 'Polygon':
        return _shoelace_area(geometry.get('coordinates', [[]])[0])
    if geom_type == 'MultiPolygon':
        return sum(_shoelace_area((polygon or [[]])[0]) for polygon in geometry.get('coordinates', []))
    return 0.0


def _haversine_meters(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _extract_city_from_props(props):
    micronodo = str(props.get('micronodo') or '').strip()
    if ' - ' in micronodo:
        city_raw = micronodo.split(' - ')[-1].split('/')[0].strip()
        if city_raw:
            return city_raw

    for field in ('municipio', 'municipality', 'cidade', 'city'):
        raw = str(props.get(field) or '').strip()
        if raw:
            return raw
    return ''


def _build_display_name_map():
    display_map = {}
    for _, row in app.nodes_gdf.iterrows():
        raw_name = str(row.get('name') or '').strip()
        if raw_name:
            display_map[normalize_name(raw_name)] = raw_name
    return display_map


def _resolve_parent_area(area_raw, micronodo_raw, risk_scores):
    candidates = []
    if area_raw:
        candidates.append(normalize_name(area_raw))

    city_candidate = normalize_name(_extract_city_from_props({'micronodo': micronodo_raw}))
    if city_candidate:
        candidates.append(city_candidate)

    micronodo_norm = normalize_name(micronodo_raw)
    if micronodo_norm:
        candidates.append(micronodo_norm)

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)

    for candidate in ordered:
        if candidate in risk_scores:
            return candidate, risk_scores[candidate]

    for candidate in ordered:
        for known_name, score in risk_scores.items():
            if candidate in known_name or known_name in candidate:
                return known_name, score

    return (ordered[0] if ordered else normalize_name(area_raw)), 0.0


def _build_region_node_sets():
    region_node_sets = {}
    for reg_key, spec in app.orchestrator.specialists.items():
        reg_mapped = 'capital' if reg_key == 'fortaleza' else reg_key
        region_node_sets[reg_mapped] = {
            normalize_name(str(row.get('name') or ''))
            for _, row in spec['data']['nodes_gdf'].iterrows()
        }
    return region_node_sets


def _classify_region(parent_norm, city_norm, region_node_sets):
    for reg_name, node_set in region_node_sets.items():
        if parent_norm in node_set or city_norm in node_set:
            return reg_name

    for reg_name, node_set in region_node_sets.items():
        for known_name in node_set:
            if parent_norm and (parent_norm in known_name or known_name in parent_norm):
                return reg_name
            if city_norm and (city_norm in known_name or known_name in city_norm):
                return reg_name
    return 'interior'


def _faction_bonus(raw_faction):
    faction = normalize_name(raw_faction)
    if faction in ('DISPUTA', 'COMUNIDADES EM DISPUTA'):
        return 1.0
    if faction in ('CV', 'COMANDO VERMELHO', 'TCP', 'TERCEIRO COMANDO PURO', 'PCC', 'PRIMEIRO COMANDO DA CAPITAL', 'MASSA', 'OKAIDA'):
        return 0.55
    if faction and faction not in ('', 'NEUTRO', 'N/A'):
        return 0.30
    return 0.0


def load_geo_street_points():
    if not GEO_STREETS_FILE.exists():
        return []

    with open(GEO_STREETS_FILE, 'r', encoding='utf-8') as f:
        raw_points = json.load(f) or []

    points = []
    for item in raw_points:
        lat = item.get('lat')
        lng = item.get('lng')
        if lat is None or lng is None:
            continue
        points.append({
            'street': str(item.get('rua') or item.get('street') or 'Área crítica').strip(),
            'bairro_norm': normalize_name(str(item.get('bairro') or '')),
            'municipality_norm': normalize_name(str(item.get('cidade') or item.get('municipio') or '')),
            'lat': _safe_float(lat),
            'lng': _safe_float(lng),
            'occurrences': max(1.0, _safe_float(item.get('ocorrencias', 1.0), 1.0)),
        })
    return points


def _compute_local_street_signal(node, street_points):
    lon = node.get('lon')
    lat = node.get('lat')
    if lon is None or lat is None:
        return 0.0, 0, []

    parent_norm = node.get('parent_norm', '')
    municipality_norm = node.get('municipality_norm', '')
    pressure = 0.0
    street_weights = {}

    for street in street_points:
        distance = _haversine_meters(lon, lat, street['lng'], street['lat'])
        if distance > 1800:
            continue

        if distance > 900 and municipality_norm and street['municipality_norm'] and street['municipality_norm'] != municipality_norm and street['bairro_norm'] != parent_norm:
            continue

        weight = street['occurrences'] / (1.0 + (distance / 250.0))
        if parent_norm and street['bairro_norm'] == parent_norm:
            weight *= 1.35
        elif parent_norm and street['bairro_norm'] and (street['bairro_norm'] in parent_norm or parent_norm in street['bairro_norm']):
            weight *= 1.15
        elif municipality_norm and street['municipality_norm'] == municipality_norm:
            weight *= 0.90
        elif distance > 800:
            continue

        pressure += weight
        street_weights[street['street']] = max(street_weights.get(street['street'], 0.0), weight)

    top_streets = [street for street, _ in sorted(street_weights.items(), key=lambda item: item[1], reverse=True)[:3]]
    return pressure, len(street_weights), top_streets


def get_sentinela_scores():
    ensure_runtime_ready()
    exogenous_shocks, _ = app.build_current_exogenous_shocks()
    scores_map, _ = app.orchestrator.get_combined_risk(exogenous_shocks, return_trends=True)
    if getattr(app.orchestrator, 'champion_challenger', None) is None and getattr(app, 'champion_challenger', None) is not None:
        print('⚔️ Aplicando Blend Champion/Challenger (fallback único)...')
        scores_map = app.champion_challenger.apply(scores_map)
    return {normalize_name(key): value for key, value in scores_map.items()}


def process_micronodes(risk_scores):
    print(f'📂 Lendo micronodos de {MICRONODES_FILE.name}...')
    if not MICRONODES_FILE.exists():
        raise FileNotFoundError(f'Arquivo {MICRONODES_FILE} não encontrado.')

    with open(MICRONODES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    display_name_map = _build_display_name_map()
    region_node_sets = _build_region_node_sets()
    street_points = load_geo_street_points()
    processed = []

    for feat in data.get('features', []):
        props = feat.get('properties') or {}
        geom = feat.get('geometry') or {}
        lon, lat = _centroid_from_geometry(geom)
        if lon is None or lat is None:
            continue

        micronodo_name = str(props.get('micronodo') or props.get('name') or 'DESCONHECIDO').strip()
        area_raw = str(props.get('area_oficial') or micronodo_name or 'DESCONHECIDO').strip()
        parent_norm, parent_risk = _resolve_parent_area(area_raw, micronodo_name, risk_scores)
        city_raw = _extract_city_from_props(props)
        city_norm = normalize_name(city_raw)
        region = _classify_region(parent_norm, city_norm, region_node_sets)
        if not city_raw:
            city_raw = 'Fortaleza' if region == 'capital' else display_name_map.get(parent_norm, area_raw)
            city_norm = normalize_name(city_raw)

        level, status, _, _, parent_risk = app.classify_risk_score(parent_risk)
        if level == 'baixo':
            continue

        local_pressure, nearby_streets_count, nearby_streets = _compute_local_street_signal({
            'lon': lon,
            'lat': lat,
            'parent_norm': parent_norm,
            'municipality_norm': city_norm,
        }, street_points)

        processed.append({
            'name': micronodo_name,
            'bairro': display_name_map.get(parent_norm, area_raw),
            'parent_norm': parent_norm,
            'parent_risk_score': parent_risk,
            'parent_risk_level': level,
            'parent_status': status,
            'faction': str(props.get('faction') or 'NEUTRO').strip(),
            'faction_bonus': _faction_bonus(props.get('faction')),
            'lon': lon,
            'lat': lat,
            'region': region,
            'municipality': city_raw,
            'municipality_norm': city_norm,
            'local_pressure_raw': local_pressure,
            'nearby_streets_count': nearby_streets_count,
            'nearby_streets': nearby_streets,
            'area_proxy_raw': _geometry_area_proxy(geom),
        })

    return processed


def _build_feature(node, rank, score):
    return {
        'type': 'Feature',
        'properties': {
            'node_id': rank,
            'rank': rank,
            'name': node['name'],
            'bairro': node['bairro'],
            'municipality': node['municipality'],
            'faction': node['faction'],
            'risk_score': score,
            'score': score,
            'region': node['region'],
            'parent_area': node['bairro'],
            'parent_risk_score': node['parent_risk_score'],
            'parent_risk_level': node['parent_risk_level'],
            'parent_status': node['parent_status'],
            'local_street_pressure': round(node['local_pressure_raw'], 3),
            'nearby_streets_count': node['nearby_streets_count'],
            'nearby_streets': node['nearby_streets'],
            'is_tactical': True,
            'generated_at': datetime.now().isoformat(),
        },
        'geometry': {
            'type': 'Point',
            'coordinates': [node['lon'], node['lat']]
        }
    }


def export_regional_files(processed_nodes):
    regions = {'capital': [], 'rmf': [], 'interior': []}
    for node in processed_nodes:
        regions.setdefault(node['region'], []).append(node)

    combined_features = []
    summary = {}

    for reg_name, reg_nodes in regions.items():
        local_max = max((node['local_pressure_raw'] for node in reg_nodes), default=0.0)
        area_max = max((node['area_proxy_raw'] for node in reg_nodes), default=0.0)

        for node in reg_nodes:
            local_norm = (node['local_pressure_raw'] / local_max) if local_max > 0 else 0.0
            area_norm = (node['area_proxy_raw'] / area_max) if area_max > 0 else 0.0
            micro_score = (
                (0.72 * node['parent_risk_score']) +
                (18.0 * local_norm) +
                (6.0 * node['faction_bonus']) +
                (4.0 * area_norm)
            )
            node['micro_score'] = _normalize_micro_score(micro_score)

        ranked_nodes = sorted(reg_nodes, key=lambda item: item['micro_score'], reverse=True)
        features = []
        for rank, node in enumerate(ranked_nodes, 1):
            feature = _build_feature(node, rank, node['micro_score'])
            features.append(feature)
            combined_features.append(feature)

        output_file = OUT_DIR / f'visible_micronodes_{reg_name}.geojson'
        legacy_output_file = OUT_DIR / f'top20_micro_nodes_{reg_name}.geojson'
        if reg_name == 'capital':
            output_file = OUT_DIR / 'visible_micronodes_capital.geojson'
            legacy_output_file = OUT_DIR / 'top20_micro_nodes_capital.geojson'

        payload = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'total': len(features),
                'source': 'Sentinela Micronodes Dynamic Overlay',
                'threshold_min_score': app.RISK_SCORE_THRESHOLDS['moderate_min'],
                'generated_at': datetime.now().isoformat(),
                'region': reg_name,
            }
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(legacy_output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        summary[reg_name] = len(features)
        print(f'✅ Exportados {len(features)} micronodos dinâmicos para {output_file.name}')

    combined_features = sorted(combined_features, key=lambda item: item['properties'].get('score', 0), reverse=True)
    for global_rank, feature in enumerate(combined_features, 1):
        feature['properties']['global_rank'] = global_rank

    combined_payload = {
        'type': 'FeatureCollection',
        'features': combined_features,
        'metadata': {
            'total': len(combined_features),
            'source': 'Sentinela Micronodes Dynamic Overlay',
            'threshold_min_score': app.RISK_SCORE_THRESHOLDS['moderate_min'],
            'generated_at': datetime.now().isoformat(),
            'regions': summary,
        }
    }
    with open(OUT_DIR / 'visible_micronodes.geojson', 'w', encoding='utf-8') as f:
        json.dump(combined_payload, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / 'top20_micro_nodes.geojson', 'w', encoding='utf-8') as f:
        json.dump(combined_payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Exportados {len(combined_features)} micronodos dinâmicos para visible_micronodes.geojson")
    return summary


def build_all_micronode_exports(ensure_runtime=True):
    if ensure_runtime:
        ensure_runtime_ready()
    scores = get_sentinela_scores()
    processed = process_micronodes(scores)
    return export_regional_files(processed)


if __name__ == '__main__':
    print('🚀 Iniciando extração dinâmica de micronodos visíveis...')
    try:
        build_all_micronode_exports(ensure_runtime=True)
        print('\n✨ Processo concluído com sucesso!')
    except Exception as e:
        print(f'❌ Erro fatal: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
