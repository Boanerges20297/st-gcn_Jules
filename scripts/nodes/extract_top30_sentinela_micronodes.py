#!/usr/bin/env python3
"""
Extract top 30 micro-nodes using Sentinela Risk Scores
Integrates directly with the predictive pipeline to map neighborhood risk to tactical micro-polygons.
"""
import os
import json
import sys
import unicodedata
from pathlib import Path
from datetime import datetime

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Importar componentes do app para consistência
import app
from src.core.orchestrator import normalize_name

# Configurações de saída
OUT_DIR = BASE_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MICRONODES_FILE = BASE_DIR / 'data' / 'raw' / 'inteligencia' / 'micronodos_faccoes_2026.geojson'

def get_sentinela_scores():
    """Obtém o ranking de risco consolidado do Sentinela (Champion/Challenger Blend)"""
    print("🧠 Inicializando Motor de Inteligência Sentinela...")
    app.load_data_and_models()
    
    # Simula o fluxo do app.py para obter o blend champion/challenger
    exogenous_shocks = None # Simplificado para extração estática
    scores_map, _ = app.orchestrator.get_combined_risk(exogenous_shocks, return_trends=True)
    
    if app.champion_challenger is not None:
        print("⚔️ Aplicando Blend Champion/Challenger (Sentinela V3)...")
        scores_map = app.champion_challenger.apply(scores_map)
    
    # Normalizar nomes para o match
    normalized_scores = {normalize_name(k): v for k, v in scores_map.items()}
    return normalized_scores

def process_micronodes(risk_scores):
    """Processa o arquivo de micronodos e associa os scores de risco"""
    print(f"📂 Lendo micronodos de {MICRONODES_FILE.name}...")
    if not MICRONODES_FILE.exists():
        print(f"❌ Erro: Arquivo {MICRONODES_FILE} não encontrado.")
        return []

    with open(MICRONODES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get('features', [])
    processed = []
    
    for feat in features:
        props = feat.get('properties', {})
        geom = feat.get('geometry', {})
        
        # Identificar o bairro/área para o match do score
        area_raw = props.get('area_oficial') or props.get('micronodo') or "DESCONHECIDO"
        area_norm = normalize_name(area_raw)
        
        # Obter score do bairro (default 0 se não encontrado)
        risk_score = risk_scores.get(area_norm, 0.0)
        
        # Se for 0, tenta um match parcial (muito comum em nomes complexos)
        if risk_score == 0.0:
            for b_name, b_score in risk_scores.items():
                if b_name in area_norm or area_norm in b_name:
                    risk_score = b_score
                    break
        
        # Calcular centroide simples para o GeoJSON de pontos (dashboard compat)
        if geom.get('type') == 'Polygon':
            coords = geom['coordinates'][0]
            lon = sum(p[0] for p in coords) / len(coords)
            lat = sum(p[1] for p in coords) / len(coords)
        elif geom.get('type') == 'MultiPolygon':
            all_coords = []
            for poly in geom['coordinates']:
                all_coords.extend(poly[0])
            lon = sum(p[0] for p in all_coords) / len(all_coords)
            lat = sum(p[1] for p in all_coords) / len(all_coords)
        else:
            continue

        processed.append({
            'name': props.get('micronodo', area_raw),
            'bairro': area_raw,
            'faction': props.get('faction', 'NEUTRO'),
            'risk_score': float(risk_score),
            'lon': lon,
            'lat': lat,
            'geometry': geom, # Mantém a geometria original para o cache do app
            'original_props': props
        })
    
    return processed

def export_regional_files(processed_nodes):
    """Filtra, ordena e exporta os arquivos por região"""
    # Mapeamento de regionais para Fortaleza (simplificado ou via app)
    # Aqui usaremos a classificação de região do app se disponível ou heurística
    
    # Ordenar por score de risco (descendente)
    sorted_nodes = sorted(processed_nodes, key=lambda x: x['risk_score'], reverse=True)
    
    # Como o objetivo é o Top 30 solicitado pelo usuário:
    # Vamos gerar os arquivos top20_micro_nodes_*.geojson (mesmo nome para não quebrar o app)
    # Mas com 30 itens.
    
    regions = {
        'capital': [],
        'rmf': [],
        'interior': []
    }
    
    # Separar por região
    for node in sorted_nodes:
        # Tenta identificar região pelo bairro no app
        # No app, temos _RMF_NODES para distinguir.
        name_norm = normalize_name(node['bairro'])
        if name_norm in app._RMF_NODES:
            reg = 'rmf'
        elif node['risk_score'] > 0: # Se tem score no ranking de Fortaleza
             # Nota: O orquestrador carrega rankings separados. 
             # Para simplificar, se estiver no risk_scores obtido, classificamos conforme a origem.
             reg = 'capital' # Default para este script focado em Fortaleza
        else:
            reg = 'interior'
        
        regions[reg].append(node)

    for reg_name, reg_nodes in regions.items():
        # Pega o Top 30
        top_30 = reg_nodes[:30]
        
        features = []
        for rank, node in enumerate(top_30, 1):
            feat = {
                'type': 'Feature',
                'properties': {
                    'node_id': rank,
                    'rank': rank,
                    'name': node['name'],
                    'bairro': node['bairro'],
                    'faction': node['faction'],
                    'risk_score': app.normalize_risk_score(node['risk_score']),
                    'score': app.normalize_risk_score(node['risk_score']), # Alias
                    'region': reg_name,
                    'is_tactical': True,
                    'generated_at': datetime.now().isoformat()
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [node['lon'], node['lat']]
                }
            }
            features.append(feat)
            
        output_file = OUT_DIR / f'top20_micro_nodes_{reg_name}.geojson'
        # Fallback para o nome global se for capital
        if reg_name == 'capital':
            output_file = OUT_DIR / 'top20_micro_nodes_capital.geojson'
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'type': 'FeatureCollection', 'features': features}, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Exportado Top {len(features)} micronodos para {output_file.name}")

if __name__ == "__main__":
    print("🚀 Iniciando extração de Inteligência Tática (Top 30 Micronodos)...")
    
    try:
        # 1. Pegar scores do Sentinela
        scores = get_sentinela_scores()
        
        # 2. Processar micronodos
        processed = process_micronodes(scores)
        
        # 3. Exportar
        export_regional_files(processed)
        
        print("\n✨ Processo concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
