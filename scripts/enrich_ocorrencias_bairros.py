import json
from shapely.geometry import Point, shape
import os

# Caminhos dos arquivos
DADOS_PATH = os.path.join('data', 'raw', 'dados_status_ocorrencias_gerais.json')
BAIRROS_PATH = os.path.join('data', 'static', 'bairros_2025.geojson')
LIM_CEARA_PATH = os.path.join('data', 'raw', 'limites_ceara_ibge_completo.geojson')

# Carregar bairros (Fortaleza)
def load_bairros():
    with open(BAIRROS_PATH, 'r', encoding='utf-8') as f:
        bairros = json.load(f)['features']
    return [
        {
            'name': feat['properties'].get('bairro') or feat['properties'].get('Name'),
            'geometry': shape(feat['geometry'])
        }
        for feat in bairros
    ]

# Carregar distritos/cidades (Ceará)
def load_distritos():
    with open(LIM_CEARA_PATH, 'r', encoding='utf-8') as f:
        distritos = json.load(f)['features']
    return [
        {
            'name': feat['properties'].get('name') or feat['properties'].get('codarea'),
            'geometry': shape(feat['geometry'])
        }
        for feat in distritos
    ]

# Enriquecer ocorrências
def enrich_ocorrencias():
    with open(DADOS_PATH, 'r', encoding='utf-8') as f:
        ocorrencias = json.load(f)

    bairros = load_bairros()
    distritos = load_distritos()

    for oc in ocorrencias:
        try:
            lat = float(oc['latitude'])
            lon = float(oc['longitude'])
            pt = Point(lon, lat)
            bairro_found = None
            distrito_found = None

            # Buscar bairro (Fortaleza)
            for bairro in bairros:
                if bairro['geometry'].contains(pt):
                    bairro_found = bairro['name']
                    break

            # Buscar distrito/cidade (Interior)
            if not bairro_found:
                for distrito in distritos:
                    if distrito['geometry'].contains(pt):
                        distrito_found = distrito['name']
                        break

            oc['bairro_enriquecido'] = bairro_found
            oc['distrito_enriquecido'] = distrito_found
        except Exception:
            oc['bairro_enriquecido'] = None
            oc['distrito_enriquecido'] = None

    # Salvar arquivo enriquecido
    enriched_path = os.path.join('data', 'raw', 'dados_status_ocorrencias_gerais_enriquecido.json')
    with open(enriched_path, 'w', encoding='utf-8') as f:
        json.dump(ocorrencias, f, ensure_ascii=False, indent=2)
    print(f'Arquivo enriquecido salvo em: {enriched_path}')

if __name__ == '__main__':
    enrich_ocorrencias()
