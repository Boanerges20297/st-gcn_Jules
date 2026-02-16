#!/usr/bin/env python3
"""Extrai micro-nós de arquivos GeoJSON em data/raw/inteligencia
Gera:
 - outputs/top20_micro_nodes.geojson  (Point features - centroid)
 - outputs/top20_micro_nodes_map.html (visualização Leaflet inline)

Uso:
    py scripts/extract_top20_micro_nodes.py
"""
import os
import json
from pathlib import Path

try:
    import geopandas as gpd
    HAVE_GPD = True
except Exception:
    gpd = None
    HAVE_GPD = False

try:
    from shapely.geometry import shape, mapping
    from shapely.ops import transform
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

try:
    from pyproj import Transformer
    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

# Fallback centroid for GeoJSON-like geometries (if shapely not available)
def compute_centroid_from_geom(geom):
    t = geom.get('type')
    coords = geom.get('coordinates')
    if t == 'Point':
        return coords[0], coords[1]
    if t == 'MultiPoint':
        pts = coords
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return sum(xs)/len(xs), sum(ys)/len(ys)
    if t == 'LineString' or t == 'Polygon':
        # For Polygon, coords may be [ [ [lon,lat], ... ] , ... ]
        ring = coords[0] if t == 'Polygon' else coords
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs)/len(xs), sum(ys)/len(ys)
    if t == 'MultiPolygon':
        # take centroid of first polygon
        first = coords[0]
        ring = first[0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs)/len(xs), sum(ys)/len(ys)
    # fallback
    return None, None

BASE_DIR = Path(__file__).resolve().parents[1]
INT_DIR = BASE_DIR / 'data' / 'raw' / 'inteligencia'
OUT_DIR = BASE_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def guess_name(props):
    keys = ('name','Name','NOME','nome','NOME_BAIRRO','AREA','area','bairro','comunidade','community','title','titulo','descricao','description')
    for k in keys:
        v = props.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: nested 'properties' keys sometimes appear; try common alternatives
    for k, v in props.items():
        if isinstance(v, str) and len(v) < 200:
            return v
    # last resort
    return None  # return None so caller can build a better fallback using source/index

items = []
for fp in INT_DIR.glob('*.geojson'):
    if HAVE_GPD:
        try:
            gdf = gpd.read_file(fp)
        except Exception as e:
            print(f'Erro ao ler {fp} com geopandas: {e}. Tentando fallback JSON.')
            gdf = None
    else:
        gdf = None

    features = None
    if gdf is not None:
        # ensure crs
        try:
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
        except Exception:
            pass

        for idx, row in gdf.iterrows():
            props = dict(row)
            geom = row.geometry
            if geom is None:
                continue
            # determine geometry type
            gtype = geom.geom_type
            # compute score: polygon -> area in m2 (projected), point -> 1
            score = 1.0
            try:
                if gtype in ('Polygon','MultiPolygon'):
                    proj = gdf.to_crs(epsg=3857)
                    score = float(proj.geometry.iloc[idx].area if idx < len(proj) else proj.geometry[proj.index==idx].area.iloc[0])
                else:
                    score = 1.0
            except Exception:
                score = 1.0

            props_candidate = props.get('properties') if isinstance(props.get('properties'), dict) and props.get('properties') else props
            name = guess_name(props_candidate) or None

            # compute centroid coordinate in lon,lat
            try:
                centroid = geom.centroid
                lon, lat = float(centroid.x), float(centroid.y)
            except Exception:
                try:
                    centroid = shape(mapping(geom)).centroid
                    lon, lat = float(centroid.x), float(centroid.y)
                except Exception:
                    continue

            items.append({
                'name': name,
                'source': fp.name,
                'score': float(score),
                'geometry_type': gtype,
                'lon': lon,
                'lat': lat,
                'properties': props.get('properties') if isinstance(props.get('properties'), dict) else props
            })
    else:
        # fallback: read raw geojson and parse features with shapely
        try:
            with open(fp, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                features = data.get('features') or []
        except Exception as e:
            print(f'Erro ao ler JSON {fp}: {e}')
            continue

        for f in features:
            props = f.get('properties') or {}
            geom_json = f.get('geometry')
            if not geom_json:
                continue
            if HAVE_SHAPELY:
                try:
                    geom = shape(geom_json)
                except Exception:
                    continue
                gtype = geom.geom_type
                # score: try project to 3857 for area if possible
                score = 1.0
                try:
                    if gtype in ('Polygon','MultiPolygon') and HAVE_PYPROJ:
                        project = Transformer.from_crs('EPSG:4326','EPSG:3857', always_xy=True).transform
                        area_m2 = transform(project, geom).area
                        score = float(area_m2)
                    else:
                        score = 1.0
                except Exception:
                    score = 1.0

                name = guess_name(props) or None

                # centroid
                try:
                    c = geom.centroid
                    lon, lat = float(c.x), float(c.y)
                except Exception:
                    continue
            else:
                # fallback centroid from coordinates
                gtype = geom_json.get('type')
                lon, lat = compute_centroid_from_geom(geom_json)
                if lon is None:
                    continue
                score = 1.0
                name = guess_name(props) or None

            items.append({
                'name': name,
                'source': fp.name,
                'score': float(score),
                'geometry_type': gtype,
                'lon': lon,
                'lat': lat,
                'properties': props
            })

# deduplicate by (name, lon, lat) roughly
seen = set()
uniq = []
for it in items:
    key = (it['name'], round(it['lon'],6), round(it['lat'],6))
    if key in seen:
        continue
    seen.add(key)
    uniq.append(it)

# sort by score desc
uniq_sorted = sorted(uniq, key=lambda x: x['score'], reverse=True)

from math import radians, sin, cos, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

# classify regions: capital bbox (Fortaleza), RMF by distance, else interior
FORTALEZA_BBOX = {'lon_min': -38.66, 'lon_max': -38.40, 'lat_min': -3.90, 'lat_max': -3.65}
FORTALEZA_CENTER = (-38.5267, -3.8100)  # lon, lat
RMF_RADIUS_M = 35000  # 35 km

def classify_region(lon, lat):
    if (FORTALEZA_BBOX['lon_min'] <= lon <= FORTALEZA_BBOX['lon_max'] and
        FORTALEZA_BBOX['lat_min'] <= lat <= FORTALEZA_BBOX['lat_max']):
        return 'capital'
    d = haversine(lon, lat, FORTALEZA_CENTER[0], FORTALEZA_CENTER[1])
    if d <= RMF_RADIUS_M:
        return 'rmf'
    return 'interior'

# group by region and pick top20 each (prefer point micro-nodes)
regions = {'capital': [], 'rmf': [], 'interior': []}
for it in uniq_sorted:
    r = classify_region(it['lon'], it['lat'])
    regions[r].append(it)

out_features = []
node_counter = 1
for rname, feats in regions.items():
    # Prefer point features (micro-nodes) first, then fill with polygon-based areas if necessary
    points = [f for f in feats if f.get('geometry_type') in ('Point', 'MultiPoint')]
    polys = [f for f in feats if f.get('geometry_type') not in ('Point', 'MultiPoint')]
    top = points[:20]
    if len(top) < 20:
        top += polys[:(20 - len(top))]

    for i, it in enumerate(top):
        # ensure name exists, otherwise build from source + rank
        name = it.get('name') or f"{Path(it.get('source', 'orig')).stem} - Área {i+1}"
        # NOTE: output geometry is always a Point (centroid). Preserve original type in source_geometry_type
        feat = {
            'type': 'Feature',
            'properties': {
                'node_id': node_counter,
                'rank': i+1,
                'name': name,
                'source': it['source'],
                'score': it['score'],
                # geometry_type describes the actual geometry of the feature (Point since we use centroids)
                'geometry_type': 'Point',
                # keep original geometry type for provenance/debugging
                'source_geometry_type': it.get('geometry_type'),
                # flag to mark this was generated from a polygon centroid
                'is_centroid': (it.get('geometry_type') not in ('Point', 'MultiPoint')),
                'region': rname
            },
            'geometry': {
                'type': 'Point',
                'coordinates': [it['lon'], it['lat']]
            }
        }
        node_counter += 1
        out_features.append(feat)

# write combined geojson (all regions)
out_geo = {'type': 'FeatureCollection', 'features': out_features}
out_path = OUT_DIR / 'top20_micro_nodes.geojson'
with open(out_path, 'w', encoding='utf-8') as fh:
    json.dump(out_geo, fh, ensure_ascii=False, indent=2)

print(f'Gravado {out_path} ({len(out_features)} features)')

# create simple HTML viewer with embedded GeoJSON
html_path = OUT_DIR / 'top20_micro_nodes_map.html'
html = '''<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Top20 Micro Nodes</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>#map{{height:90vh;width:100%}}</style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var geo = __GEO__;
        var map = L.map('map').setView([-3.8,-38.53],11);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19}).addTo(map);
        function colorByRank(r){ return r<=3?'#8B0000': r<=10? '#FF6B6B' : '#FFA500'; }
        geo.features.forEach(function(f){
            var c = f.geometry.coordinates;
            var rank = f.properties.rank || 0;
            var marker = L.circleMarker([c[1],c[0]],{radius:10,fillColor:colorByRank(rank),color:'#fff',weight:1,fillOpacity:0.9}).addTo(map);
            marker.bindPopup('<strong>'+f.properties.rank+'. '+f.properties.name+'</strong><br/>' + f.properties.source + '<br/>ID: '+ (f.properties.node_id||'') + '<br/>Score: '+Math.round(f.properties.score));
        });
    </script>
</body>
</html>'''

# write per-region geojsons and build layer collections
region_collections = {'capital': [], 'rmf': [], 'interior': []}
for feat in out_features:
    r = feat['properties'].get('region', 'interior')
    region_collections[r].append(feat)

for rname, feats in region_collections.items():
    p = OUT_DIR / f'top20_micro_nodes_{rname}.geojson'
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump({'type': 'FeatureCollection', 'features': feats}, fh, ensure_ascii=False, indent=2)
    print(f'Gravado {p} ({len(feats)} features)')

# build map html with layer control
layers_js = []
for rname, feats in region_collections.items():
    varname = f"{rname}_geo"
    layers_js.append(f"var {varname} = {json.dumps({'type':'FeatureCollection','features':feats})};")

layers_code = "\n".join(layers_js)

html = '''<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Top20 Micro Nodes by Region</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>#map{{height:90vh;width:100%}}</style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([-3.8,-38.53],11);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19}).addTo(map);
        function colorByRank(r){ return r<=3?'#8B0000': r<=10? '#FF6B6B' : '#FFA500'; }

        // helper: create pin-like icon using DivIcon
        function makePin(color){
            var html = '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;">'
                + '<div style="width:14px;height:14px;background:'+color+';border-radius:7px;border:2px solid #fff;box-shadow:0 1px 2px rgba(0,0,0,0.4);"></div>'
                + '<div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid '+color+';margin-top:-2px;border-radius:1px 1px 0 0;opacity:0.95"></div>'
                + '</div>';
            return L.divIcon({html: html, className: '', iconSize: [18,22], iconAnchor: [9,20]});
        }

        // region data
        __LAYERS__
        // create layer groups
        var capitalLayer = L.layerGroup();
        var rmfLayer = L.layerGroup();
        var interiorLayer = L.layerGroup();
        function addFeaturesTo(layer, geo){
            geo.features.forEach(function(f){
                var c=f.geometry.coordinates; var rank=f.properties.rank||0; var color = colorByRank(rank);
                var icon = makePin(color);
                var m = L.marker([c[1], c[0]], {icon: icon});
                m.bindPopup('<strong>'+f.properties.rank+'. '+(f.properties.name||'Área')+'</strong><br/>'+ (f.properties.source||'') +'<br/>ID: '+ (f.properties.node_id||'') + '<br/>Score: '+Math.round(f.properties.score));
                layer.addLayer(m);
            });
        }
        addFeaturesTo(capitalLayer, capital_geo); addFeaturesTo(rmfLayer, rmf_geo); addFeaturesTo(interiorLayer, interior_geo);
        capitalLayer.addTo(map);
        var overlays = { 'Fortaleza (capital)': capitalLayer, 'RMF': rmfLayer, 'Interior': interiorLayer };
        L.control.layers(null, overlays).addTo(map);
    </script>
</body>
</html>
'''

html = html.replace('__LAYERS__', layers_code)
with open(html_path, 'w', encoding='utf-8') as fh:
    fh.write(html)

print(f'Gravado visualização em {html_path}')

print('Concluído.')
