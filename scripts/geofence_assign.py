#!/usr/bin/env python3
"""Assign occurrence points to bairro polygons and produce aggregates.

Outputs written to `outputs/occurrences_with_bairro_geo.csv`,
`outputs/bairro_daily_counts.csv`, and `outputs/fortaleza_bairros_fence.geojson`.
"""
import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from shapely.geometry import shape, Point, mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_STATIC = PROJECT_ROOT / 'data' / 'static'
OUTPUTS = PROJECT_ROOT / 'outputs'
OUTPUTS.mkdir(exist_ok=True)


def normalize(s):
    if s is None:
        return ''
    return ''.join(c for c in str(s).upper() if c.isalnum() or c.isspace()).strip()


def load_polygons():
    # Prefer city/neighborhood polygons from available geojsons
    candidates = [
        DATA_RAW / 'limites_ceara_ibge_completo.geojson',
        DATA_RAW / 'limites_ceara.geojson',
        DATA_RAW / 'limites_ceara_ibge_simples.geojson',
    ]
    polys = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            jb = json.load(open(p, 'r', encoding='utf-8'))
        except Exception:
            continue
        feats = jb.get('features') or []
        for f in feats:
            props = f.get('properties', {})
            name = props.get('name') or props.get('NAME') or props.get('bairro') or props.get('Bairro') or props.get('NOME')
            geom = f.get('geometry')
            if not geom:
                continue
            try:
                sh = shape(geom)
            except Exception:
                continue
            polys.append({'name': normalize(name), 'props': props, 'geom': sh})
    return polys


def load_occurrences():
    # Try several sources that contain explicit lat/lon
    rows = []
    # 1) data_with_coordinates.js (contains array-like JSON)
    jsf = DATA_RAW / 'data_with_coordinates.js'
    if jsf.exists():
        txt = jsf.read_text(encoding='utf-8')
        # attempt to extract first '[' ... last ']'
        start = txt.find('[')
        end = txt.rfind(']')
        if start != -1 and end != -1:
            arr = json.loads(txt[start:end+1])
            for r in arr:
                lat = r.get('latitude') or r.get('lat') or r.get('Latitude')
                lon = r.get('longitude') or r.get('lon') or r.get('Longitude')
                if lat is None or lon is None:
                    continue
                rows.append({
                    'source': 'data_with_coordinates.js',
                    'date': r.get('Data') or r.get('data') or r.get('DataOcorrencia') or r.get('DataHora'),
                    'lat': float(lat) if lat not in (None, '') else None,
                    'lon': float(lon) if lon not in (None, '') else None,
                    'bairro_text': r.get('BairroOcor') or r.get('BairroAbord') or r.get('Bairro') or r.get('BairroOcurrencia'),
                    'raw': r,
                })
    # 2) dados_status_1201_2701.json or ocurrences JSON
    js2 = DATA_RAW / 'dados_status_1201_2701.json'
    if js2.exists():
        arr = json.load(open(js2, 'r', encoding='utf-8'))
        for r in arr:
            lat = r.get('latitude') or r.get('lat') or r.get('LATITUDE')
            lon = r.get('longitude') or r.get('lon') or r.get('LONGITUDE')
            if lat in (None, '') or lon in (None, ''):
                continue
            try:
                rows.append({
                    'source': 'dados_status_1201_2701.json',
                    'date': r.get('Data') or r.get('data'),
                    'lat': float(lat),
                    'lon': float(lon),
                    'bairro_text': r.get('BairroOcor') or r.get('BairroAbord') or r.get('Bairro'),
                    'raw': r,
                })
            except Exception:
                continue
    # 3) enriched CSV with lat/long columns
    csvf = DATA_RAW / 'View_Ocorrencias_2022_ENRIQUECIDO.csv'
    if csvf.exists():
        df = pd.read_csv(csvf, dtype=str, encoding='utf-8', low_memory=False)
        for _, r in df.iterrows():
            lat = r.get('latitude') if 'latitude' in r.index else r.get('lat') if 'lat' in r.index else r.get('lat_long')
            lon = r.get('longitude') if 'longitude' in r.index else r.get('lon') if 'lon' in r.index else None
            if pd.isna(lat) or pd.isna(lon) or lat in (None, ''):
                # some csvs have lat_long as '-3.7,-38.5'
                ll = r.get('lat_long') if 'lat_long' in r.index else None
                if isinstance(ll, str) and ',' in ll:
                    try:
                        lat_s, lon_s = ll.split(',')
                        lat = float(lat_s); lon = float(lon_s)
                    except Exception:
                        lat = None
            try:
                if lat is None or lon is None:
                    continue
                rows.append({
                    'source': 'View_Ocorrencias_2022_ENRIQUECIDO.csv',
                    'date': r.get('Data') or r.get('data') or r.get('DataOcorrencia'),
                    'lat': float(lat),
                    'lon': float(lon),
                    'bairro_text': r.get('BairroOcor') if 'BairroOcor' in r.index else r.get('Bairro') if 'Bairro' in r.index else None,
                    'raw': r.to_dict(),
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


def main():
    polys = load_polygons()
    print(f'Loaded {len(polys)} polygons (candidates)')
    occ = load_occurrences()
    print(f'Loaded {len(occ)} occurrences with coordinates')

    # load static bairro centroids for fallback
    cent_file = DATA_STATIC / 'fortaleza_bairros_coords.json'
    centroids = {}
    if cent_file.exists():
        centroids = json.load(open(cent_file, 'r', encoding='utf-8'))
        centroids = {normalize(k): tuple(v) for k, v in centroids.items()}

    assigned = []
    for _, r in occ.iterrows():
        lat = r['lat']; lon = r['lon']
        pt = Point(lon, lat)
        found = None
        # find polygon containing point
        for p in polys:
            try:
                if p['geom'].contains(pt) or p['geom'].covers(pt):
                    found = p
                    break
            except Exception:
                continue
        bairro_assigned = None
        if found:
            bairro_assigned = found['name']
        else:
            # try text bairro match to centroid fallback
            btxt = normalize(r.get('bairro_text'))
            if btxt and btxt in centroids:
                bairro_assigned = btxt
        assigned.append({
            'source': r.get('source'),
            'date': r.get('date'),
            'lat': lat,
            'lon': lon,
            'bairro_assigned': bairro_assigned,
            'bairro_text': r.get('bairro_text'),
        })

    adf = pd.DataFrame(assigned)
    out_occ = OUTPUTS / 'occurrences_with_bairro_geo.csv'
    adf.to_csv(out_occ, index=False)
    print('Wrote', out_occ)

    # produce daily counts per bairro (by date)
    # normalize date to YYYY-MM-DD where possible
    def parse_date(x):
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d'):
            try:
                return datetime.strptime(str(x), fmt).date()
            except Exception:
                continue
        try:
            return pd.to_datetime(x).date()
        except Exception:
            return None

    adf['date_norm'] = adf['date'].apply(parse_date)
    counts = adf[adf['bairro_assigned'].notna()].groupby(['bairro_assigned', 'date_norm']).size().reset_index(name='count')
    out_counts = OUTPUTS / 'bairro_daily_counts.csv'
    counts.to_csv(out_counts, index=False)
    print('Wrote', out_counts)

    # write reference fence: polygons that matched known assigned bairros
    features = []
    seen = set()
    for p in polys:
        nm = p['name']
        if nm in adf['bairro_assigned'].values:
            if nm in seen:
                continue
            seen.add(nm)
            features.append({
                'type': 'Feature',
                'properties': {'bairro_normalized': nm},
                'geometry': mapping(p['geom'])
            })
    fence = {'type': 'FeatureCollection', 'features': features}
    out_fence = OUTPUTS / 'fortaleza_bairros_fence.geojson'
    json.dump(fence, open(out_fence, 'w', encoding='utf-8'), ensure_ascii=False)
    print('Wrote', out_fence)


if __name__ == '__main__':
    main()
