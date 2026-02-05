"""Match intelligence geojson bairro names to official node bairros and produce mapping.

Outputs:
 - outputs/intel_bairro_mapping.csv
 - outputs/intel_bairro_details.json
"""
import os
import re
import json
import unicodedata
from collections import defaultdict
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_CANDIDATES = [
    os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json'),
    os.path.join(BASE_DIR, 'outputs', 'fortaleza_bairros_fence.geojson'),
    os.path.join(BASE_DIR, 'outputs', 'nodes.geojson')
]
INTEL_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia')
OUT_CSV = os.path.join(BASE_DIR, 'outputs', 'intel_bairro_mapping.csv')
OUT_JSON = os.path.join(BASE_DIR, 'outputs', 'intel_bairro_details.json')


def norm(s):
    if s is None:
        return ''
    s = str(s)
    s = s.strip()
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def find_nodes_file():
    for p in NODES_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def extract_bairro_from_description(desc):
    if not desc:
        return None
    # desc may be dict with '@type','value' as in examples
    if isinstance(desc, dict) and 'value' in desc:
        text = desc['value']
    else:
        text = str(desc)
    # Look for 'Bairro:' or 'bairro:' patterns
    m = re.search(r'Bairro:\s*([^<\\n]+)', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: look for '-> Bairro: NAME' or '-> Bairro:NAME'
    m = re.search(r'->\s*Bairro:\s*([^<\\n]+)', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: last parenthesis content containing known bairros
    return None


def main():
    nodes_file = find_nodes_file()
    if nodes_file is None:
        print('No nodes polygon/geojson file found in candidates.')
        return
    print('Using nodes file:', nodes_file)
    try:
        # bairros_centros_latlong.json is a plain JSON mapping; handle separately
        if nodes_file.lower().endswith('.json') and 'bairros_centros_latlong' in nodes_file:
            import json as _json
            with open(nodes_file, 'r', encoding='utf-8') as fh:
                raw = _json.load(fh)
            records = []
            for name, info in raw.items():
                lat = info.get('lat') or info.get('latitude')
                lon = info.get('long') or info.get('lon') or info.get('longitude')
                records.append({'bairro_node': name, 'bairro_norm': norm(name), 'geometry': Point(lon, lat)})
            nodes = gpd.GeoDataFrame(records, geometry='geometry', crs='EPSG:4326')
        else:
            nodes = gpd.read_file(nodes_file)
    except Exception as e:
        print('Failed to read nodes file:', e)
        return
    # find a column with bairro name
    candidates = ['bairro', 'Bairro', 'BAIRRO', 'name', 'NAME', 'NOME', 'nome', 'bairro_name']
    bairro_col = next((c for c in candidates if c in nodes.columns), None)
    if bairro_col is None:
        # create a synthetic bairro column from index
        nodes['bairro_node'] = nodes.index.astype(str)
        bairro_col = 'bairro_node'
    else:
        nodes['bairro_node'] = nodes[bairro_col].astype(str)
    nodes['bairro_norm'] = nodes['bairro_node'].apply(norm)

    # iterate intelligence files
    mapping = defaultdict(set)  # bairro_norm -> set of factions
    details = defaultdict(list)
    if not os.path.exists(INTEL_DIR):
        print('No inteligencia dir found at', INTEL_DIR)
    else:
        for fn in sorted(os.listdir(INTEL_DIR)):
            if not fn.lower().endswith('.geojson'):
                continue
            path = os.path.join(INTEL_DIR, fn)
            try:
                g = gpd.read_file(path)
            except Exception as e:
                print('Failed to read', fn, e)
                continue
            for idx, row in g.iterrows():
                props = row.get('properties') if 'properties' in row else None
                pname = None
                pdesc = None
                if isinstance(props, dict):
                    pname = props.get('name') or props.get('Name') or props.get('NAME')
                    pdesc = props.get('description') or props.get('Descricao') or props.get('descricao')
                else:
                    # sometimes properties are columns
                    for k in ['name','Name','NAME','description','desc','Descricao','descricao']:
                        if k in g.columns:
                            if pname is None and k.lower().startswith('name'):
                                pname = row.get(k)
                            if pdesc is None and k.lower().startswith('descr'):
                                pdesc = row.get(k)
                # extract candidate bairro names
                candidates_bairros = set()
                if pname:
                    # sometimes name contains bairro in parentheses: 'Areia Grossa (Complexo do Pirambu)'
                    # try to extract trailing part or parenthesis
                    pb = re.sub(r"\(.*?\)", "", str(pname)).strip()
                    candidates_bairros.add(pb)
                    # also add full name
                    candidates_bairros.add(str(pname))
                bd = extract_bairro_from_description(pdesc)
                if bd:
                    candidates_bairros.add(bd)
                # normalize and match
                matched = False
                for cb in candidates_bairros:
                    cbn = norm(cb)
                    if not cbn:
                        continue
                    # direct match to nodes bairro_norm
                    hits = nodes[nodes['bairro_norm'] == cbn]
                    if not hits.empty:
                        for bn in hits['bairro_node'].unique():
                            mapping[cbn].add(fn.replace('.geojson',''))
                            details[cbn].append({'file': fn, 'raw_name': cb, 'matched_bairro': bn})
                            matched = True
                if not matched:
                    # try fuzzy substring match: check if cb appears within any node bairro
                    for cb in list(candidates_bairros):
                        cbn = norm(cb)
                        if not cbn:
                            continue
                        for _, nrow in nodes.iterrows():
                            if cbn in nrow['bairro_norm'] or nrow['bairro_norm'] in cbn:
                                mapping[cbn].add(fn.replace('.geojson',''))
                                details[cbn].append({'file': fn, 'raw_name': cb, 'matched_bairro': nrow['bairro_node']})
                                matched = True
                                break
                        if matched:
                            break
    # prepare outputs
    out_rows = []
    for k, v in mapping.items():
        out_rows.append({'bairro_norm': k, 'bairros_sources': list(v), 'count': len(v)})
    df = pd.DataFrame(out_rows).sort_values('count', ascending=False)
    os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(details, fh, ensure_ascii=False, indent=2)
    print('Saved mapping to', OUT_CSV)
    print('Saved details to', OUT_JSON)
    print('Top matches:')
    print(df.head(20).to_string(index=False))

if __name__ == '__main__':
    main()
