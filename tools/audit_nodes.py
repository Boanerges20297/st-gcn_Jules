import json
import pandas as pd
import os
import unicodedata
import re

# Lista Oficial RMF (18 Cidades sem Fortaleza)
RMF_OFFICIAL = [
    'AQUIRAZ', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 'GUAIUBA', 
    'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 
    'PACATUBA', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE', 'SAO LUIS DO CURU', 
    'TRAIRI', 'PARACURU', 'BEBERIBE'
]

def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def clean_name(n):
    n = normalize_text(n)
    merges = ['CONJUNTO CEARA', 'PRAIA DO FUTURO', 'VILA MANOEL SATIRO', 'ALTO ALEGRE', 'EDSON QUEIROZ', 'JOSE WALTER']
    for m in merges:
        if m in n: return m
    n = re.sub(r'\s+[IVXLCDM]+$', '', n)
    n = re.sub(r'\s+\d+$', '', n)
    return n.strip()

def audit():
    with open('data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.json', 'r', encoding='utf-8') as f:
        occ_raw = json.load(f)
    
    clean_occ = []
    for o in occ_raw:
        loc = o.get('bairro_geo') or o.get('bairro') or o.get('municipio') or o.get('cidade')
        if not loc: continue
        clean_occ.append({
            'tipo': str(o.get('tipo', '')).lower(),
            'loc': clean_name(loc)
        })
    occ_df = pd.DataFrame(clean_occ)
    months = 1000 / 30.0
    counts = occ_df[occ_df['tipo'] == 'cvli'].groupby('loc').size()

    with open('data/raw/bairros_centros_latlong.json', 'r', encoding='utf-8') as f:
        nodes_raw = json.load(f)
    
    records = []
    for name, info in nodes_raw.items():
        c_name = clean_name(name)
        reg = info.get('regiao', 'interior').lower()
        faction = info.get('faction', 'NEUTRO').upper()
        if c_name in RMF_OFFICIAL: reg = 'rmf'
        elif reg == 'rmf' and c_name not in RMF_OFFICIAL: continue 
        records.append({'name': c_name, 'regiao': reg, 'faction': faction})
    
    df_nodes = pd.DataFrame(records).drop_duplicates(subset=['name'])
    results = {'fortaleza': [], 'rmf': [], 'interior': []}
    
    for _, row in df_nodes.iterrows():
        name = row['name']
        reg = row['regiao']
        has_f = row['faction'] != 'NEUTRO'
        c_per_m = counts.get(name, 0) / months
        
        keep = False
        if has_f: keep = True
        elif reg == 'rmf' and name in RMF_OFFICIAL: keep = True
        elif reg == 'fortaleza' and c_per_m >= 1.0: keep = True
        elif reg == 'interior' and c_per_m >= 1.0: keep = True
        
        if keep:
            results[reg].append(f"{name} ({c_per_m:.1f} CVLI/m, {'FAC' if has_f else '---'})")

    for reg_name, items in results.items():
        print("\n--- " + reg_name.upper() + " (" + str(len(items)) + " nos) ---")
        for item in sorted(items):
            print(item)

if __name__ == '__main__':
    audit()
