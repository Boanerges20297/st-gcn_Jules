import json
import pandas as pd
import os
import unicodedata
import re

RMF_OFFICIAL = [
    'AQUIRAZ', 'BEBERIBE', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 
    'GUAIUBA', 'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 
    'PACATUBA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE', 
    'SAO LUIS DO CURU', 'TRAIRI'
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

def generate_conference_list():
    p_occ = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
    with open(p_occ, 'r', encoding='utf-8') as f:
        occ_raw = json.load(f)
    
    clean_occ = []
    for o in occ_raw:
        if str(o.get('tipo', '')).lower() != 'cvli': continue
        loc = o.get('bairro_geo') or o.get('bairro') or o.get('municipio') or o.get('cidade')
        if loc: clean_occ.append({'loc': clean_name(loc)})
    
    occ_df = pd.DataFrame(clean_occ)
    months = 1000 / 30.0
    cvli_counts = occ_df.groupby('loc').size()

    p_nodes = 'data/raw/bairros_centros_latlong.json'
    with open(p_nodes, 'r', encoding='utf-8') as f:
        nodes_raw = json.load(f)
    
    RMF_CORRECTIONS = ['GUADALAJARA', 'GUAJERU', 'INDUSTRIAL', 'IPARANA', 'MARECHAL RONDON', 'PARQUE ALBANO', 'PARQUE SOLEDADE', 'ALTO ALEGRE']
    REMOVAL_MARK = ['DIF']

    records = []
    for name, info in nodes_raw.items():
        c_name = clean_name(name)
        if c_name in REMOVAL_MARK: continue
        
        reg = info.get('regiao', 'interior').lower()
        if c_name in RMF_CORRECTIONS or c_name in RMF_OFFICIAL:
            reg = 'rmf'
        elif reg == 'rmf' and c_name not in RMF_OFFICIAL: continue 
        
        records.append({'name': c_name, 'regiao': reg, 'faction': info.get('faction', 'NEUTRO').upper()})
    
    df_nodes = pd.DataFrame(records).drop_duplicates(subset=['name'])
    
    output = []
    output.append("="*60)
    output.append("AUDITORIA DE NOS - FILTRO JULES (VERSAO CORRIGIDA)")
    output.append("Filtro: FTZ >= 1.0/m, INT >= 1.0/m, RMF ALL, FAC ALWAYS")
    output.append("="*60 + "\n")

    summary = {'fortaleza': [], 'rmf': [], 'interior': []}
    for _, row in df_nodes.iterrows():
        name = row['name']
        reg = row['regiao']
        has_f = row['faction'] != 'NEUTRO'
        c_per_m = cvli_counts.get(name, 0) / months
        
        keep = False
        if has_f: keep = True
        elif reg == 'rmf' and name in RMF_OFFICIAL: keep = True
        elif reg == 'fortaleza' and c_per_m >= 1.0: keep = True
        elif reg == 'interior' and c_per_m >= 1.0: keep = True
        
        if keep:
            line = f"{name.ljust(30)} | CVLI/m: {c_per_m:>4.1f} | Fac: {'SIM' if has_f else '---'}"
            summary[reg].append(line)

    for reg_name in ['fortaleza', 'rmf', 'interior']:
        output.append("--- " + reg_name.upper() + " (" + str(len(summary[reg_name])) + " nos) ---")
        for line in sorted(summary[reg_name]):
            output.append(line)
        output.append("")

    with open('LISTA_NOS_JULES.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(output))
    print("✅ Nova LISTA_NOS_JULES.txt gerada.")

if __name__ == '__main__':
    generate_conference_list()
