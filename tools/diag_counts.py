import json
import pandas as pd
import os
import unicodedata
import re

DATA_DIR = 'data/raw'
BAIRROS_FILE = os.path.join(DATA_DIR, 'bairros_centros_latlong.json')
OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'dados_status_ocorrencias_gerais_ENRIQUECIDO.json')

def normalize_text(text):
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def clean_name(n):
    n = normalize_text(n)
    if 'CONJUNTO CEARA' in n: n = 'CONJUNTO CEARA'
    if 'PRAIA DO FUTURO' in n: n = 'PRAIA DO FUTURO'
    if 'VILA MANOEL SATIRO' in n: n = 'MANOEL SATIRO'
    n = re.sub(r'\s+[IVXLCDM]+$', '', n)
    n = re.sub(r'\s+\d+$', '', n)
    return n.strip()

def diag():
    if not os.path.exists(OCORRENCIAS_FILE):
        print(f"File not found: {OCORRENCIAS_FILE}")
        return

    with open(OCORRENCIAS_FILE, 'r', encoding='utf-8') as f:
        occ_data = json.load(f)
    
    clean_records = []
    for item in occ_data:
        if not isinstance(item, dict): continue
        if 'data' in item and isinstance(item['data'], dict): 
            item_data = item['data']
        else:
            item_data = item
        
        clean_item = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in item_data.items()}
        clean_records.append(clean_item)
    
    df = pd.DataFrame(clean_records)
    if 'data' not in df.columns:
        print("Coluna 'data' nao encontrada.")
        return
        
    df['data'] = pd.to_datetime(df['data'].astype(str), errors='coerce')
    df = df.dropna(subset=['data'])
    
    cvli = df[df['tipo'].fillna('').astype(str).str.lower() == 'cvli'].copy()
    cvli['b_clean'] = cvli.apply(lambda r: clean_name(r.get('bairro_geo') or r.get('municipio') or r.get('bairro')), axis=1)
    
    counts = cvli.groupby('b_clean').size().sort_values(ascending=False)
    
    print(f"Total de CVLIs na base: {len(cvli)}")
    print("Top 20 Geral:")
    print(counts.head(20))
    
    with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
        bairros_list = json.load(f)
    fortaleza_bairros = [clean_name(name) for name, info in bairros_list.items() if info.get('regiao') == 'fortaleza']
    
    ftz_counts = counts[counts.index.isin(fortaleza_bairros)]
    print("\nTop 20 Bairros de FORTALEZA:")
    print(ftz_counts.head(20))
    
    min_d, max_d = df['data'].min(), df['data'].max()
    months = (max_d - min_d).days / 30.0
    print(f"\nPeriodo: {min_d.date()} ate {max_d.date()} ({months:.1f} meses)")
    
    threshold_absolute = 2.0 * months
    passed = ftz_counts[ftz_counts >= threshold_absolute]
    print(f"\nBairros de Fortaleza que passam de 2.0 crimes/mes (>= {threshold_absolute:.1f} total): {len(passed)}")
    for b, c in passed.items():
        print(f" - {b}: {c} crimes ({c/months:.2f}/mes)")

if __name__ == '__main__':
    diag()
