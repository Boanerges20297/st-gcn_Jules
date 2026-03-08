import pandas as pd
import json
import os
import unicodedata
from datetime import datetime, timedelta

def normalize_text(text):
    if not text or pd.isna(text): return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def generate_critical_streets():
    path_occ = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    output_path = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\ruas_criticas_por_bairro.json'
    
    if not os.path.exists(path_occ):
        print(f"❌ Arquivo {path_occ} não encontrado.")
        return

    print(f"📖 Lendo {path_occ} para extração de logradouros...")
    # Carregar dados oficiais (colunas: name para rua, bairro para bairro)
    df = pd.read_csv(path_occ, usecols=['name', 'bairro', 'tipo', 'tipo_evento'], low_memory=False)
    
    # Filtrar apenas CVLIs
    df_cvli = df[df['tipo'].str.lower() == 'cvli'].copy()
    
    print(f"✅ {len(df_cvli)} ocorrências de CVLI encontradas na base oficial.")
    
    # Normalizar
    df_cvli['BairroClean'] = df_cvli['bairro'].apply(normalize_text)
    df_cvli['RuaClean'] = df_cvli['name'].apply(normalize_text)
    
    # Agrupar e contar por Rua dentro do Bairro
    grouped = df_cvli.groupby(['BairroClean', 'RuaClean']).size().reset_index(name='count')
    
    # Filtros de Ruído (Termos que não são nomes de ruas)
    natureza_terms = ['HOMICIDIO', 'BALA', 'FOGO', 'LESAO', 'MORTE', 'CADAVER', 'LATROCINIO', 'FEMINICIDIO', 'EXECUCAO', 'TIRO', 'ACHADO']
    
    streets_map = {}
    for bairro in grouped['BairroClean'].unique():
        if not bairro or len(bairro) < 2: continue
        
        bairro_data = grouped[grouped['BairroClean'] == bairro].sort_values('count', ascending=False).head(15)
        ruas = bairro_data['RuaClean'].tolist()
        
        ruas_clean = []
        for r in ruas:
            r_strip = str(r).strip()
            if len(r_strip) <= 4: continue
            if any(term in r_strip for term in natureza_terms): continue
            ruas_clean.append(r_strip)
            
        if ruas_clean:
            streets_map[bairro] = ", ".join(ruas_clean[:5])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(streets_map, f, ensure_ascii=False, indent=4)
    
    print(f"🚀 Inteligência de ruas (Base Oficial) gerada: {len(streets_map)} bairros cobertos.")

if __name__ == "__main__":
    generate_critical_streets()
