import json
import csv
import re
import os
import unicodedata
import xml.etree.ElementTree as ET

def normalize_name(name):
    if not name:
        return ""
    # Remove acentos
    n = "".join(c for c in unicodedata.normalize('NFD', str(name)) if unicodedata.category(c) != 'Mn')
    n = n.upper().strip()
    # Remover sufixos de AIS (ex: - AIS 20)
    n = re.sub(r'\s*-\s*AIS\s*\d+', '', n)
    # Remover sufixos de faccao no nome (ex: - CV)
    n = re.sub(r'\s*-\s*(CV|PCC|GDE|TCP|MASSA|OKAIDA).*', '', n)
    # Pegar apenas a primeira parte antes de traços se for um bairro conhecido
    n = n.split(' - ')[0].strip()
    return n

def extract_faction_mapping_from_kml_v3(kml_path):
    mappings = {}
    print(f"Lendo KML (Estrutura de Pastas): {kml_path}")
    
    try:
        # Usar parse completo para lidar com a hierarquia de pastas
        tree = ET.parse(kml_path)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Encontrar todos os Folders
        for folder in root.findall('.//kml:Folder', ns):
            folder_name_elem = folder.find('kml:name', ns)
            if folder_name_elem is None: continue
            
            fname = folder_name_elem.text.upper()
            faction = "N/D"
            
            if "COMANDO VERMELHO" in fname or " CV " in fname:
                faction = "CV"
            elif "TCP" in fname or "GDE" in fname or "GUARDIÕES" in fname:
                faction = "TCP/GDE"
            elif "PRIMEIRO COMANDO" in fname or "PCC" in fname:
                faction = "PCC"
            elif "MASSA" in fname:
                faction = "MASSA"
            elif "OKAIDA" in fname:
                faction = "OKAIDA"
            elif "DISPUTA" in fname:
                faction = "DISPUTA"
            
            if faction == "N/D": continue
            
            # Mapear todos os Placemarks dentro desta pasta
            for pm in folder.findall('.//kml:Placemark', ns):
                pm_name_elem = pm.find('kml:name', ns)
                if pm_name_elem is not None and pm_name_elem.text:
                    orig_name = pm_name_elem.text
                    norm_name = normalize_name(orig_name)
                    # Se ja existir e for DISPUTA, priorizamos DISPUTA. 
                    # Senao, mantemos o que ja foi mapeado.
                    if norm_name not in mappings or faction == "DISPUTA":
                        mappings[norm_name] = faction
                        
        print(f"Mapeamento concluído: {len(mappings)} áreas identificadas.")
    except Exception as e:
        print(f"Erro ao processar KML: {e}")
        
    return mappings

# --- EXECUÇÃO ---

kml_path = r'C:\Users\Boanerges\Downloads\ORCRIMS_2026.kml'
json_path = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
output_csv = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\analise_cvli_fortaleza_completa.csv'
output_xlsx = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\TABELA_ANALISE_CVLI_FORTALEZA.xlsx'

# 1. Mapeamento
faction_map = extract_faction_mapping_from_kml_v3(kml_path)

# 2. Carregar Ocorrências
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def extract_date(d):
    if not d: return None
    if isinstance(d, dict): d = d.get('data')
    if isinstance(d, list): d = d[0] if d else None
    if not d or not isinstance(d, str): return None
    match = re.search(r'(\d{4}-\d{2}-\d{2})', d)
    return match.group(1) if match else None

# 3. Filtrar Fortaleza
cvli_fortaleza = []
all_dates = set()
for item in data:
    d = extract_date(item.get('data'))
    if d: all_dates.add(d)
    if (str(item.get('cidade')).upper() == 'FORTALEZA') and item.get('tipo') == 'cvli':
        cvli_fortaleza.append(item)

valid_dates = sorted(list(all_dates))
total_days_analyzed = len(valid_dates)
years = sorted(list(set(d[:4] for d in valid_dates if d)))

# 4. Estatísticas
stats = {}
for item in cvli_fortaleza:
    b_raw = item.get('bairro')
    if b_raw is None:
        b_raw = "NÃO INFORMADO"
    
    b_norm = normalize_name(b_raw)
    
    if b_norm not in stats:
        # Tentar match de facção
        faction = faction_map.get(b_norm, "N/D")
        if faction == "N/D":
            # Busca parcial (substring)
            for k, v in faction_map.items():
                if b_norm and k and (b_norm in k or k in b_norm):
                    faction = v
                    break
        
        stats[b_norm] = {
            'name_display': str(b_raw).upper(),
            'total': 0, 
            'by_year': {y: 0 for y in years}, 
            'days': set(), 
            'faction': faction
        }
    
    d = extract_date(item.get('data'))
    if d:
        year = d[:4]
        if year in stats[b_norm]['by_year']:
            stats[b_norm]['by_year'][year] += 1
        stats[b_norm]['days'].add(d)
    stats[b_norm]['total'] += 1

dias_2026 = len([d for d in valid_dates if d.startswith('2026')])
fator_projecao_2026 = 365 / dias_2026 if dias_2026 > 0 else 1

ranked = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)

# 5. Salvar CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['RESUMO DO DATASET - FORTALEZA (MAPEAMENTO POR PASTAS KML)'])
    writer.writerow(['Total de Dias Únicos', total_days_analyzed])
    writer.writerow([])
    header = ['Bairro', 'Facção Predominante', 'Total Geral CVLI', 'Dias com Ocorrência', '% do Período Total', 'Periodicidade (1 a cada X dias)']
    for y in years:
        header.append(f'Total {y}')
        if y == '2026': header.append(f'Projeção {y}')
    writer.writerow(header)
    for _, s in ranked:
        total_cvli = s['total']
        dias_occ = len(s['days'])
        freq_perc = round((dias_occ / total_days_analyzed) * 100, 2) if total_days_analyzed > 0 else 0
        periodicidade = round(total_days_analyzed / total_cvli, 1) if total_cvli > 0 else 0
        
        row = [
            s['name_display'], 
            s['faction'], 
            total_cvli, 
            dias_occ, 
            f"{freq_perc}%", 
            f"{periodicidade} dias"
        ]
        for y in years:
            row.append(s['by_year'][y])
            if y == '2026': 
                row.append(round(s['by_year'][y] * fator_projecao_2026, 1))
        writer.writerow(row)

# 6. Salvar Excel (TABELA LIMPA)
import pandas as pd
df_final = pd.read_csv(output_csv, skiprows=3, encoding='utf-8')
df_final.to_excel(output_xlsx, index=False, sheet_name='Analise_CVLI')

print(f"Processo concluído. Arquivos atualizados com facções corrigidas.")
