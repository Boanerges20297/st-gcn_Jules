import json
import csv
import re
import os
import unicodedata
import xml.etree.ElementTree as ET

def normalize_name(name):
    if not name:
        return ""
    n = str(name).upper().strip()
    n = "".join(c for c in unicodedata.normalize('NFD', n) if unicodedata.category(c) != 'Mn')
    # Remover sufixos de cidade ou detalhes entre parenteses/hifens para match mais flexivel
    n = re.sub(r' - .*', '', n)
    n = re.sub(r' \(.*\)', '', n)
    return n.strip()

def extract_faction_mapping_from_kml(kml_path):
    mappings = {}
    print(f"Lendo KML (Estado Inteiro): {kml_path}")
    
    try:
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        context = ET.iterparse(kml_path, events=('end',))
        
        count = 0
        for event, elem in context:
            if elem.tag.endswith('Placemark'):
                count += 1
                name_elem = elem.find('.//kml:name', ns)
                name = name_elem.text if name_elem is not None else ""
                
                faction = "N/D"
                # 1. Tentar achar nos ExtendedData (Metodo mais preciso)
                for data in elem.findall('.//kml:Data', ns):
                    if data.get('name') == 'FACÇÃO':
                        val = data.find('kml:value', ns)
                        if val is not None and val.text:
                            faction = val.text.strip().upper()
                            break
                
                # 2. Se não achou, tentar por palavras-chave no nome/descrição
                if faction == "N/D" or not faction:
                    desc_elem = elem.find('.//kml:description', ns)
                    desc = desc_elem.text if desc_elem is not None else ""
                    full_text = (name + " " + desc).upper()
                    
                    if "COMANDO VERMELHO" in full_text or " CV " in full_text or " CV-" in full_text:
                        faction = "CV"
                    elif "TERCEIRO COMANDO" in full_text or " TCP " in full_text:
                        faction = "TCP"
                    elif "PCC" in full_text or "PRIMEIRO COMANDO" in full_text:
                        faction = "PCC"
                    elif "MASSA" in full_text:
                        faction = "MASSA"
                    elif "GDE" in full_text:
                        faction = "GDE"

                if name and faction and faction != "N/D":
                    norm_name = normalize_name(name)
                    if norm_name:
                        mappings[norm_name] = faction
                
                if count % 10000 == 0:
                    print(f"Processados {count} Placemarks...")
                
                elem.clear() 
    except Exception as e:
        print(f"Erro ao processar KML: {e}")
        
    return mappings

# --- EXECUÇÃO ---

kml_path = r'C:\Users\Boanerges\Downloads\ORCRIMS_2026.kml'
json_path = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
output_csv = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\analise_cvli_fortaleza_completa.csv'

# 1. Mapeamento de Facções
faction_map = extract_faction_mapping_from_kml(kml_path)
print(f"Total de áreas únicas mapeadas no Ceará: {len(faction_map)}")

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
min_date = valid_dates[0]; max_date = valid_dates[-1]
years = sorted(list(set(d[:4] for d in valid_dates if d)))

# 4. Estatísticas
stats = {}
for item in cvli_fortaleza:
    b_raw = item.get('bairro')
    if not b_raw:
        b_raw = "NÃO INFORMADO"
    
    b_norm = normalize_name(b_raw)
    if b_norm not in stats:
        # Tentar busca parcial se não achar exato
        faction = faction_map.get(b_norm, "N/D")
        if faction == "N/D":
            # Busca por substring (ex: 'BARRA DO CEARA' em 'BARRA DO CEARA - SETOR A')
            for k, v in faction_map.items():
                if b_norm in k or k in b_norm:
                    faction = v
                    break
        
        stats[b_norm] = {
            'name_display': b_raw.upper(),
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
    writer.writerow(['RESUMO DO DATASET - FORTALEZA (BASEADO EM MAPEAMENTO ESTADUAL)'])
    writer.writerow(['Periodo Analisado', f'{min_date} ate {max_date}'])
    writer.writerow(['Total de Dias Únicos', total_days_analyzed])
    writer.writerow([])

    header = ['Bairro', 'Facção Predominante (KML 2026)', 'Total Geral CVLI', 'Dias com Ocorrencia', '% do Periodo Total', 'Periodicidade (1 a cada X dias)']
    for y in years:
        header.append(f'Total {y}')
        if y == '2026': header.append(f'Projeção {y}')
    writer.writerow(header)

    for _, s in ranked[:137]: # Limite aproximado de bairros de Fortaleza
        total_cvli = s['total']
        dias_occ = len(s['days'])
        freq = round((dias_occ / total_days_analyzed) * 100, 2)
        per = round(total_days_analyzed / total_cvli, 1)
        
        row = [s['name_display'], s['faction'], total_cvli, dias_occ, f'{freq}%', f'{per} dias']
        for y in years:
            row.append(s['by_year'][y])
            if y == '2026': row.append(round(s['by_year'][y] * fator_projecao_2026, 1))
        writer.writerow(row)

print(f"Análise concluída. Arquivo atualizado: {output_csv}")
