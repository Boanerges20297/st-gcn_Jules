import json
import csv

path_json = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
path_out = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\analise_cvli_fortaleza_completa.csv'

with open(path_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

import re

def extract_date(d):
    if not d:
        return None
    if isinstance(d, dict):
        d = d.get('data')
    if isinstance(d, list):
        d = d[0] if d else None
    
    if not d or not isinstance(d, str):
        return None
        
    # Busca padrao YYYY-MM-DD
    match = re.search(r'(\d{4}-\d{2}-\d{2})', d)
    return match.group(1) if match else None

# Identificar todas as datas únicas no dataset inteiro
all_dates = set()
for item in data:
    d = extract_date(item.get('data'))
    if d:
        all_dates.add(d)

valid_dates = sorted(list(all_dates))
total_days_analyzed = len(valid_dates)
min_date = valid_dates[0] if valid_dates else "N/A"
max_date = valid_dates[-1] if valid_dates else "N/A"

# Filtrar CVLI em Fortaleza
cvli_fortaleza = [item for item in data if (str(item.get('cidade')).upper() == 'FORTALEZA') and item.get('tipo') == 'cvli']
import glob
import os

def load_faction_mapping():
    base_path = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\outputs'
    mappings = {}
    for fpath in glob.glob(os.path.join(base_path, '*_territory_from_kml.json')):
        faction_name = os.path.basename(fpath).split('_')[0].upper()
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    for area in content:
                        mappings[str(area).upper()] = faction_name
                elif isinstance(content, dict):
                    # Se for dicionario com bairros como chaves
                    for area in content.keys():
                        mappings[str(area).upper()] = faction_name
        except:
            continue
    return mappings

import unicodedata

def normalize_name(name):
    if not name:
        return ""
    # Remove acentos e converte para maiusculo
    n = str(name).upper().strip()
    n = "".join(c for c in unicodedata.normalize('NFD', n) if unicodedata.category(c) != 'Mn')
    return n

faction_map = load_faction_mapping()
# Normalizar tambem as chaves do mapa de faccoes
faction_map = {normalize_name(k): v for k, v in faction_map.items()}

# Identificar anos presentes
years = sorted(list(set(d[:4] for d in valid_dates if d)))

# Estatísticas por bairro e por ano
stats = {}
for item in cvli_fortaleza:
    b = normalize_name(item.get('bairro'))
    if b:
        if b not in stats:
            stats[b] = {'total': 0, 'by_year': {y: 0 for y in years}, 'days': set(), 'faction': faction_map.get(b, 'N/D')}

        d = extract_date(item.get('data'))
        if d and len(d) >= 10:
            year = d[:4]
            if year in stats[b]['by_year']:
                stats[b]['by_year'][year] += 1
            stats[b]['days'].add(d[:10])

        stats[b]['total'] += 1

dias_2026 = len([d for d in valid_dates if d.startswith('2026')])
fator_projecao_2026 = 365 / dias_2026 if dias_2026 > 0 else 1

# Ordenar por contagem total decrescente
ranked = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)

with open(path_out, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['RESUMO DO DATASET'])
    writer.writerow(['Periodo Analisado', f'{min_date} ate {max_date}'])
    writer.writerow(['Total de Dias Únicos no Periodo', total_days_analyzed])
    writer.writerow([])

    # Header Completo: Metricas Gerais + Dados Brutos por Ano
    header = [
        'Bairro', 
        'Facção Predominante', 
        'Total Geral CVLI', 
        'Dias com Ocorrencia', 
        '% do Periodo Total', 
        'Periodicidade (1 a cada X dias)'
    ]
    
    for y in years:
        if y == '2026':
            header.append(f'Total {y} (Bruto Parcial)')
            header.append(f'Projeção Final {y}')
        else:
            header.append(f'Total {y} (Bruto)')

    writer.writerow(header)

    for b, s in ranked:
        total_cvli = s['total']
        dias_com_ocorrencia = len(s['days'])
        frequencia_periodo = round((dias_com_ocorrencia / total_days_analyzed) * 100, 2) if total_days_analyzed > 0 else 0
        periodicidade = round(total_days_analyzed / total_cvli, 1) if total_cvli > 0 else 0
        
        # Colunas base
        row = [
            b, 
            s['faction'], 
            total_cvli, 
            dias_com_ocorrencia, 
            f'{frequencia_periodo}%', 
            f'{periodicidade} dias'
        ]

        # Colunas anuais com numeros brutos
        for y in years:
            count_year = s['by_year'][y]
            row.append(count_year)
            if y == '2026':
                projecao = round(count_year * fator_projecao_2026, 1)
                row.append(projecao)

        writer.writerow(row)

print(f"Análise concluída.")
print(f"Total de dias analisados no dataset: {total_days_analyzed} ({min_date} a {max_date})")
print(f"Arquivo gerado: {path_out}")
