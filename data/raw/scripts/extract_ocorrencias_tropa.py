import csv
import json
import re
import os
import unicodedata
from datetime import datetime

# --- GESTÃO DE CAMINHOS ---
# Garante que aponta para a raiz do repositório, saindo de tests/
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_DIR)

ARQUIVO_ENTRADA = os.path.join(DATA_RAW_DIR, 'ocorrencias_tropa.json')
ARQUIVO_SAIDA = os.path.join(DATA_RAW_DIR, 'ocorrencias_tropa_limpo_fortaleza.csv')

SECTION_KEYWORDS = (
    'TURNO', 'DATA', 'HORA', 'PELOTAO', 'CIA', 'BTL', 'EQUIPE', 'PATRULHA',
    'COMPOSICAO', 'FICHA', 'LOCAL', 'NATUREZA', 'CONDUTOR', 'TESTEMUNHA',
    'ACUSADO', 'VITIMA', 'ARMA', 'MUNICAO', 'VEICULO', 'OBJETO', 'QUANTIA',
    'DROGA', 'DELEGACIA', 'DELEGADO', 'PROCEDIMENTO', 'NARRATIVA', 'HTS'
)
WEAPON_TERMS = ('PISTOLA', 'REVOLVER', 'ESPINGARDA', 'FUZIL', 'GARRUCHA', 'RIFLE', 'CARABINA','SMT','PT','CT','GLOCK', 'CTT')
VEHICLE_TERMS = ('MOTOCICLETA', 'MOTO', 'VEICULO', 'CARRO', 'AUTOMOVEL', 'CAMINHONETE', 'ONIX', 'HONDA', 'TOYOTA', 'CHEVROLET', 'FIAT', 'VW', 'VOLKSWAGEN')
DRUG_GROUPS = {
    'MACONHA': ('MACONHA', 'SKANK'),
    'COCAINA': ('COCAINA',),
    'CRACK': ('CRACK',),
}
COUNT_UNITS = ('PINO', 'PINOS', 'PAPEL', 'PAPEIS', 'PAPELOTE', 'PAPELOTES', 'TROUXA', 'TROUXAS', 'TROUXINHA', 'TROUXINHAS', 'PEDRA', 'PEDRAS', 'BALINHA', 'BALINHAS', 'TABLETE', 'TABLETES', 'PORCAO', 'PORCOES', 'PEDACO', 'PEDACOS')
NUMBER_WORDS = {
    'UM': 1,
    'UMA': 1,
    'DOIS': 2,
    'DUAS': 2,
    'TRES': 3,
    'QUATRO': 4,
    'CINCO': 5,
    'SEIS': 6,
    'SETE': 7,
    'OITO': 8,
    'NOVE': 9,
    'DEZ': 10,
}

def normalize_string(value):
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize('NFD', value)
    normalized = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n').replace('\t', ' ')
    return normalized.upper().strip()

def compact_spaces(value):
    return re.sub(r'\s+', ' ', value).strip()

def parse_decimal(value):
    cleaned = value.strip().replace(' ', '')
    if ',' in cleaned and '.' in cleaned:
        if cleaned.rfind(',') > cleaned.rfind('.'):
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '.')
    return float(cleaned)


def convert_to_grams(amount, unit):
    unit = normalize_string(unit)
    if unit.startswith('KG') or unit.startswith('QUILO') or unit.startswith('KILO'):
        return amount * 1000.0
    if unit.startswith('MG'):
        return amount / 1000.0
    return amount


def cleanup_field(value):
    cleaned = normalize_string(value)
    cleaned = re.sub(r'[*_`]', ' ', cleaned)
    cleaned = re.split(r'\b(?:BAIRRO|DISTRITO|LOCALIDADE|MUNICIPIO|CIDADE|LAT|LONG|CEP|NARRATIVA|CONDUTOR)\b', cleaned)[0]
    cleaned = re.sub(r'\b/\s*CE\b', ' ', cleaned)
    cleaned = re.sub(r'\bCE\b$', ' ', cleaned)
    cleaned = re.sub(r'[^A-Z0-9/\- ]', ' ', cleaned)
    return compact_spaces(cleaned)


def section_header_number(line):
    candidate = compact_spaces(normalize_string(line))
    match = re.match(r'^\*?\s*(\d{1,2})\s*(?:[-–])?\s*(.*)$', candidate)
    if not match:
        return None
    section_number = int(match.group(1))
    if section_number > 29:
        return None
    remainder = match.group(2).strip()
    if not remainder:
        return None
    if any(keyword in remainder for keyword in SECTION_KEYWORDS):
        return f'{section_number:02d}'
    return None


def extract_sections(texto_bruto):
    sections = {}
    current_section = None
    for line in normalize_string(texto_bruto).split('\n'):
        header = section_header_number(line)
        if header:
            current_section = header
            sections.setdefault(current_section, [])
        if current_section:
            sections[current_section].append(line)
    return {section: '\n'.join(lines).strip() for section, lines in sections.items()}


def extract_label_value(text, labels):
    for label in labels:
        match = re.search(rf'\b{label}\b\s*:?\s*([^\n]+)', text)
        if match:
            value = cleanup_field(match.group(1))
            if value:
                return value
    return ''


def extract_city(sections, fallback_text):
    for section_key in ('08', '09'):
        section_text = sections.get(section_key, '')
        city = extract_label_value(section_text, ('MUNICIPIO', 'MUNICIPIO', 'CIDADE'))
        if city:
            return city.split('/')[0].strip()
    return extract_label_value(normalize_string(fallback_text), ('MUNICIPIO', 'CIDADE'))


def extract_bairro(sections, fallback_text):
    for section_key in ('08', '09'):
        section_text = sections.get(section_key, '')
        bairro = extract_label_value(section_text, ('BAIRRO',))
        if bairro:
            return bairro
    bairro = extract_label_value(normalize_string(fallback_text), ('BAIRRO',))
    return bairro or 'DESCONHECIDO'


def section_title(section_text):
    lines = [compact_spaces(line) for line in normalize_string(section_text).split('\n')]
    if not lines:
        return ''
    return re.sub(r'^\*?\s*\d{1,2}\s*(?:[-–])?\s*', '', lines[0]).strip(' :*-')


def strip_section_heading(section_text):
    lines = [compact_spaces(line) for line in normalize_string(section_text).split('\n')]
    if not lines:
        return ''
    first_line = section_title(section_text)
    remainder = []
    if ':' in first_line:
        after_colon = first_line.split(':', 1)[1].strip()
        if after_colon:
            remainder.append(after_colon)
    for line in lines[1:]:
        cleaned = line.strip(' *:-')
        if cleaned:
            remainder.append(cleaned)
    return compact_spaces(' '.join(remainder))


def find_section_by_title(sections, keywords):
    for section_text in sections.values():
        title = section_title(section_text)
        if all(keyword in title for keyword in keywords):
            return section_text
    return ''


def has_vehicle_keyword(text):
    normalized = normalize_string(text)
    return any(re.search(rf'\b{term}\b', normalized) for term in VEHICLE_TERMS)


def extract_natureza(sections, fallback_text):
    section_text = find_section_by_title(sections, ('NATUREZA',)) or sections.get('10', '')
    if section_text:
        natureza = strip_section_heading(section_text)
        natureza = re.sub(r'^\(?TIPO/ART\.?\)?[ :]+', '', natureza)
        natureza = re.sub(r'^(TIPO/ART\.?|TIPO|ART\.?)[ :]+', '', natureza)
        if natureza and not re.fullmatch(r'\d+\s*-?', natureza) and 'NATUREZA DA OCORRENCIA' not in natureza and 'DA OCORRENCIA TIPO/ART' not in natureza:
            return natureza
    match = re.search(r'(?mi)^.*NATUREZA[^\n:]*:[ \t]*([^\n]+)$', normalize_string(fallback_text))
    if match:
        natureza = cleanup_field(match.group(1))
        natureza = re.sub(r'^\(?TIPO/ART\.?\)?[ :]+', '', natureza)
        if natureza and not re.fullmatch(r'\d+\s*-?', natureza) and 'CONDUTOR' not in natureza:
            return natureza
    return 'NAO INFORMADA'


def count_keyword_items(text, keywords):
    normalized = normalize_string(text)
    total = 0
    for keyword in keywords:
        for number in re.findall(rf'\b(\d+)\s+{keyword}\b', normalized):
            total += int(number)
        for word, value in NUMBER_WORDS.items():
            total += len(re.findall(rf'\b{word}\s+{keyword}\b', normalized)) * value
    if total:
        return total
    occurrences = sum(len(re.findall(rf'\b{keyword}\b', normalized)) for keyword in keywords)
    if occurrences:
        return occurrences
    return 1 if 'ARMA DE FOGO' in normalized else 0


def extrair_qtd_armas(sections, fallback_text):
    blocks = [sections.get(section_key, '') for section_key in ('16', '13', '19', '20', '28')]
    counts = [count_keyword_items(block, WEAPON_TERMS) for block in blocks if block]
    if counts:
        return max(counts)
    return count_keyword_items(fallback_text, WEAPON_TERMS)


def extract_drug_fragments(block, aliases):
    normalized = normalize_string(block)
    fragments = []
    for alias in aliases:
        fragments.extend(
            match.group(0)
            for match in re.finditer(
                rf'{alias}.*?(?=(?:MACONHA|COCAINA|CRACK|SKANK|OUTROS TIPOS|DELEGACIA|NARRATIVA|$))',
                normalized,
                flags=re.DOTALL,
            )
        )
    return fragments


def extract_weight_and_item_count(fragment):
    normalized = normalize_string(fragment)
    weights = re.findall(r'(\d+(?:[.,]\d+)?)\s*(KG|KILO(?:S)?|QUILO(?:S)?|G|GRAMA(?:S)?|MG)', normalized)
    if weights:
        grams = sum(convert_to_grams(parse_decimal(amount), unit) for amount, unit in weights)
        return grams, 0.0

    count_pattern = '|'.join(COUNT_UNITS)
    counts = re.findall(rf'(\d+)\s*(?:X\s*)?(?:{count_pattern})\b', normalized)
    if counts:
        return 0.0, float(sum(int(value) for value in counts))
    return 0.0, 0.0


def extrair_qtd_drogas(sections):
    candidate_blocks = [sections.get(section_key, '') for section_key in ('23', '12', '13', '20', '28')]
    totals_by_group = {}
    for group_name, aliases in DRUG_GROUPS.items():
        block_totals = []
        for block in candidate_blocks:
            if not block:
                continue
            fragments = extract_drug_fragments(block, aliases)
            if not fragments:
                continue
            grams_total = 0.0
            item_total = 0.0
            for fragment in fragments:
                grams, items = extract_weight_and_item_count(fragment)
                grams_total += grams
                item_total += items
            if grams_total or item_total:
                block_totals.append((grams_total, item_total))
        if not block_totals:
            totals_by_group[group_name] = (0.0, 0.0)
            continue
        max_grams = max(grams for grams, _ in block_totals)
        if max_grams > 0:
            totals_by_group[group_name] = (max_grams, 0.0)
        else:
            totals_by_group[group_name] = (0.0, max(items for _, items in block_totals))
    total_grams = round(sum(grams for grams, _ in totals_by_group.values()), 2)
    total_items = int(sum(items for _, items in totals_by_group.values()))
    return total_grams, total_items


def vehicle_blocks(sections):
    blocks = []
    for section_key in ('15', '18', '19', '20'):
        block = sections.get(section_key, '')
        if not block:
            continue
        normalized = normalize_string(block)
        body = strip_section_heading(block)
        body = re.sub(r'\b(MARCA|MODELO|COR|ANO|PLACA|RENAVAM|CHASSI|MUNICIPIO|LOGRADOURO)\b\s*:?', ' ', body)
        body = compact_spaces(body)
        has_real_content = bool(body)
        if has_real_content and ('VEICULO' in normalized or 'PLACA' in normalized or has_vehicle_keyword(normalized)):
            blocks.append(normalized)
    return blocks


def count_vehicle_mentions(block):
    plates = set(re.findall(r'\b[A-Z]{3}[- ]?\d[A-Z0-9]\d{2}\b|\b[A-Z]{3}[- ]?\d{4}\b', block))
    if plates:
        return len(plates)

    total = 0
    for term in VEHICLE_TERMS:
        total += sum(int(value) for value in re.findall(rf'\b(\d+)\s+{term}\b', block))
        total += len(re.findall(rf'\b(?:UM|UMA)\s+{term}\b', block))
    if total:
        return total
    return 1 if has_vehicle_keyword(block) else 0


def extrair_qtd_veiculos(sections):
    blocks = vehicle_blocks(sections)
    if not blocks:
        return 0
    return max(count_vehicle_mentions(block) for block in blocks)


def carregar_registros(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as json_file:
        conteudo = json_file.read()

    try:
        dados = json.loads(conteudo)
    except json.JSONDecodeError as erro:
        conteudo_limpo = re.sub(r',\s*([}\]])', r'\1', conteudo)
        try:
            dados = json.loads(conteudo_limpo)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Falha ao ler o arquivo JSON '{arquivo}'. Verifique se o export veio bem formatado."
            ) from erro

    if isinstance(dados, dict):
        for chave in ('data', 'rows', 'records', 'result'):
            if chave in dados:
                dados = dados[chave]
                break

    if isinstance(dados, dict):
        return list(dados.values())

    if isinstance(dados, list):
        return dados

    raise RuntimeError(f"Formato JSON inesperado em '{arquivo}'.")


def processar_granular():
    resumo_final = []

    registros = carregar_registros(ARQUIVO_ENTRADA)

    for row in registros:
        ocorrencia = str(row.get('ocorrencia', ''))
        sections = extract_sections(ocorrencia)
        cidade = extract_city(sections, ocorrencia)
        if cidade != 'FORTALEZA':
            continue

        try:
            dt = datetime.strptime(str(row.get('data_registro', '')).strip(), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue

        qtd_drogas_gramas, qtd_drogas_itens = extrair_qtd_drogas(sections)

        resumo_final.append({
            'data': dt.strftime('%Y-%m-%d'),
            'hora': dt.strftime('%H:%M:%S'),
            'bairro': extract_bairro(sections, ocorrencia),
            'cidade': 'FORTALEZA',
            'natureza': extract_natureza(sections, ocorrencia),
            'qtd_armas': extrair_qtd_armas(sections, ocorrencia),
            'qtd_drogas': qtd_drogas_gramas,
            'qtd_drogas_itens': qtd_drogas_itens,
            'qtd_veiculos_apreendidos': extrair_qtd_veiculos(sections),
        })

    resumo_final.sort(key=lambda registro: (registro['data'], registro['hora']))

    # --- LÓGICA DE INCREMENTO (MERGE COM DEDUPLICAÇÃO) ---
    registros_existentes = []
    if os.path.exists(ARQUIVO_SAIDA):
        try:
            with open(ARQUIVO_SAIDA, 'r', encoding='utf-8-sig') as f:
                reader_old = csv.DictReader(f)
                registros_existentes = list(reader_old)
        except Exception as e:
            print(f"Aviso: Não foi possível ler o histórico existente: {e}")

    # Combinar e remover duplicatas baseadas em (data, hora, bairro)
    # Usamos um dicionário indexado pela tupla única para garantir unicidade
    combinados = { (r['data'], r['hora'], r.get('bairro', '')): r for r in registros_existentes }
    
    novos_adicionados = 0
    for r in resumo_final:
        chave = (r['data'], r['hora'], r['bairro'])
        if chave not in combinados:
            combinados[chave] = r
            novos_adicionados += 1
        elif r['natureza'] != 'NAO INFORMADA' and combinados[chave].get('natureza') == 'NAO INFORMADA':
            # Atualiza registro se o novo tiver dados mais completos
            combinados[chave] = r

    lista_final = sorted(combinados.values(), key=lambda x: (x['data'], x['hora']))

    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['data', 'hora', 'bairro', 'cidade', 'natureza', 'qtd_armas', 'qtd_drogas', 'qtd_drogas_itens', 'qtd_veiculos_apreendidos'],
        )
        writer.writeheader()
        writer.writerows(lista_final)

    print(f"Sucesso! Arquivo atualizado: {ARQUIVO_SAIDA}")
    print(f"Registros novos adicionados: {novos_adicionados}")
    print(f"Total de registros acumulados: {len(lista_final)}")
    for registro in resumo_final[:10]:
        print(registro)

if __name__ == "__main__":
    processar_granular()